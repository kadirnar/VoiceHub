from loguru import logger
import torch
from tqdm import tqdm
import asyncio
from tqdm.auto import tqdm as atqdm

from .config import GenerationConfig
from .info import GenerationType

try:
    from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
    from exllamav2.generator import (
        ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, 
        ExLlamaV2Sampler, ExLlamaV2DynamicGeneratorAsync, 
        ExLlamaV2DynamicJobAsync
    )
    _EXL2_AVAILABLE = True
except Exception as e:
    _EXL2_AVAILABLE = False

class EXL2Model:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int,
            additional_model_config: dict = {},
    ) -> None:

        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the EXL2 model you must install exllamav2 manually."
            )
        
        config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=1024*32, lazy=False)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        additional_dynamic_generator_config = additional_model_config.get("additional_dynamic_generator_config", {})
        self.generator = ExLlamaV2DynamicGenerator(
            model=self.model, cache=self.cache, tokenizer=self.tokenizer, **additional_dynamic_generator_config
        )

    def generate(self, input_ids: str, config: GenerationConfig):
        if config.generation_type == GenerationType.STREAM:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)
    
    def _generate_stream(self, input_ids: str, config: GenerationConfig):
        gen_settings = ExLlamaV2Sampler.Settings(
            token_repetition_penalty=config.sampler_config.repetition_penalty,
            temperature=config.sampler_config.temperature,
            token_repetition_range=config.sampler_config.repetition_range,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
            mirostat_eta=config.sampler_config.mirostat_eta,
            mirostat_tau=config.sampler_config.mirostat_tau,
            mirostat=config.sampler_config.mirostat,
            **config.additional_gen_config
        )
        tokens = self.tokenizer.encode(input_ids, add_bos=False, add_eos=False, encode_special_tokens=True)
        tokens_size = len(tokens)
        input_size = len(tokens.flatten().tolist())
        job = ExLlamaV2DynamicJob(
            input_ids=tokens,
            max_new_tokens=config.max_length - input_size,
            stop_conditions=[self.tokenizer.eos_token_id],
            gen_settings=gen_settings,
            identifier=1,
            decode_special_tokens=True,
        )
        self.generator.enqueue(job)
        with tqdm(desc="Generating") as pbar:
            while self.generator.num_remaining_jobs() > 0:
                results = self.generator.iterate()
                for result in results:
                    token = result.get("token_ids", None)
                    if token is None:
                        continue
                    pbar.update(1)
                    pbar.set_postfix({"tokens": input_size + tokens_size, "max tokens": config.max_length})
                    for t in token.flatten().tolist():
                        yield t
                        tokens_size += 1

    def _generate(self, input_ids: str, config: GenerationConfig):
        new_tokens = []
        for i in self._generate_stream(input_ids, config):
            new_tokens.append(i)
        return new_tokens
    
class EXL2ModelAsync:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int,
            additional_model_config: dict = {},
    ) -> None:

        if not _EXL2_AVAILABLE:
            raise ImportError(
                "exllamav2 python module not found."
                "To use the EXL2 model you must install exllamav2 manually."
            )

        config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(config)
        self.cache = ExLlamaV2Cache(self.model, max_seq_len=additional_model_config.get("exl2_cache_size", 8192*4), lazy=True)
        self.model.load_autosplit(self.cache, progress=True)
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.loop = asyncio.new_event_loop()
        additional_dynamic_generator_config = additional_model_config.get("additional_dynamic_generator_config", {})
        self.generator = self.loop.run_until_complete(self._create_generator(additional_dynamic_generator_config))
        self.sem = None
    
    async def _create_generator(self, additional_dynamic_generator_config):
        return ExLlamaV2DynamicGeneratorAsync(
            model=self.model, 
            cache=self.cache, 
            tokenizer=self.tokenizer, 
            **additional_dynamic_generator_config
        )

    async def _generate(
            self, 
            input_ids: str, 
            sampling_params, 
            request_id, 
            config: GenerationConfig,
            pbar: atqdm,
            lock: asyncio.Lock
        ):

        async with self.sem:

            tokens = self.tokenizer.encode(input_ids, add_bos=False, add_eos=False, encode_special_tokens=True)
            input_size = len(tokens.flatten().tolist())
            job = ExLlamaV2DynamicJobAsync(
                self.generator,
                input_ids=tokens,
                max_new_tokens=config.max_length - input_size,
                stop_conditions=[self.tokenizer.eos_token_id],
                gen_settings=sampling_params,
                identifier=request_id,
                decode_special_tokens=True,
            )
            text = ""

            async for result in job:
                delta_text = result.get("text")
                if delta_text is None:
                    await asyncio.sleep(0)
                    continue
                    
                text += delta_text

                async with lock:
                    pbar.update(result.get("token_ids").size(1))
                    speed = pbar.format_dict.get("rate", 0)
                    rtf = round(1/(speed/150), 4)
                    pbar.set_postfix({
                        "RTF": rtf,
                    })

            print("\n")
            logger.success(f"Batch {request_id} finished.\n")

            return {"text": text, "id": request_id}
        
    async def _generate_batch_async(self, input_ids: list[str], sampling_params, config: GenerationConfig):
        """Helper coroutine to run multiple _generate tasks in parallel and gather results in order."""
        pbar = atqdm(total=None)
        lock = asyncio.Lock()
        self.sem = asyncio.Semaphore(config.max_batch_size)

        tasks = [
            asyncio.create_task(
                self._generate(i, sampling_params, idx, config, pbar, lock)
            )
            for idx, i in enumerate(input_ids)
        ]
        return await asyncio.gather(*tasks)
    
    def generate_batch(self, input_ids: list[str], config: GenerationConfig):
        sampling_params = ExLlamaV2Sampler.Settings(
            token_repetition_penalty=config.sampler_config.repetition_penalty,
            temperature=config.sampler_config.temperature,
            token_repetition_range=config.sampler_config.repetition_range,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
        )

        output = self.loop.run_until_complete(
            self._generate_batch_async(input_ids, sampling_params, config)
        )
        torch.cuda.empty_cache()
        return output