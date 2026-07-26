from loguru import logger
import time
from tqdm import tqdm
from tqdm.auto import tqdm as atqdm
import asyncio
import torch
import uuid

from .config import GenerationConfig
from .info import GenerationType

import logging
from typing import Tuple

try:
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
    import vllm.model_executor.layers.utils
    _VLLM_AVAILABLE = True
except:
    _VLLM_AVAILABLE = False

def get_recent_tokens(prompt_tokens: torch.Tensor, 
                      output_tokens: torch.Tensor,
                      window_size: int,
                      vocab_size: int) -> torch.Tensor:

    num_seqs = prompt_tokens.shape[0]
    device = prompt_tokens.device
    
    recent_tokens = torch.full((num_seqs, window_size), 
                               vocab_size,
                               dtype=torch.long, 
                               device=device)
    
    for seq_idx in range(num_seqs):
        valid_prompt = prompt_tokens[seq_idx][prompt_tokens[seq_idx] != vocab_size]
        
        valid_output = output_tokens[seq_idx][output_tokens[seq_idx] != vocab_size]
        
        all_tokens = torch.cat([valid_prompt, valid_output])
        
        token_count = min(window_size, all_tokens.shape[0])
        if token_count > 0:
            recent_tokens[seq_idx, -token_count:] = all_tokens[-token_count:]
    
    return recent_tokens

def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long,
                             device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask

def apply_penalties(logits: torch.Tensor, prompt_tokens_tensor: torch.Tensor,
                    output_tokens_tensor: torch.Tensor,
                    presence_penalties: torch.Tensor,
                    frequency_penalties: torch.Tensor,
                    repetition_penalties: torch.Tensor,
                    ) -> torch.Tensor:
    rep_window_size = 64
    num_seqs, vocab_size = logits.shape
    
    _, prompt_mask = get_token_bin_counts_and_mask(prompt_tokens_tensor,
                                                  vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs)
    
    if rep_window_size is not None:
        recent_tokens = get_recent_tokens(prompt_tokens_tensor, 
                                         output_tokens_tensor, 
                                         rep_window_size,
                                         vocab_size)
        
        _, rep_mask = get_token_bin_counts_and_mask(recent_tokens,
                                                  vocab_size, num_seqs)
    else:
        rep_mask = prompt_mask | output_mask
    
    repetition_penalties = repetition_penalties.unsqueeze(dim=1).repeat(
        1, vocab_size)
    logits[logits > 0] /= torch.where(rep_mask,
                                     repetition_penalties, 1.0)[logits > 0]
    logits[logits <= 0] *= torch.where(rep_mask,
                                      repetition_penalties, 1.0)[logits <= 0]

    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits

if _VLLM_AVAILABLE:
    vllm.model_executor.layers.utils.apply_penalties = apply_penalties

class VLLMModelBatch:
    def __init__(
            self,
            model_path: str,
            max_seq_length: int,
            additional_model_config: dict = {},
    ) -> None:

        if not _VLLM_AVAILABLE:
            raise ImportError(
                "VLLM python module not found."
                "To use the VLLM you must install it manually."
            )
        
        model_config = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            tokenizer_mode="auto",
            trust_remote_code=True, 
            tensor_parallel_size=additional_model_config.get("vllm_tensor_parallel_size", 1),
            max_model_len=8192, 
            gpu_memory_utilization=additional_model_config.get("vllm_gpu_memory_utilization", 0.9),
            disable_log_stats=True,
        )
        self.model = AsyncLLMEngine.from_engine_args(model_config)

        logging.getLogger("vllm").setLevel(logging.WARNING)

        self.loop = asyncio.new_event_loop()

        self.sem = None

    async def _generate(
            self, 
            input_ids: str, 
            config: GenerationConfig, 
            request_id,
            pbar: atqdm,
            lock: asyncio.Lock
        ):

        async with self.sem:

            sampling_params = SamplingParams(
                temperature=config.sampler_config.temperature,
                repetition_penalty=config.sampler_config.repetition_penalty,
                top_k=config.sampler_config.top_k,
                top_p=config.sampler_config.top_p,
                min_p=config.sampler_config.min_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=config.max_length,
                skip_special_tokens=False
            )

            results_generator = self.model.generate(input_ids, sampling_params, uuid.uuid4().hex)
            tokens = []
            output_text = ""

            async for result in results_generator:
                current_output_text = result.outputs[0].text
                delta_text = current_output_text[len(output_text):]
                output_text = current_output_text

                if not delta_text:
                    await asyncio.sleep(0)
                    continue
                tokens.extend(delta_text)

                async with lock:
                    pbar.update(1)
                    speed = pbar.format_dict.get("rate", 0)
                    rtf = round(1 / (speed / 150), 4) if speed else 0
                    pbar.set_postfix({"RTF": rtf})

            print("\n")
            logger.success(f"Batch {request_id} finished.\n")

            return {"text": "".join(tokens), "id": request_id}
    
    async def _generate_batch_async(self, input_ids: list[str], config: GenerationConfig):
        """Helper coroutine to run multiple _generate tasks in parallel and gather results in order."""
        pbar = atqdm(
            total=None,
        )
        lock = asyncio.Lock()
        self.sem = asyncio.Semaphore(config.max_batch_size)

        tasks = [
            asyncio.create_task(
                self._generate(i, config, idx, pbar, lock)
            )
            for idx, i in enumerate(input_ids)
        ]
        return await asyncio.gather(*tasks)

    def generate_batch(self, input_ids: list[str], config: GenerationConfig):
        output = self.loop.run_until_complete(
            self._generate_batch_async(input_ids, config)
        )
        torch.cuda.empty_cache()
        return output

        