from loguru import logger
from tqdm import tqdm
from packaging import version
import json
import requests
import time
from tqdm.auto import tqdm as atqdm
import asyncio
import aiohttp
from urllib.parse import urljoin

from .info import GenerationType
from .config import GenerationConfig

class LlamaCPPServerModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_seq_length: int = 4096,
            additional_model_config: dict = {},
            tokenizer = None
    ) -> None:
        self.tokenizer = tokenizer
        
    def generate(self, input_ids: str, config: GenerationConfig):
        if config.generation_type == GenerationType.STREAM:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)

    def _generate_stream(self, input_ids: str, config: GenerationConfig):

        requests_config = {
            "top_p": config.sampler_config.top_p,
            "top_k": config.sampler_config.top_k,
            "min_p": config.sampler_config.top_p,
            "stream": True,
            "repeat_penalty": config.sampler_config.repetition_penalty,
            "temperature": config.sampler_config.temperature,
            "stop": ["<|im_end|>"],
            "n_predict": config.max_length
        }
        requests_config["prompt"] = input_ids
        try:
            url = urljoin(config.server_host, "/completion")
            response = requests.post(
                url,
                json=requests_config,
                stream=True
            )
            gen = tqdm(response.iter_lines())
            for line in gen:
                if line:
                    try:
                        text = json.loads(line.decode('utf-8').replace('data: ', ''))['content']
                        if not text:
                            time.sleep(0.001)
                            continue
                        yield text
                    except Exception as e:
                        logger.warning(e)
        except Exception as e:
            logger.error(e)

    def _generate(self, input_ids: str, config: GenerationConfig) -> list:
        new_tokens = []
        for token in self._generate_stream(input_ids, config):
            new_tokens.append(token)
        return self.tokenizer.encode("".join(new_tokens), add_special_tokens=False)
    
class LlamaCPPServerAsyncModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_seq_length: int = 4096,
            additional_model_config: dict = {}
    ) -> None:
        self.loop = asyncio.new_event_loop()
        self._session: aiohttp.ClientSession | None = None
        self.sem = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def _generate(
        self,
        prompt: str,
        request_id: int,
        config: GenerationConfig,
        pbar: atqdm,
        lock: asyncio.Lock,
    ):

        async with self.sem:

            logger.success(f"started: {request_id}")

            payload = {
                "top_p": config.sampler_config.top_p,
                "top_k": config.sampler_config.top_k,
                "min_p": config.sampler_config.top_p,
                "repeat_penalty": config.sampler_config.repetition_penalty,
                "temperature": config.sampler_config.temperature,
                "stop": ["<|im_end|>"],
                "n_predict": config.max_length,
                "stream": True,
                "prompt": prompt,
            }

            final_output = ""

            try:
                url = urljoin(config.server_host, "/completion")
                async with self._session.post(
                    url, json=payload
                ) as resp:
                    resp.raise_for_status()
                    async for raw in resp.content:
                        for line in raw.split(b"\n"):
                            if not line.strip():
                                continue
                            try:
                                text = json.loads(
                                    line.decode("utf-8").removeprefix("data: ")
                                )["content"]
                                if not text:
                                    # throttle tiny empty events
                                    await asyncio.sleep(0.001)
                                    continue

                                final_output += text

                                async with lock:
                                    pbar.update(1)
                                    speed = pbar.format_dict.get("rate", 0)
                                    rtf = round(1 / (speed / 150), 4) if speed else 0
                                    pbar.set_postfix({"RTF": rtf})
                            except Exception as e:
                                logger.warning(f"[{request_id}] parse error: {e}")
            except Exception as e:
                logger.error(f"[{request_id}] request failed: {e}")

            print("\n")
            logger.success(f"Batch {request_id} finished.\n")

            return {"text": final_output, "id": request_id}
    
    async def _generate_batch_async(self, input_ids: list[str], config: GenerationConfig):
        """Helper coroutine to run multiple _generate tasks in parallel and gather results in order."""
        pbar = atqdm(total=None)
        lock = asyncio.Lock()
        self.sem = asyncio.Semaphore(config.max_batch_size)

        await self._ensure_session()

        tasks = [
            asyncio.create_task(
                self._generate(i, idx, config, pbar, lock)
            )
            for idx, i in enumerate(input_ids)
        ]
        return await asyncio.gather(*tasks)
    
    def generate_batch(self, input_ids: list[str], config: GenerationConfig):
        try:
            output = self.loop.run_until_complete(
                self._generate_batch_async(input_ids, config)
            )
            return output
        finally:
            # clean up session
            if self._session and not self._session.closed:
                self.loop.run_until_complete(self._session.close())