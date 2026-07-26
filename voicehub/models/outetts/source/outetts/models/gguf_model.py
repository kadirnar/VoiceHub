from loguru import logger
from tqdm import tqdm
from packaging import version

from .info import GenerationType
from .config import GenerationConfig

try:
    from llama_cpp import Llama, llama_token_is_eog
    from llama_cpp import __version__ as llama_cpp_version
    _GGUF_AVAILABLE = True
except:
    llama_cpp_version = "0.0.0"
    _GGUF_AVAILABLE = False
    raise ImportError(
        "llama.cpp Python bindings not found. This is required for GGUF model support.\n\n"
        "To install, please follow our installation guide:\n"
        "https://github.com/edwko/OuteTTS?tab=readme-ov-file#installation\n\n"
    )

CURRENT_VERSION = version.parse(llama_cpp_version)
VERSION_0_3_7 = version.parse("0.3.7")

class GGUFModel:
    def __init__(
            self,
            model_path: str,
            n_gpu_layers: int = 0,
            max_seq_length: int = 4096,
            additional_model_config: dict = {}
    ) -> None:

        if not _GGUF_AVAILABLE:
            raise ImportError(
                "llama_cpp python module not found."
            )

        additional_model_config["n_ctx"] = max_seq_length
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            last_n_tokens_size=64,
            **additional_model_config
        )

    def is_eog(self):
        if CURRENT_VERSION >= VERSION_0_3_7:
            return self.model._model.vocab
        else:
            return self.model._model.model

    def generate(self, input_ids: list[int], config: GenerationConfig):
        if config.generation_type == GenerationType.STREAM:
            return self._generate_stream(input_ids, config)
        return self._generate(input_ids, config)

    def _generate_stream(self, input_ids: list[int], config: GenerationConfig):
        input_size = len(input_ids)
        gen = tqdm(self.model.generate(
            input_ids,
            temp=config.sampler_config.temperature,
            repeat_penalty=config.sampler_config.repetition_penalty,
            top_k=config.sampler_config.top_k,
            top_p=config.sampler_config.top_p,
            min_p=config.sampler_config.min_p,
            mirostat_eta=config.sampler_config.mirostat_eta,
            mirostat_tau=config.sampler_config.mirostat_tau,
            **config.additional_gen_config,
        ))
        for token in gen:
            yield token
            input_size += 1
            if (llama_token_is_eog(self.is_eog(), token) or 
                input_size >= config.max_length):
                break
            gen.set_postfix({"tokens": input_size,  "max tokens": config.max_length})

    def _generate(self, input_ids: list[int], config: GenerationConfig) -> list:
        new_tokens = []
        for token in self._generate_stream(input_ids, config):
            new_tokens.append(token)
        return new_tokens