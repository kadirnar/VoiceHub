"""External vLLM/SGLang serving support for LLM-based TTS models."""

from voicehub.llm_serving.configuration import LLMBackend, LLMBackendConfig, LLMBackendTransport
from voicehub.llm_serving.protocol import TokenGenerationRequest, TokenGenerationResult
from voicehub.llm_serving.support import (
    LLMBackendSupport,
    get_llm_backend_support,
    list_llm_backend_support,
    register_llm_backend_support,
    unregister_llm_backend_support,
)

__all__ = [
    "LLMBackend",
    "LLMBackendConfig",
    "LLMBackendSupport",
    "LLMBackendTransport",
    "LLMServingClient",
    "RemoteCausalLMProxy",
    "TokenGenerationRequest",
    "TokenGenerationResult",
    "get_llm_backend_support",
    "list_llm_backend_support",
    "register_llm_backend_support",
    "unregister_llm_backend_support",
]


def __getattr__(name: str):
    """Load Torch/audio-aware clients only when serving is configured."""
    if name in {"LLMServingClient", "RemoteCausalLMProxy"}:
        from voicehub.llm_serving.backends import LLMServingClient, RemoteCausalLMProxy

        return {
            "LLMServingClient": LLMServingClient,
            "RemoteCausalLMProxy": RemoteCausalLMProxy,
        }[name]
    raise AttributeError(name)
