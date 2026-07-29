"""Compatibility namespace for VoiceHub-native multimodal ASR providers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "MultimodalTransformersASRConfig": (
        "voicehub.models.asr_transformers_multimodal.configuration_asr_transformers_multimodal",
        "MultimodalTransformersASRConfig",
    ),
    "MultimodalTransformersASRForSpeechRecognition": (
        "voicehub.models.asr_transformers_multimodal.modeling_asr_transformers_multimodal",
        "MultimodalTransformersASRForSpeechRecognition",
    ),
    "Qwen3ASRConfig": (
        "voicehub.models.asr_qwen3.configuration_asr_qwen3",
        "Qwen3ASRConfig",
    ),
    "Qwen3ASRForSpeechRecognition": (
        "voicehub.models.asr_qwen3.modeling_asr_qwen3",
        "Qwen3ASRForSpeechRecognition",
    ),
    "VibeVoiceASRConfig": (
        "voicehub.models.asr_vibevoice.configuration_asr_vibevoice",
        "VibeVoiceASRConfig",
    ),
    "VibeVoiceASRForSpeechRecognition": (
        "voicehub.models.asr_vibevoice.modeling_asr_vibevoice",
        "VibeVoiceForSpeechRecognition",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
