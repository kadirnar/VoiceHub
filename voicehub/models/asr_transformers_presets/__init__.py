"""Compatibility exports for ASR presets now implemented by VoiceHub.

This historical namespace is intentionally lazy.  Importing it does not
load PyTorch or any model graph, and every public symbol resolves to the
dedicated VoiceHub-native provider that owns the corresponding
architecture.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CohereASRConfig": (
        "voicehub.models.asr_cohere.configuration_asr_cohere",
        "CohereASRConfig",
    ),
    "CohereForSpeechRecognition": (
        "voicehub.models.asr_cohere.modeling_asr_cohere",
        "CohereForSpeechRecognition",
    ),
    "HubertASRConfig": (
        "voicehub.models.asr_hubert.configuration_asr_hubert",
        "HubertASRConfig",
    ),
    "HubertForSpeechRecognition": (
        "voicehub.models.asr_hubert.modeling_asr_hubert",
        "HubertForSpeechRecognition",
    ),
    "MedASRConfig": (
        "voicehub.models.asr_medasr.configuration_asr_medasr",
        "MedASRConfig",
    ),
    "MedASRForSpeechRecognition": (
        "voicehub.models.asr_medasr.modeling_asr_medasr",
        "MedASRForSpeechRecognition",
    ),
    "MoonshineASRConfig": (
        "voicehub.models.asr_moonshine.configuration_asr_moonshine",
        "MoonshineASRConfig",
    ),
    "MoonshineForSpeechRecognition": (
        "voicehub.models.asr_moonshine.modeling_asr_moonshine",
        "MoonshineForSpeechRecognition",
    ),
    "NemotronASRConfig": (
        "voicehub.models.asr_nemotron.configuration_asr_nemotron",
        "NemotronASRConfig",
    ),
    "NemotronForSpeechRecognition": (
        "voicehub.models.asr_nemotron.modeling_asr_nemotron",
        "NemotronForSpeechRecognition",
    ),
    "ParakeetTDTASRConfig": (
        "voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt",
        "ParakeetTDTASRConfig",
    ),
    "ParakeetTDTForSpeechRecognition": (
        "voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt",
        "ParakeetTDTForSpeechRecognition",
    ),
    "SeamlessM4Tv2ASRConfig": (
        "voicehub.models.asr_seamless_m4t_v2.configuration_asr_seamless_m4t_v2",
        "SeamlessM4Tv2ASRConfig",
    ),
    "SeamlessM4Tv2ForSpeechRecognition": (
        "voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2",
        "SeamlessM4Tv2ForSpeechRecognition",
    ),
    "Wav2Vec2ASRConfig": (
        "voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2",
        "Wav2Vec2ASRConfig",
    ),
    "Wav2Vec2ForSpeechRecognition": (
        "voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2",
        "Wav2Vec2ForSpeechRecognition",
    ),
    "WavLMASRConfig": (
        "voicehub.models.asr_wavlm.configuration_asr_wavlm",
        "WavLMASRConfig",
    ),
    "WavLMForSpeechRecognition": (
        "voicehub.models.asr_wavlm.modeling_asr_wavlm",
        "WavLMForSpeechRecognition",
    ),
    "WhisperASRConfig": (
        "voicehub.models.asr_whisper_native.configuration_asr_whisper_native",
        "WhisperASRConfig",
    ),
    "WhisperForSpeechRecognition": (
        "voicehub.models.asr_whisper_native.modeling_asr_whisper_native",
        "WhisperForSpeechRecognition",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a legacy symbol to its canonical native implementation."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
