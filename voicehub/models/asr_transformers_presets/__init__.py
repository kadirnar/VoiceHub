"""First-party architecture presets for the universal Transformers ASR
provider."""

from voicehub.models.asr_transformers_presets.configuration_asr_transformers_presets import (
    HubertASRConfig,
    MoonshineASRConfig,
    SeamlessM4Tv2ASRConfig,
    Wav2Vec2ASRConfig,
    WavLMASRConfig,
)
from voicehub.models.asr_transformers_presets.modeling_asr_transformers_presets import (
    HubertForSpeechRecognition,
    MoonshineForSpeechRecognition,
    SeamlessM4Tv2ForSpeechRecognition,
    Wav2Vec2ForSpeechRecognition,
    WavLMForSpeechRecognition,
)

__all__ = [
    "HubertASRConfig",
    "HubertForSpeechRecognition",
    "MoonshineASRConfig",
    "MoonshineForSpeechRecognition",
    "SeamlessM4Tv2ASRConfig",
    "SeamlessM4Tv2ForSpeechRecognition",
    "Wav2Vec2ASRConfig",
    "Wav2Vec2ForSpeechRecognition",
    "WavLMASRConfig",
    "WavLMForSpeechRecognition",
]
