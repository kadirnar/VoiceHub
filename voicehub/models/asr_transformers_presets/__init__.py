"""First-party architecture presets for the universal Transformers ASR
provider."""

from voicehub.models.asr_transformers_presets.configuration_asr_transformers_presets import (
    CohereASRConfig,
    HubertASRConfig,
    MedASRConfig,
    MoonshineASRConfig,
    NemotronASRConfig,
    ParakeetTDTASRConfig,
    SeamlessM4Tv2ASRConfig,
    Wav2Vec2ASRConfig,
    WavLMASRConfig,
    WhisperASRConfig,
)
from voicehub.models.asr_transformers_presets.modeling_asr_transformers_presets import (
    CohereForSpeechRecognition,
    HubertForSpeechRecognition,
    MedASRForSpeechRecognition,
    MoonshineForSpeechRecognition,
    NemotronForSpeechRecognition,
    ParakeetTDTForSpeechRecognition,
    SeamlessM4Tv2ForSpeechRecognition,
    Wav2Vec2ForSpeechRecognition,
    WavLMForSpeechRecognition,
    WhisperForSpeechRecognition,
)

__all__ = [
    "CohereASRConfig",
    "CohereForSpeechRecognition",
    "HubertASRConfig",
    "HubertForSpeechRecognition",
    "MedASRConfig",
    "MedASRForSpeechRecognition",
    "MoonshineASRConfig",
    "MoonshineForSpeechRecognition",
    "NemotronASRConfig",
    "NemotronForSpeechRecognition",
    "ParakeetTDTASRConfig",
    "ParakeetTDTForSpeechRecognition",
    "SeamlessM4Tv2ASRConfig",
    "SeamlessM4Tv2ForSpeechRecognition",
    "Wav2Vec2ASRConfig",
    "Wav2Vec2ForSpeechRecognition",
    "WavLMASRConfig",
    "WavLMForSpeechRecognition",
    "WhisperASRConfig",
    "WhisperForSpeechRecognition",
]
