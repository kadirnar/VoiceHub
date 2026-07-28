"""Chat-template Transformers ASR integrations."""

from voicehub.models.asr_transformers_multimodal.configuration_asr_transformers_multimodal import (
    MultimodalTransformersASRConfig,
    Qwen3ASRConfig,
    VibeVoiceASRConfig,
)
from voicehub.models.asr_transformers_multimodal.modeling_asr_transformers_multimodal import (
    MultimodalTransformersASRForSpeechRecognition,
    Qwen3ASRForSpeechRecognition,
    VibeVoiceASRForSpeechRecognition,
)

__all__ = [
    "MultimodalTransformersASRConfig",
    "MultimodalTransformersASRForSpeechRecognition",
    "Qwen3ASRConfig",
    "Qwen3ASRForSpeechRecognition",
    "VibeVoiceASRConfig",
    "VibeVoiceASRForSpeechRecognition",
]
