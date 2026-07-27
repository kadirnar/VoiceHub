"""Universal Transformers speech-recognition integration."""

from voicehub.models.asr_transformers.configuration_asr_transformers import TransformersASRConfig
from voicehub.models.asr_transformers.modeling_asr_transformers import TransformersASRForSpeechRecognition

__all__ = [
    "TransformersASRConfig",
    "TransformersASRForSpeechRecognition",
]
