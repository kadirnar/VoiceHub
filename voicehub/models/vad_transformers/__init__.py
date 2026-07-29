"""Native Wav2Vec2 audio/frame classification VAD integration."""

from voicehub.models.vad_transformers.configuration_vad_transformers import TransformersVADConfig
from voicehub.models.vad_transformers.modeling_vad_transformers import TransformersVADForVoiceActivityDetection

__all__ = [
    "TransformersVADConfig",
    "TransformersVADForVoiceActivityDetection",
]
