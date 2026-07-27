"""NVIDIA NeMo VAD integration."""

from voicehub.models.vad_nemo.configuration_vad_nemo import NeMoVADConfig
from voicehub.models.vad_nemo.modeling_vad_nemo import NeMoVADForVoiceActivityDetection

__all__ = [
    "NeMoVADConfig",
    "NeMoVADForVoiceActivityDetection",
]
