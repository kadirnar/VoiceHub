"""Pyannote Brouhaha multi-task voice activity detection."""

from .configuration_vad_pyannote_brouhaha import PyannoteBrouhahaVADConfig
from .modeling_vad_pyannote_brouhaha import PyannoteBrouhahaVADForVoiceActivityDetection

__all__ = [
    "PyannoteBrouhahaVADConfig",
    "PyannoteBrouhahaVADForVoiceActivityDetection",
]
