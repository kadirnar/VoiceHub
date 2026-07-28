"""Pyannote Brouhaha multi-task voice activity detection."""

from voicehub.models.vad_pyannote_brouhaha.configuration_vad_pyannote_brouhaha import PyannoteBrouhahaVADConfig
from voicehub.models.vad_pyannote_brouhaha.modeling_vad_pyannote_brouhaha import (
    PyannoteBrouhahaVADForVoiceActivityDetection, )

__all__ = [
    "PyannoteBrouhahaVADConfig",
    "PyannoteBrouhahaVADForVoiceActivityDetection",
]
