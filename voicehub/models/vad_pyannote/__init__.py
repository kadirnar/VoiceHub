"""pyannote.audio VAD integration."""

from voicehub.models.vad_pyannote.configuration_vad_pyannote import PyannoteVADConfig
from voicehub.models.vad_pyannote.modeling_vad_pyannote import PyannoteVADForVoiceActivityDetection

__all__ = [
    "PyannoteVADConfig",
    "PyannoteVADForVoiceActivityDetection",
]
