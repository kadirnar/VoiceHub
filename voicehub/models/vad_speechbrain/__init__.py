"""SpeechBrain VAD integration."""

from voicehub.models.vad_speechbrain.configuration_vad_speechbrain import SpeechBrainVADConfig
from voicehub.models.vad_speechbrain.modeling_vad_speechbrain import SpeechBrainVADForVoiceActivityDetection

__all__ = [
    "SpeechBrainVADConfig",
    "SpeechBrainVADForVoiceActivityDetection",
]
