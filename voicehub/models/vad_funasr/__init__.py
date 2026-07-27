"""FunASR FSMN VAD integration."""

from voicehub.models.vad_funasr.configuration_vad_funasr import FunASRVADConfig
from voicehub.models.vad_funasr.modeling_vad_funasr import FunASRVADForVoiceActivityDetection

__all__ = [
    "FunASRVADConfig",
    "FunASRVADForVoiceActivityDetection",
]
