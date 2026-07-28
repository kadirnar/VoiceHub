"""Tiron speaker-attributed automatic speech recognition."""

from voicehub.models.asr_tiron.configuration_asr_tiron import TironASRConfig
from voicehub.models.asr_tiron.modeling_asr_tiron import TironForSpeechRecognition

__all__ = [
    "TironASRConfig",
    "TironForSpeechRecognition",
]
