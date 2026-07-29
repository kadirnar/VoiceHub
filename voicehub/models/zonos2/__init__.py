"""ZONOS2 model family."""

from voicehub.models.zonos2.inference import Zonos2Config, Zonos2ForTextToSpeech, Zonos2TTS
from voicehub.models.zonos2.training import Zonos2TrainingAdapter

__all__ = [
    "Zonos2Config",
    "Zonos2ForTextToSpeech",
    "Zonos2TTS",
    "Zonos2TrainingAdapter",
]
