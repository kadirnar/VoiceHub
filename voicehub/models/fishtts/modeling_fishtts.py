"""Stable model imports for fishtts."""

from voicehub.architectures.fishtts.modeling import FishS2ForConditionalGeneration
from voicehub.models.fishtts.inference import FishTTSForTextToSpeech

__all__ = [
    "FishS2ForConditionalGeneration",
    "FishTTSForTextToSpeech",
]
