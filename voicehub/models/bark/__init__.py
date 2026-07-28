"""Hugging Face Bark support."""

from voicehub.models.bark.configuration_bark import BarkConfig
from voicehub.models.bark.modeling_bark import BarkForTextToSpeech

__all__ = [
    "BarkConfig",
    "BarkForTextToSpeech",
]
