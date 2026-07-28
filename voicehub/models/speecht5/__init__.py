"""Hugging Face SpeechT5 support."""

from voicehub.models.speecht5.configuration_speecht5 import SpeechT5Config
from voicehub.models.speecht5.modeling_speecht5 import SpeechT5ForTextToSpeech

__all__ = [
    "SpeechT5Config",
    "SpeechT5ForTextToSpeech",
]
