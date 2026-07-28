"""Hugging Face VITS and MMS-TTS support."""

from voicehub.models.vits.configuration_vits import VitsConfig
from voicehub.models.vits.modeling_vits import VitsForTextToSpeech

__all__ = [
    "VitsConfig",
    "VitsForTextToSpeech",
]
