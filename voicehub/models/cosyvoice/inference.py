"""Backward-compatible inference imports for native CosyVoice 3."""

from voicehub.models.cosyvoice_native.configuration_cosyvoice import CosyVoiceConfig
from voicehub.models.cosyvoice_native.modeling_cosyvoice import CosyVoiceForTextToSpeech, CosyVoiceTTS

__all__ = [
    "CosyVoiceConfig",
    "CosyVoiceForTextToSpeech",
    "CosyVoiceTTS",
]
