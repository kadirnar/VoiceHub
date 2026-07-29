"""Backward-compatible inference imports for VoiceHub-native OmniVoice."""

from voicehub.models.omnivoice.configuration_omnivoice import OmniVoiceConfig
from voicehub.models.omnivoice.modeling_omnivoice import OmniVoiceForTextToSpeech, OmniVoiceTTS

__all__ = [
    "OmniVoiceConfig",
    "OmniVoiceForTextToSpeech",
    "OmniVoiceTTS",
]
