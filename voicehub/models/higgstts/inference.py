"""Backward-compatible imports for VoiceHub-native Higgs Audio v2."""

from voicehub.models.higgstts.configuration_higgstts import HiggsTTSConfig
from voicehub.models.higgstts.modeling_higgstts import HiggsTTS, HiggsTTSForTextToSpeech

__all__ = [
    "HiggsTTS",
    "HiggsTTSConfig",
    "HiggsTTSForTextToSpeech",
]
