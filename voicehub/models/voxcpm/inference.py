"""Backward-compatible inference imports for VoiceHub-native VoxCPM2."""

from voicehub.models.voxcpm.configuration_voxcpm import VoxCPMConfig
from voicehub.models.voxcpm.modeling_voxcpm import VoxCPMForTextToSpeech, VoxCPMTTS

__all__ = ["VoxCPMConfig", "VoxCPMForTextToSpeech", "VoxCPMTTS"]
