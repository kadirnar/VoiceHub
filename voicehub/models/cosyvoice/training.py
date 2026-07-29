"""Backward-compatible training imports for native CosyVoice 3."""

from voicehub.models.cosyvoice_native.training_cosyvoice import CosyVoiceTrainingAdapter, CosyVoiceTrainingCollator

__all__ = [
    "CosyVoiceTrainingAdapter",
    "CosyVoiceTrainingCollator",
]
