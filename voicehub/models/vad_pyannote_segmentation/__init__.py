"""Pyannote segmentation-3.0 voice activity detection."""

from .configuration_vad_pyannote_segmentation import PyannoteSegmentationVADConfig
from .modeling_vad_pyannote_segmentation import PyannoteSegmentationVADForVoiceActivityDetection

__all__ = [
    "PyannoteSegmentationVADConfig",
    "PyannoteSegmentationVADForVoiceActivityDetection",
]
