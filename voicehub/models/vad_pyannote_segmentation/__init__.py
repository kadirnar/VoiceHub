"""Pyannote segmentation-3.0 voice activity detection."""

from voicehub.models.vad_pyannote_segmentation.configuration_vad_pyannote_segmentation import (
    PyannoteSegmentationVADConfig, )
from voicehub.models.vad_pyannote_segmentation.modeling_vad_pyannote_segmentation import (
    PyannoteSegmentationVADForVoiceActivityDetection, )

__all__ = [
    "PyannoteSegmentationVADConfig",
    "PyannoteSegmentationVADForVoiceActivityDetection",
]
