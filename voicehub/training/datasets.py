"""Compatibility facade for speech and TTS training datasets.

The implementation lives in focused modules:

* :mod:`voicehub.training.dataset_base` contains the framework-free base
  dataset.
* :mod:`voicehub.training.data_contracts` contains TTS source-data contracts.
* :mod:`voicehub.training.tts_datasets` contains manifest readers and the
  portable TTS dataset.

Existing imports from :mod:`voicehub.training.datasets` remain supported.
"""

from voicehub.training.data_contracts import (
    TTSDataArchitecture,
    TTSDataReadiness,
    TTSDatasetSpec,
    TTSRecordVariant,
    get_tts_dataset_spec,
    list_tts_dataset_specs,
)
from voicehub.training.dataset_base import SpeechDataset
from voicehub.training.tts_datasets import TTSDataset

__all__ = [
    "SpeechDataset",
    "TTSDataArchitecture",
    "TTSDataReadiness",
    "TTSDataset",
    "TTSDatasetSpec",
    "TTSRecordVariant",
    "get_tts_dataset_spec",
    "list_tts_dataset_specs",
]
