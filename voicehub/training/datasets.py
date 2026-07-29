"""Compatibility facade for speech, ASR, and TTS training datasets.

The implementation lives in focused modules:

* :mod:`voicehub.training.dataset_base` contains the framework-free base
  dataset.
* :mod:`voicehub.training.asr_data_contracts` contains ASR source-data
  contracts.
* :mod:`voicehub.training.asr_datasets` contains portable ASR manifests.
* :mod:`voicehub.training.data_contracts` contains TTS source-data contracts.
* :mod:`voicehub.training.tts_datasets` contains manifest readers and the
  portable TTS dataset.

Existing imports from :mod:`voicehub.training.datasets` remain supported.
"""

from voicehub.training.asr_data_contracts import (
    ASRDataArchitecture,
    ASRDataReadiness,
    ASRDatasetSpec,
    ASRRecordVariant,
    get_asr_dataset_spec,
    list_asr_dataset_specs,
)
from voicehub.training.asr_datasets import ASRDataset, EpochGroupedBatchSampler
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
    "ASRDataArchitecture",
    "ASRDataReadiness",
    "ASRDataset",
    "ASRDatasetSpec",
    "ASRRecordVariant",
    "EpochGroupedBatchSampler",
    "SpeechDataset",
    "TTSDataArchitecture",
    "TTSDataReadiness",
    "TTSDataset",
    "TTSDatasetSpec",
    "TTSRecordVariant",
    "get_asr_dataset_spec",
    "get_tts_dataset_spec",
    "list_asr_dataset_specs",
    "list_tts_dataset_specs",
]
