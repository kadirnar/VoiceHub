"""Stable processor contract shared by inference and fine-tuning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.processing.schema import ModelBatch, TrainingExample


class SpeechProcessor(ABC):
    """Architecture-owned preprocessing, collation, and decoding boundary."""

    processor_id: str
    processor_version: str

    @abstractmethod
    def encode_inference(self, request: Any) -> ModelBatch:
        """Convert one public request into a model-ready batch."""

    @abstractmethod
    def encode_training(
        self,
        record: Mapping[str, Any],
    ) -> TrainingExample:
        """Convert one dataset record using inference-identical primitives."""

    @abstractmethod
    def collate(
        self,
        examples: Sequence[TrainingExample],
    ) -> ModelBatch:
        """Pad and combine processed examples."""

    @abstractmethod
    def decode(self, model_output: Any, context: Any = None) -> Any:
        """Convert architecture output into the public task output."""
