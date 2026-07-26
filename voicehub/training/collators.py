"""Padding collator for token, acoustic, and waveform training examples."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

from voicehub.dependencies import import_optional


@dataclass
class DataCollatorForTTSTraining:
    """Pad variable-length tensors while retaining model-specific fields."""

    padding_value: float = 0.0
    label_pad_token_id: int = -100
    return_attention_mask: bool = True

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}
        if not all(isinstance(feature, dict) for feature in features):
            raise TypeError("TTS training samples must be dictionaries.")

        batch = {}
        keys = tuple(dict.fromkeys(key for feature in features for key in feature))
        input_lengths = None
        for key in keys:
            values = [feature.get(key) for feature in features]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                batch[key] = values
                continue
            batch[key], lengths = self._collate_values(key, values)
            if key == "input_ids":
                input_lengths = lengths

        if (self.return_attention_mask and "input_ids" in batch and "attention_mask" not in batch and
                input_lengths is not None):
            torch = self._import_torch()
            sequence_length = batch["input_ids"].shape[1]
            positions = torch.arange(sequence_length).unsqueeze(0)
            batch["attention_mask"] = positions < torch.tensor(input_lengths).unsqueeze(1)
        return batch

    @staticmethod
    def _import_torch():
        return import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )

    def _collate_values(self, key: str, values: list[Any]):
        torch = self._import_torch()
        first = values[0]
        if isinstance(first, dict):
            nested = self([dict(value) for value in values])
            return nested, None
        if isinstance(first, str):
            return values, None
        if isinstance(first, (int, float, bool)):
            return torch.tensor(values), None

        tensors = self._as_tensors(values)
        if tensors is None:
            return values, None
        if all(tensor.ndim == 0 for tensor in tensors):
            return torch.stack(tensors), None

        lengths = [int(tensor.shape[0]) for tensor in tensors]
        same_shape = all(tuple(tensor.shape) == tuple(tensors[0].shape) for tensor in tensors)
        if same_shape:
            return torch.stack(tensors), lengths

        tail_shape = tuple(tensors[0].shape[1:])
        padding_value = (
            self.label_pad_token_id
            if key in ("labels", "label_ids") and not tensors[0].is_floating_point() else self.padding_value)
        if all(tuple(tensor.shape[1:]) == tail_shape for tensor in tensors):
            padded = torch.nn.utils.rnn.pad_sequence(
                tensors,
                batch_first=True,
                padding_value=padding_value,
            )
            return padded, lengths

        leading_shape = tuple(tensors[0].shape[:-1])
        if all(tuple(tensor.shape[:-1]) == leading_shape for tensor in tensors):
            lengths = [int(tensor.shape[-1]) for tensor in tensors]
            max_length = max(lengths)
            padded = [
                torch.nn.functional.pad(
                    tensor,
                    (0, max_length - tensor.shape[-1]),
                    value=padding_value,
                ) for tensor in tensors
            ]
            return torch.stack(padded), lengths
        return values, None

    def _as_tensors(self, values: list[Any]):
        torch = self._import_torch()
        if all(isinstance(value, torch.Tensor) for value in values):
            return values
        if all(hasattr(value, "dtype") and hasattr(value, "shape") for value in values):
            try:
                numpy = import_module("numpy")
                return [torch.as_tensor(numpy.asarray(value)) for value in values]
            except (ModuleNotFoundError, TypeError, ValueError):
                return None
        if all(isinstance(value, (list, tuple)) for value in values):
            try:
                return [torch.tensor(value) for value in values]
            except (TypeError, ValueError):
                return None
        return None
