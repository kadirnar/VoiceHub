"""Framework-free base datasets for speech training records."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def _copy_structure(value: Any) -> Any:
    """Copy mutable record containers without cloning tensor payloads."""
    if isinstance(value, Mapping):
        return {key: _copy_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_structure(item) for item in value)
    if isinstance(value, set):
        return {_copy_structure(item) for item in value}
    return value


def _has_value(record: Mapping[str, Any], name: str) -> bool:
    if name not in record or record[name] is None:
        return False
    value = record[name]
    return not isinstance(value, (str, bytes, bytearray)) or bool(value.strip())


class SpeechDataset(Sequence[dict[str, Any]]):
    """An immutable-indexed view over validated speech records.

    The dataset intentionally does not decode audio or import a tensor
    framework. Model-specific processors keep ownership of that work at
    batch preparation time, which lets the same records target CTC,
    sequence, transducer, classification, and synthesis models.
    """

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        required_fields: Iterable[str] = (),
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
    ):
        if isinstance(records, (str, bytes, Mapping)):
            raise TypeError("`records` must be an iterable of mappings.")
        required = tuple(required_fields)
        if any(not isinstance(name, str) or not name.strip() for name in required):
            raise ValueError("`required_fields` must contain non-empty strings.")
        required = tuple(dict.fromkeys(name.strip() for name in required))
        if transform is not None and not callable(transform):
            raise TypeError("`transform` must be callable or None.")

        normalized = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"Speech record {index} must be a mapping, received "
                    f"{type(record).__name__}.")
            copied = _copy_structure(record)
            missing = tuple(name for name in required if not _has_value(copied, name))
            if missing:
                raise ValueError(
                    f"Speech record {index} is missing required field(s): "
                    f"{', '.join(missing)}.")
            normalized.append(copied)
        self._records = tuple(normalized)
        self.required_fields = required
        self.transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._materialize(record) for record in self._records[index]]
        return self._materialize(self._records[index])

    def _materialize(self, record: Mapping[str, Any]) -> dict[str, Any]:
        value = _copy_structure(record)
        if self.transform is None:
            return value
        transformed = self.transform(value)
        if not isinstance(transformed, Mapping):
            raise TypeError("SpeechDataset transforms must return a mapping.")
        return _copy_structure(transformed)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return columns in first-seen order across all records."""
        return tuple(dict.fromkeys(key for record in self._records for key in record))


__all__ = ["SpeechDataset"]
