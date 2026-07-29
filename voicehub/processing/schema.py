"""Typed data exchanged between native speech processors and architectures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError(f"`{name}` keys must be non-empty strings.")
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class ModelBatch(Mapping[str, Any]):
    """Immutable tensor payload plus processor-owned metadata."""

    data: Mapping[str, Any]
    batch_size: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data",
            _freeze_mapping(self.data, name="data"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )
        if (
            isinstance(self.batch_size, bool)
            or not isinstance(self.batch_size, int)
            or self.batch_size <= 0
        ):
            raise ValueError("`batch_size` must be a positive integer.")

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def to(self, *args: Any, **kwargs: Any) -> "ModelBatch":
        """Return a batch with tensor-like values moved through ``.to()``."""

        def move(value: Any) -> Any:
            method = getattr(value, "to", None)
            if callable(method):
                return method(*args, **kwargs)
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, Mapping):
                return {key: move(item) for key, item in value.items()}
            return value

        return ModelBatch(
            data={key: move(value) for key, value in self.data.items()},
            batch_size=self.batch_size,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class TrainingExample(Mapping[str, Any]):
    """One processed, uncollated training record."""

    data: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data",
            _freeze_mapping(self.data, name="data"),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, name="metadata"),
        )

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
