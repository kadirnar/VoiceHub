"""Deterministic length-aware batching for variable-length TTS data.

The sampler is deliberately framework-free.  ``Trainer`` consumes it
through the same ``create_batch_sampler`` hook used by the ASR datasets,
while model profiles decide whether lengths represent spectrogram
frames, codec tokens, or another architecture-specific unit.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Real
from typing import Any


class TTSBatchingStrategy(str, Enum):
    """Supported variable-length batching policies."""

    LENGTH_BUCKET = "length-bucket"
    MAX_UNITS = "max-units"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: TTSBatchingStrategy | str) -> TTSBatchingStrategy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("TTS batching strategy must be a non-empty string or enum value.")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "bucket": cls.LENGTH_BUCKET.value,
            "buckets": cls.LENGTH_BUCKET.value,
            "budget": cls.MAX_UNITS.value,
            "frame-budget": cls.MAX_UNITS.value,
            "token-budget": cls.MAX_UNITS.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown TTS batching strategy {value!r}. Expected one of: "
                f"{choices}.") from exc


@dataclass(frozen=True, slots=True)
class TTSBatchingConfig:
    """Serializable length batching settings attached to a ``TTSDataset``.

    ``length_field`` must contain a positive numeric value in every
    manifest row. ``length_multiplier`` converts that metadata to the
    architecture's batching unit; for example, seconds become mel frames
    with ``sample_rate / hop_length``.

    ``max-units`` keeps the summed or padded batch cost under
    ``max_batch_units``. An individual oversized item is emitted as a
    singleton rather than silently discarded.
    """

    strategy: TTSBatchingStrategy | str
    length_field: str
    length_multiplier: float = 1.0
    bucket_boundaries: tuple[int, ...] = ()
    max_batch_units: int | None = None
    max_samples: int | None = None
    max_sequence_length: int | None = None
    budget_mode: str = "sum"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strategy",
            TTSBatchingStrategy.coerce(self.strategy),
        )
        if not isinstance(self.length_field, str) or not self.length_field.strip():
            raise ValueError("`length_field` must be a non-empty string.")
        object.__setattr__(self, "length_field", self.length_field.strip())
        if (isinstance(self.length_multiplier, bool) or not isinstance(self.length_multiplier, Real) or
                not math.isfinite(float(self.length_multiplier)) or float(self.length_multiplier) <= 0):
            raise ValueError("`length_multiplier` must be finite and positive.")
        object.__setattr__(
            self,
            "length_multiplier",
            float(self.length_multiplier),
        )

        boundaries = tuple(self.bucket_boundaries)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in boundaries):
            raise ValueError("`bucket_boundaries` must contain positive integers.")
        if tuple(sorted(set(boundaries))) != boundaries:
            raise ValueError("`bucket_boundaries` must be strictly increasing.")
        object.__setattr__(self, "bucket_boundaries", boundaries)

        for name in (
                "max_batch_units",
                "max_samples",
                "max_sequence_length",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"`{name}` must be a positive integer or None.")
        normalized_mode = str(self.budget_mode).strip().lower()
        if normalized_mode not in {"sum", "padded"}:
            raise ValueError("`budget_mode` must be 'sum' or 'padded'.")
        object.__setattr__(self, "budget_mode", normalized_mode)

        if self.strategy is TTSBatchingStrategy.LENGTH_BUCKET:
            if not boundaries:
                raise ValueError("Length-bucket batching requires `bucket_boundaries`.")
            if self.max_batch_units is not None:
                raise ValueError("Length-bucket batching does not accept `max_batch_units`.")
        elif self.max_batch_units is None:
            raise ValueError("Max-units batching requires `max_batch_units`.")

    @classmethod
    def from_mapping(
        cls,
        value: TTSBatchingConfig | Mapping[str, Any],
    ) -> TTSBatchingConfig:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("TTS batching configuration must be a mapping.")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["strategy"] = self.strategy.value
        values["bucket_boundaries"] = list(self.bucket_boundaries)
        return values


def _record_lengths(
    records: Sequence[Mapping[str, Any]],
    config: TTSBatchingConfig,
) -> tuple[int, ...]:
    lengths = []
    for index, record in enumerate(records):
        if config.length_field not in record:
            raise ValueError(
                f"TTS record {index} is missing batching length field "
                f"{config.length_field!r}.")
        value = record[config.length_field]
        if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or
                float(value) <= 0):
            raise ValueError(
                f"TTS record {index} field {config.length_field!r} must be "
                "finite and positive.")
        length = max(1, math.ceil(float(value) * config.length_multiplier))
        if (config.max_sequence_length is not None and length > config.max_sequence_length):
            raise ValueError(
                f"TTS record {index} has length {length}, exceeding "
                f"`max_sequence_length={config.max_sequence_length}`.")
        lengths.append(length)
    return tuple(lengths)


class EpochLengthBatchSampler:
    """Epoch-addressable bucket or budget batches with exact-resume state."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        config: TTSBatchingConfig | Mapping[str, Any],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("TTS batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("TTS batch_size must be positive.")
        self.config = TTSBatchingConfig.from_mapping(config)
        self.lengths = _record_lengths(records, self.config)
        if not self.lengths:
            raise ValueError("Length-aware TTS batching requires at least one record.")
        self.batch_size = batch_size
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        payload = json.dumps(
            list(self.lengths),
            separators=(",", ":"),
        ).encode("utf-8")
        self.length_sha256 = hashlib.sha256(payload).hexdigest()

    @property
    def max_samples(self) -> int:
        configured = self.config.max_samples
        return self.batch_size if configured is None else min(self.batch_size, configured)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _bucket_batches(self, randomizer: random.Random) -> list[list[int]]:
        groups: dict[int, list[int]] = {}
        for index, length in enumerate(self.lengths):
            bucket = bisect.bisect_left(
                self.config.bucket_boundaries,
                length,
            )
            groups.setdefault(bucket, []).append(index)
        batches = []
        for bucket in sorted(groups):
            indices = groups[bucket]
            if self.shuffle:
                randomizer.shuffle(indices)
            for start in range(0, len(indices), self.max_samples):
                batch = indices[start:start + self.max_samples]
                if len(batch) == self.max_samples or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            randomizer.shuffle(batches)
        return batches

    def _batch_cost(self, indices: Sequence[int]) -> int:
        values = [self.lengths[index] for index in indices]
        if self.config.budget_mode == "padded":
            return max(values) * len(values)
        return sum(values)

    def _budget_batches(self, randomizer: random.Random) -> list[list[int]]:
        indices = sorted(
            range(len(self.lengths)),
            key=lambda index: (self.lengths[index], index),
        )
        if self.shuffle:
            # Keep neighboring lengths together while changing membership
            # deterministically across epochs.
            window = max(1, self.max_samples * 8)
            for start in range(0, len(indices), window):
                chunk = indices[start:start + window]
                randomizer.shuffle(chunk)
                indices[start:start + window] = chunk

        batches: list[list[int]] = []
        current: list[int] = []
        for index in indices:
            candidate = [*current, index]
            exceeds_count = len(candidate) > self.max_samples
            exceeds_budget = self._batch_cost(candidate) > int(self.config.max_batch_units or 0)
            if current and (exceeds_count or exceeds_budget):
                batches.append(current)
                current = [index]
            else:
                current = candidate
        if current:
            batches.append(current)
        if self.drop_last and batches and len(batches[-1]) < self.max_samples:
            batches.pop()
        if self.shuffle:
            randomizer.shuffle(batches)
        return batches

    def _batches(self) -> list[list[int]]:
        randomizer = random.Random(self.seed + self.epoch)
        if self.config.strategy is TTSBatchingStrategy.LENGTH_BUCKET:
            return self._bucket_batches(randomizer)
        return self._budget_batches(randomizer)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "batching": self.config.to_dict(),
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "length_sha256": self.length_sha256,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = {
            "batch_size": self.batch_size,
            "batching": self.config.to_dict(),
            "drop_last": self.drop_last,
            "length_sha256": self.length_sha256,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        for name, value in expected.items():
            if state_dict.get(name) != value:
                raise ValueError(
                    f"TTS batch sampler {name} differs from the checkpoint "
                    f"({state_dict.get(name)!r} != {value!r}).")
        self.epoch = int(state_dict["epoch"])


__all__ = [
    "EpochLengthBatchSampler",
    "TTSBatchingConfig",
    "TTSBatchingStrategy",
]
