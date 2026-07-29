"""Explicit, composable tensor mappings for upstream checkpoints.

Checkpoint conversion is data, not ad-hoc loader code.  Every rule
declares the source tensors it consumes and the canonical VoiceHub
tensors it emits. This makes mappings reviewable, versionable, and
testable without importing an upstream model implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import prod
from typing import Any, Protocol, runtime_checkable

from voicehub.checkpointing.errors import CheckpointCompatibilityError


@runtime_checkable
class TensorSource(Protocol):
    """Minimal lazy tensor source consumed by a :class:`TensorPlan`."""

    def keys(self) -> Sequence[str]:
        """Return every tensor name available in the checkpoint."""

    def get_tensor(self, name: str):
        """Materialize one tensor by name."""


class MappingTensorSource:
    """Expose an in-memory state-dict through the lazy source protocol."""

    def __init__(self, tensors: Mapping[str, Any]) -> None:
        if not isinstance(tensors, Mapping):
            raise TypeError("`tensors` must be a mapping.")
        if any(not isinstance(name, str) or not name for name in tensors):
            raise ValueError("Tensor source names must be non-empty strings.")
        self._tensors = tensors

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._tensors))

    def get_tensor(self, name: str):
        try:
            return self._tensors[name]
        except KeyError as error:
            raise KeyError(f"Checkpoint tensor {name!r} was not found.") from error

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        """Return one tensor shape without copying its payload."""
        tensor = self.get_tensor(name)
        try:
            return tuple(tensor.shape)
        except (AttributeError, TypeError) as error:
            raise TypeError(f"Checkpoint value {name!r} does not expose a tensor shape.") from error


class TensorResolver:
    """Read-through cache shared by all rules in one materialization."""

    def __init__(self, source: TensorSource) -> None:
        self.source = source
        self._available = frozenset(source.keys())
        self._cache: dict[str, Any] = {}
        self.consumed: set[str] = set()

    @property
    def available(self) -> frozenset[str]:
        return self._available

    def get(self, name: str):
        if name not in self._available:
            raise CheckpointCompatibilityError(
                f"Checkpoint tensor {name!r} required by the conversion plan "
                "is missing.")
        self.consumed.add(name)
        if name not in self._cache:
            self._cache[name] = self.source.get_tensor(name)
        return self._cache[name]


def _name(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"`{field_name}` must be a non-empty string.")
    return value


def _names(
    values: Sequence[str],
    *,
    field_name: str,
    minimum: int = 1,
) -> tuple[str, ...]:
    result = tuple(values)
    if len(result) < minimum:
        raise ValueError(f"`{field_name}` must contain at least {minimum} name(s).")
    for value in result:
        _name(value, field_name=field_name)
    if len(result) != len(set(result)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return result


class TensorRule(ABC):
    """One deterministic mapping from checkpoint to canonical tensor names."""

    @property
    @abstractmethod
    def source_names(self) -> tuple[str, ...]:
        """Checkpoint tensors consumed by this rule."""

    @property
    @abstractmethod
    def target_names(self) -> tuple[str, ...]:
        """Canonical tensors produced by this rule."""

    @abstractmethod
    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        """Evaluate the rule using a shared lazy tensor resolver."""


@dataclass(frozen=True)
class CopyTensor(TensorRule):
    """Copy or rename one source tensor."""

    source: str
    target: str

    def __post_init__(self) -> None:
        _name(self.source, field_name="source")
        _name(self.target, field_name="target")

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self.target, )

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        return {self.target: resolver.get(self.source)}


@dataclass(frozen=True)
class TransposeTensor(TensorRule):
    """Permute tensor dimensions while renaming it."""

    source: str
    target: str
    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.source, field_name="source")
        _name(self.target, field_name="target")
        if (not isinstance(self.dimensions, tuple) or
                any(isinstance(value, bool) or not isinstance(value, int) for value in self.dimensions)):
            raise TypeError("`dimensions` must be a tuple of integers.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self.target, )

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        tensor = resolver.get(self.source)
        rank = tensor.ndim
        normalized = tuple(value % rank for value in self.dimensions) if rank else ()
        if len(normalized) != rank or set(normalized) != set(range(rank)):
            raise CheckpointCompatibilityError(
                f"Transpose for {self.source!r} requires a permutation of "
                f"{rank} dimensions, found {self.dimensions!r}.")
        return {self.target: tensor.permute(*normalized).contiguous()}


@dataclass(frozen=True)
class SplitTensor(TensorRule):
    """Split one packed upstream tensor into canonical tensors."""

    source: str
    targets: tuple[str, ...]
    sizes: tuple[int, ...]
    dimension: int = 0

    def __post_init__(self) -> None:
        _name(self.source, field_name="source")
        object.__setattr__(
            self,
            "targets",
            _names(self.targets, field_name="targets", minimum=2),
        )
        if (not isinstance(self.sizes, tuple) or len(self.sizes) != len(self.targets) or
                any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in self.sizes)):
            raise ValueError("`sizes` must contain one non-negative integer per target.")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("`dimension` must be an integer.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.targets

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        tensor = resolver.get(self.source)
        if tensor.ndim == 0:
            raise CheckpointCompatibilityError(f"Cannot split scalar checkpoint tensor {self.source!r}.")
        dimension = self.dimension % tensor.ndim
        if sum(self.sizes) != tensor.shape[dimension]:
            raise CheckpointCompatibilityError(
                f"Split sizes {self.sizes!r} do not cover dimension "
                f"{dimension} of {self.source!r} with size "
                f"{tensor.shape[dimension]}.")
        pieces = tensor.split(self.sizes, dim=dimension)
        return dict(zip(self.targets, pieces))


@dataclass(frozen=True)
class ConcatenateTensors(TensorRule):
    """Concatenate multiple upstream tensors into one canonical tensor."""

    sources: tuple[str, ...]
    target: str
    dimension: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sources",
            _names(self.sources, field_name="sources", minimum=2),
        )
        _name(self.target, field_name="target")
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise TypeError("`dimension` must be an integer.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return self.sources

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self.target, )

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        tensors = tuple(resolver.get(source) for source in self.sources)
        if not tensors or tensors[0].ndim == 0:
            raise CheckpointCompatibilityError(f"Cannot concatenate scalar tensors for {self.target!r}.")
        rank = tensors[0].ndim
        dimension = self.dimension % rank
        if any(tensor.ndim != rank for tensor in tensors):
            raise CheckpointCompatibilityError(
                f"Concatenation sources for {self.target!r} have different ranks.")
        reference = tensors[0].shape
        if any(any(left != right for index, (left, right) in enumerate(zip(reference, tensor.shape))
                   if index != dimension) for tensor in tensors[1:]):
            raise CheckpointCompatibilityError(
                f"Concatenation sources for {self.target!r} have incompatible "
                "shapes.")
        try:
            import torch
            return {self.target: torch.cat(tensors, dim=dimension)}
        except RuntimeError as error:
            raise CheckpointCompatibilityError(
                f"Could not concatenate sources for {self.target!r}: {error}.") from error


@dataclass(frozen=True)
class ReshapeTensor(TensorRule):
    """Reshape a source tensor without changing element order."""

    source: str
    target: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.source, field_name="source")
        _name(self.target, field_name="target")
        if (not isinstance(self.shape, tuple) or not self.shape or
                any(isinstance(value, bool) or not isinstance(value, int) or value == 0 or value < -1
                    for value in self.shape) or self.shape.count(-1) > 1):
            raise ValueError("`shape` must contain non-zero integers and at most one -1.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self.target, )

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        tensor = resolver.get(self.source)
        known_size = prod((value for value in self.shape if value != -1), start=1)
        if -1 not in self.shape and tensor.numel() != known_size:
            raise CheckpointCompatibilityError(
                f"Cannot reshape {self.source!r} with {tensor.numel()} elements "
                f"to {self.shape!r}.")
        if -1 in self.shape and (known_size == 0 or tensor.numel() % known_size != 0):
            raise CheckpointCompatibilityError(
                f"Cannot infer reshape dimension for {self.source!r} from "
                f"{tensor.numel()} elements and shape {self.shape!r}.")
        return {self.target: tensor.reshape(self.shape)}


@dataclass(frozen=True)
class SqueezeTensor(TensorRule):
    """Remove one or more explicitly singleton dimensions."""

    source: str
    target: str
    dimensions: tuple[int, ...]

    def __post_init__(self) -> None:
        _name(self.source, field_name="source")
        _name(self.target, field_name="target")
        if (not isinstance(self.dimensions, tuple) or not self.dimensions or
                any(isinstance(value, bool) or not isinstance(value, int) for value in self.dimensions)):
            raise ValueError("`dimensions` must be a non-empty tuple of integers.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self.target, )

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        tensor = resolver.get(self.source)
        normalized = tuple(sorted({value % tensor.ndim for value in self.dimensions}, reverse=True))
        for dimension in normalized:
            if tensor.shape[dimension] != 1:
                raise CheckpointCompatibilityError(
                    f"Cannot squeeze non-singleton dimension {dimension} of "
                    f"{self.source!r} with shape {tuple(tensor.shape)!r}.")
            tensor = tensor.squeeze(dimension)
        return {self.target: tensor}


@dataclass(frozen=True)
class CastTensor(TensorRule):
    """Convert one source tensor to an explicit PyTorch dtype."""

    source: str
    target: str
    dtype: Any

    def __post_init__(self) -> None:
        _name(self.source, field_name="source")
        _name(self.target, field_name="target")
        if self.dtype is None:
            raise ValueError("`dtype` cannot be None.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return (self.source, )

    @property
    def target_names(self) -> tuple[str, ...]:
        return (self.target, )

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        return {self.target: resolver.get(self.source).to(dtype=self.dtype)}


@dataclass(frozen=True)
class TransformTensor(TensorRule):
    """Apply an audited pure tensor callback for exceptional mappings.

    Architecture adapters should prefer the explicit rules above.  This
    escape hatch exists for mathematically named operations such as
    interleaving rotary projections.  Callers must give it a stable
    ``operation`` name so a plan remains inspectable and serializable in
    provenance logs.
    """

    sources: tuple[str, ...]
    targets: tuple[str, ...]
    operation: str
    transform: Callable[[tuple[Any, ...]], Sequence[Any]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sources",
            _names(self.sources, field_name="sources"),
        )
        object.__setattr__(
            self,
            "targets",
            _names(self.targets, field_name="targets"),
        )
        _name(self.operation, field_name="operation")
        if not callable(self.transform):
            raise TypeError("`transform` must be callable.")

    @property
    def source_names(self) -> tuple[str, ...]:
        return self.sources

    @property
    def target_names(self) -> tuple[str, ...]:
        return self.targets

    def apply(self, resolver: TensorResolver) -> Mapping[str, Any]:
        try:
            values = tuple(self.transform(tuple(resolver.get(source) for source in self.sources)))
        except CheckpointCompatibilityError:
            raise
        except Exception as error:
            raise CheckpointCompatibilityError(
                f"Tensor transform {self.operation!r} failed: {error}.") from error
        if len(values) != len(self.targets):
            raise CheckpointCompatibilityError(
                f"Tensor transform {self.operation!r} returned {len(values)} "
                f"values for {len(self.targets)} targets.")
        return dict(zip(self.targets, values))


@dataclass(frozen=True)
class TensorPlan:
    """Validated conversion plan for one checkpoint namespace."""

    rules: tuple[TensorRule, ...]
    ignored_source_patterns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rules = tuple(self.rules)
        if any(not isinstance(rule, TensorRule) for rule in rules):
            raise TypeError("`rules` must contain TensorRule instances.")
        object.__setattr__(self, "rules", rules)
        targets = [target for rule in rules for target in rule.target_names]
        duplicates = sorted({target for target in targets if targets.count(target) > 1})
        if duplicates:
            raise ValueError(f"Tensor plan writes duplicate targets: {duplicates!r}.")
        patterns = tuple(self.ignored_source_patterns)
        if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
            raise ValueError("Ignored source patterns must be non-empty strings.")
        object.__setattr__(self, "ignored_source_patterns", patterns)

    @cached_property
    def source_names(self) -> frozenset[str]:
        return frozenset(source for rule in self.rules for source in rule.source_names)

    @cached_property
    def target_names(self) -> frozenset[str]:
        return frozenset(target for rule in self.rules for target in rule.target_names)

    def materialize(
        self,
        source: TensorSource | Mapping[str, Any],
    ) -> tuple[dict[str, Any], frozenset[str]]:
        """Return converted tensors and the exact set of consumed sources."""
        normalized_source: TensorSource
        if isinstance(source, Mapping):
            normalized_source = MappingTensorSource(source)
        elif isinstance(source, TensorSource):
            normalized_source = source
        else:
            raise TypeError("`source` must implement TensorSource or be a state-dict mapping.")
        resolver = TensorResolver(normalized_source)
        output: dict[str, Any] = {}
        for rule in self.rules:
            result = rule.apply(resolver)
            if set(result) != set(rule.target_names):
                raise CheckpointCompatibilityError(
                    f"{type(rule).__name__} emitted targets "
                    f"{sorted(result)!r}; expected "
                    f"{sorted(rule.target_names)!r}.")
            output.update(result)
        return output, frozenset(resolver.consumed)


__all__ = [
    "CastTensor",
    "ConcatenateTensors",
    "CopyTensor",
    "MappingTensorSource",
    "ReshapeTensor",
    "SplitTensor",
    "SqueezeTensor",
    "TensorPlan",
    "TensorResolver",
    "TensorRule",
    "TensorSource",
    "TransformTensor",
    "TransposeTensor",
]
