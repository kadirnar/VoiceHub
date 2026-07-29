"""Strict checkpoint adapters for native VoiceHub architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.checkpointing.transforms import (
    CopyTensor,
    MappingTensorSource,
    TensorPlan,
    TensorSource,
)


@dataclass(frozen=True)
class TensorShapeMismatch:
    """One canonical tensor whose checkpoint and model shapes differ."""

    name: str
    checkpoint_shape: tuple[int, ...]
    model_shape: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointCompatibilityReport:
    """Complete, deterministic result of adapting one checkpoint."""

    architecture: str
    adapter: str
    loaded: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    shape_mismatches: tuple[TensorShapeMismatch, ...] = ()
    unused_sources: tuple[str, ...] = ()
    ignored_sources: tuple[str, ...] = ()

    @property
    def is_compatible(self) -> bool:
        return not (
            self.missing
            or self.unexpected
            or self.shape_mismatches
            or self.unused_sources
        )

    def summary(self) -> str:
        if self.is_compatible:
            return (
                f"{self.adapter} loaded {len(self.loaded)} tensors for "
                f"{self.architecture}."
            )
        details = []
        for name, values in (
            ("missing", self.missing),
            ("unexpected", self.unexpected),
            ("shape mismatches", self.shape_mismatches),
            ("unused sources", self.unused_sources),
        ):
            if values:
                details.append(f"{len(values)} {name}")
        return (
            f"{self.adapter} is incompatible with {self.architecture}: "
            f"{', '.join(details)}."
        )

    def require_compatible(self) -> None:
        if not self.is_compatible:
            raise CheckpointCompatibilityError(self.summary())


class CheckpointAdapter(ABC):
    """Convert one upstream checkpoint namespace into a native model.

    Implementations are deliberately stateless.  Adapter IDs and versions are
    persisted in ``voicehub_manifest.json`` and must change whenever tensor
    semantics change.
    """

    architecture_id: str
    adapter_id: str
    adapter_version: str

    @classmethod
    def _validate_identity(cls) -> None:
        for name in ("architecture_id", "adapter_id", "adapter_version"):
            value = getattr(cls, name, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"Checkpoint adapter {cls.__name__} must declare "
                    f"a non-empty `{name}`."
                )

    @property
    def qualified_id(self) -> str:
        type(self)._validate_identity()
        return f"{self.adapter_id}@{self.adapter_version}"

    @abstractmethod
    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        """Return whether this adapter recognizes an immutable artifact."""

    @abstractmethod
    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        """Build the explicit tensor conversion plan for ``config``."""

    def _source(
        self,
        source: TensorSource | Mapping[str, Any],
    ) -> TensorSource:
        if isinstance(source, Mapping):
            return MappingTensorSource(source)
        if isinstance(source, TensorSource):
            return source
        raise TypeError(
            "`source` must implement TensorSource or be a state-dict mapping."
        )

    def inspect(
        self,
        model: Any,
        source: TensorSource | Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> tuple[CheckpointCompatibilityReport, dict[str, Any]]:
        """Convert and validate all tensors without mutating ``model``."""
        type(self)._validate_identity()
        if not callable(getattr(model, "state_dict", None)):
            raise TypeError("Checkpoint target must expose state_dict().")
        plan = self.tensor_plan(config)
        normalized_source = self._source(source)
        converted, consumed = plan.materialize(normalized_source)
        model_state = model.state_dict()
        if not isinstance(model_state, Mapping):
            raise TypeError("Model state_dict() must return a mapping.")

        expected = set(model_state)
        provided = set(converted)
        missing = tuple(sorted(expected - provided))
        unexpected = tuple(sorted(provided - expected))
        shape_mismatches = tuple(
            TensorShapeMismatch(
                name=name,
                checkpoint_shape=tuple(converted[name].shape),
                model_shape=tuple(model_state[name].shape),
            )
            for name in sorted(expected & provided)
            if tuple(converted[name].shape) != tuple(model_state[name].shape)
        )
        mismatch_names = {item.name for item in shape_mismatches}
        loaded = tuple(sorted(expected & provided - mismatch_names))
        available_sources = set(normalized_source.keys())
        ignored_sources = {
            name
            for name in available_sources - consumed
            if any(
                fnmatchcase(name, pattern)
                for pattern in plan.ignored_source_patterns
            )
        }
        unused_sources = tuple(
            sorted(available_sources - consumed - ignored_sources)
        )
        report = CheckpointCompatibilityReport(
            architecture=self.architecture_id,
            adapter=self.qualified_id,
            loaded=loaded,
            missing=missing,
            unexpected=unexpected,
            shape_mismatches=shape_mismatches,
            unused_sources=unused_sources,
            ignored_sources=tuple(sorted(ignored_sources)),
        )
        return report, converted

    def load(
        self,
        model: Any,
        source: TensorSource | Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate then atomically copy compatible tensors into ``model``."""
        if not callable(getattr(model, "load_state_dict", None)):
            raise TypeError("Checkpoint target must expose load_state_dict().")
        report, converted = self.inspect(model, source, config)
        if strict:
            report.require_compatible()
        mismatch_names = {item.name for item in report.shape_mismatches}
        loadable = {
            name: tensor
            for name, tensor in converted.items()
            if name in report.loaded and name not in mismatch_names
        }
        model.load_state_dict(loadable, strict=False)
        return report

    def load_streaming(
        self,
        model: Any,
        source: TensorSource | Mapping[str, Any],
        config: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> CheckpointCompatibilityReport:
        """Validate then copy a rename-only plan one tensor at a time.

        Large speech checkpoints commonly use a pure source-to-native rename
        plan.  This path validates all names and shapes from Safetensors
        headers before mutation, then retains at most one checkpoint tensor in
        memory.  Plans requiring split, concatenate, transpose, cast, reshape,
        or custom transformations must use :meth:`load`.
        """
        type(self)._validate_identity()
        if not callable(getattr(model, "state_dict", None)):
            raise TypeError("Checkpoint target must expose state_dict().")
        plan = self.tensor_plan(config)
        if any(not isinstance(rule, CopyTensor) for rule in plan.rules):
            raise CheckpointCompatibilityError(
                "Streaming checkpoint loading supports CopyTensor rules only."
            )
        normalized_source = self._source(source)
        model_state = model.state_dict()
        if not isinstance(model_state, Mapping):
            raise TypeError("Model state_dict() must return a mapping.")

        mapping = {
            rule.target: rule.source
            for rule in plan.rules
        }
        expected = set(model_state)
        provided = set(mapping)
        available_sources = set(normalized_source.keys())
        consumed = set(mapping.values())
        missing = tuple(sorted(expected - provided))
        unexpected = tuple(sorted(provided - expected))
        ignored_sources = {
            name
            for name in available_sources - consumed
            if any(
                fnmatchcase(name, pattern)
                for pattern in plan.ignored_source_patterns
            )
        }
        unused_sources = tuple(
            sorted(available_sources - consumed - ignored_sources)
        )

        tensor_shape = getattr(normalized_source, "tensor_shape", None)
        shape_mismatches: list[TensorShapeMismatch] = []
        for target_name in sorted(expected & provided):
            source_name = mapping[target_name]
            if source_name not in available_sources:
                continue
            if callable(tensor_shape):
                checkpoint_shape = tuple(tensor_shape(source_name))
            else:
                checkpoint_shape = tuple(
                    normalized_source.get_tensor(source_name).shape
                )
            model_shape = tuple(model_state[target_name].shape)
            if checkpoint_shape != model_shape:
                shape_mismatches.append(
                    TensorShapeMismatch(
                        name=target_name,
                        checkpoint_shape=checkpoint_shape,
                        model_shape=model_shape,
                    )
                )
        absent_sources = consumed - available_sources
        missing = tuple(
            sorted(set(missing) | {
                target
                for target, source_name in mapping.items()
                if source_name in absent_sources
            })
        )
        mismatch_names = {item.name for item in shape_mismatches}
        loaded = tuple(
            sorted(
                expected
                & provided
                - mismatch_names
                - set(missing)
            )
        )
        report = CheckpointCompatibilityReport(
            architecture=self.architecture_id,
            adapter=self.qualified_id,
            loaded=loaded,
            missing=missing,
            unexpected=unexpected,
            shape_mismatches=tuple(shape_mismatches),
            unused_sources=unused_sources,
            ignored_sources=tuple(sorted(ignored_sources)),
        )
        if strict:
            report.require_compatible()

        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - invariant
            raise RuntimeError(
                "Native checkpoint loading requires PyTorch."
            ) from error
        with torch.no_grad():
            for target_name in report.loaded:
                source_tensor = normalized_source.get_tensor(
                    mapping[target_name]
                )
                target_tensor = model_state[target_name]
                target_tensor.copy_(
                    source_tensor,
                    non_blocking=False,
                )
                del source_tensor
        return report


__all__ = [
    "CheckpointAdapter",
    "CheckpointCompatibilityReport",
    "TensorShapeMismatch",
]
