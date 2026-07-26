"""Framework-independent contracts for architecture-specific training recipes.

The objects in this module intentionally contain no tensor-framework
imports. They describe *how* an adapter should call a backend, not how a
particular objective is implemented.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class TrainingSupport(str, Enum):
    """How completely a model's source integration supports fine-tuning.

    ``NATIVE``     The integrated runtime exposes a differentiable,
    backend-native loss. ``PREPROCESSED``     The adapter can train from
    backend-shaped tensors, while raw     text/audio preprocessing
    remains outside the generic path. ``CUSTOM``     The source tree
    contains a recipe whose orchestration or loss requires     a
    specialized adapter. ``INFERENCE_ONLY``     The integrated runtime
    has no verified gradient path. This is explicit     metadata, not a
    claim that the architecture can never be trained.
    """

    NATIVE = "native"
    PREPROCESSED = "preprocessed"
    CUSTOM = "custom"
    INFERENCE_ONLY = "inference-only"

    def __str__(self) -> str:
        return self.value

    @property
    def is_trainable(self) -> bool:
        """Whether the profile advertises a generic or specialized path."""
        return self is not TrainingSupport.INFERENCE_ONLY

    @classmethod
    def coerce(cls, value: TrainingSupport | str) -> TrainingSupport:
        """Validate and normalize a public support value."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Training support must be a TrainingSupport value or string.")
        normalized = value.strip().lower().replace("_", "-")
        aliases = {
            "inference": cls.INFERENCE_ONLY.value,
            "inferenceonly": cls.INFERENCE_ONLY.value,
            "source-native": cls.NATIVE.value,
            "backend-native": cls.NATIVE.value,
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown training support {value!r}. Expected one of: {choices}.") from exc


class TrainingRecipeKind(str, Enum):
    """High-level orchestration shape of a model training profile."""

    SINGLE_PHASE = "single-phase"
    MULTI_PHASE = "multi-phase"
    ADVERSARIAL = "adversarial"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: TrainingRecipeKind | str) -> TrainingRecipeKind:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Training recipe kind must be an enum value or string.")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown training recipe kind {value!r}. Expected: {choices}.") from exc


class TrainingPhaseKind(str, Enum):
    """Semantic role of a phase within a training recipe."""

    OBJECTIVE = "objective"
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"
    DURATION_DISCRIMINATOR = "duration-discriminator"
    AUXILIARY = "auxiliary"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def coerce(cls, value: TrainingPhaseKind | str) -> TrainingPhaseKind:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Training phase kind must be an enum value or string.")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(f"Unknown training phase kind {value!r}. Expected: {choices}.") from exc


def _string_tuple(value: Iterable[str] | str, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value, )
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise TypeError(f"{field_name} must be an iterable of strings.") from exc
    normalized = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name} must contain non-empty strings.")
        normalized.append(item.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate values.")
    return tuple(normalized)


def _pair_tuple(
    value: Mapping[str, Any] | Iterable[tuple[str, Any]],
    *,
    field_name: str,
) -> tuple[tuple[str, Any], ...]:
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{field_name} entries must be two-item pairs.")
        name, target = item
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{field_name} keys must be non-empty strings.")
        name = name.strip()
        if name in seen:
            raise ValueError(f"{field_name} contains duplicate key {name!r}.")
        seen.add(name)
        normalized.append((name, target))
    return tuple(normalized)


@dataclass(frozen=True)
class TrainingPhaseSpec:
    """One independently callable and optimizable phase of a TTS recipe.

    Input aliases are ``(batch_name, backend_name)`` pairs. Component
    paths are resolved from the public VoiceHub wrapper. A single
    optimizer name routes every listed component to one optimizer; one
    name per component is also supported.
    """

    name: str
    component_paths: tuple[str, ...] = ()
    optimizer_names: tuple[str, ...] = ()
    forward_component: str | None = None
    forward_method: str = "forward"
    label_names: tuple[str, ...] = ("labels", "targets", "target")
    prediction_keys: tuple[str, ...] = (
        "logits",
        "predictions",
        "audio_values",
        "waveform",
    )
    loss_keys: tuple[str, ...] = ("loss", "total_loss")
    loss_weights: tuple[tuple[str, float], ...] = ()
    input_aliases: tuple[tuple[str, str], ...] = ()
    required_inputs: tuple[str, ...] = ()
    frequency: int = 1
    offset: int = 0
    fallback_objective: str | None = None
    kind: TrainingPhaseKind = TrainingPhaseKind.OBJECTIVE
    detach_inputs: tuple[str, ...] = ()
    frozen_component_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Training phase names must be non-empty strings.")
        object.__setattr__(self, "name", self.name.strip())

        for field_name in (
                "component_paths",
                "optimizer_names",
                "label_names",
                "prediction_keys",
                "loss_keys",
                "required_inputs",
                "detach_inputs",
                "frozen_component_paths",
        ):
            object.__setattr__(
                self,
                field_name,
                _string_tuple(getattr(self, field_name), field_name=field_name),
            )

        if self.forward_component is not None:
            if not isinstance(self.forward_component, str) or not self.forward_component.strip():
                raise ValueError("forward_component must be a non-empty path or None.")
            object.__setattr__(self, "forward_component", self.forward_component.strip())
        if not isinstance(self.forward_method, str) or not self.forward_method.strip():
            raise ValueError("forward_method must be a non-empty method path.")
        object.__setattr__(self, "forward_method", self.forward_method.strip())

        aliases = _pair_tuple(self.input_aliases, field_name="input_aliases")
        normalized_aliases = []
        alias_targets = set()
        for source, target in aliases:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("input_aliases targets must be non-empty strings.")
            target = target.strip()
            if target in alias_targets:
                raise ValueError(f"input_aliases route more than one input to {target!r}.")
            alias_targets.add(target)
            normalized_aliases.append((source, target))
        object.__setattr__(self, "input_aliases", tuple(normalized_aliases))

        weights = _pair_tuple(self.loss_weights, field_name="loss_weights")
        normalized_weights = []
        for loss_name, weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("Loss weights must be real numbers.")
            if not math.isfinite(float(weight)):
                raise ValueError("Loss weights must be finite.")
            normalized_weights.append((loss_name, float(weight)))
        object.__setattr__(self, "loss_weights", tuple(normalized_weights))

        if isinstance(self.frequency, bool) or not isinstance(self.frequency, int):
            raise TypeError("Phase frequency must be an integer.")
        if self.frequency <= 0:
            raise ValueError("Phase frequency must be greater than zero.")
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise TypeError("Phase offset must be an integer.")
        if not 0 <= self.offset < self.frequency:
            raise ValueError("Phase offset must be between zero and frequency - 1.")

        if self.fallback_objective is not None:
            if (not isinstance(self.fallback_objective, str) or not self.fallback_objective.strip()):
                raise ValueError("fallback_objective must be a non-empty name or None.")
            object.__setattr__(
                self,
                "fallback_objective",
                self.fallback_objective.strip().lower().replace("-", "_"),
            )
        object.__setattr__(self, "kind", TrainingPhaseKind.coerce(self.kind))
        if (self.kind in (
                TrainingPhaseKind.GENERATOR,
                TrainingPhaseKind.DISCRIMINATOR,
                TrainingPhaseKind.DURATION_DISCRIMINATOR,
        ) and not self.optimizer_names):
            raise ValueError(f"{self.kind.value} phases must declare optimizer_names.")

        optimizer_count = len(self.optimizer_names)
        component_count = len(self.component_paths)
        if optimizer_count > 1 and optimizer_count != component_count:
            raise ValueError(
                "optimizer_names must contain one name for the whole phase or "
                "one name per component path.")

    @property
    def optimizer_name(self) -> str | None:
        """Return the phase optimizer when it routes to exactly one name."""
        unique = tuple(dict.fromkeys(self.optimizer_names))
        return unique[0] if len(unique) == 1 else None

    @property
    def input_alias_map(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.input_aliases))

    @property
    def component_optimizer_routes(self) -> tuple[tuple[str, str], ...]:
        """Return explicit ``(component_path, optimizer_name)`` routes."""
        if not self.optimizer_names:
            return ()
        if len(self.optimizer_names) == 1:
            return tuple((path, self.optimizer_names[0]) for path in self.component_paths)
        return tuple(zip(self.component_paths, self.optimizer_names))

    def is_scheduled(self, step: int) -> bool:
        """Return whether this phase is due at a zero-based recipe step."""
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("Training phase planning requires an integer step.")
        if step < 0:
            raise ValueError("Training phase planning requires a non-negative step.")
        return step % self.frequency == self.offset


@dataclass(frozen=True)
class TrainingContext:
    """Runtime context passed through phase preparation and execution hooks."""

    phase: TrainingPhaseSpec
    inputs: Mapping[str, Any]
    step: int | None = None
    epoch: float | None = None
    is_training: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, TrainingPhaseSpec):
            raise TypeError("TrainingContext.phase must be a TrainingPhaseSpec.")
        if not isinstance(self.inputs, Mapping):
            raise TypeError("TrainingContext.inputs must be a mapping.")
        if self.step is not None:
            if isinstance(self.step, bool) or not isinstance(self.step, int):
                raise TypeError("TrainingContext.step must be an integer or None.")
            if self.step < 0:
                raise ValueError("TrainingContext.step must be non-negative.")
        if self.epoch is not None and (isinstance(self.epoch, bool) or not isinstance(self.epoch,
                                                                                      (int, float))):
            raise TypeError("TrainingContext.epoch must be numeric or None.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("TrainingContext.metadata must be a mapping.")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def phase_name(self) -> str:
        return self.phase.name

    @property
    def optimizer_names(self) -> tuple[str, ...]:
        return self.phase.optimizer_names

    def with_inputs(self, inputs: Mapping[str, Any]) -> TrainingContext:
        """Return an equivalent context carrying prepared backend inputs."""
        return TrainingContext(
            phase=self.phase,
            inputs=inputs,
            step=self.step,
            epoch=self.epoch,
            is_training=self.is_training,
            metadata=self.metadata,
        )
