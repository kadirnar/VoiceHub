"""Request-scoped residual caching for repeated diffusion/flow blocks.

The implementation adapts the block-boundary idea used by image DiT
caches to VoiceHub-owned speech models without depending on Diffusers or
Cache-DiT. It is deliberately approximate and opt-in: training,
gradient-enabled calls, unsupported block layouts, and incompatible
cache entries always execute the original block sequence.

Architectures integrate through :class:`DiffusionCacheMixin`.  The mixin
does not own parameters, buffers, or child modules, so enabling the
cache cannot change checkpoint keys or module topology.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from itertools import count
from threading import RLock
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from voicehub.optimization.capabilities import OptimizationCapabilities, OptimizationContext, OptimizationMode
from voicehub.optimization.passes import (
    OptimizationCompatibilityError,
    OptimizationError,
    OptimizationPass,
    PassResult,
    canonical_json_string,
)
from voicehub.optimization.protocols import OptimizationModuleRoot


class DiffusionCacheError(OptimizationError):
    """Base failure raised by diffusion-cache configuration or execution."""


class DiffusionCacheCompatibilityError(
        ValueError,
        DiffusionCacheError,
):
    """A requested cache has no compatible architecture-owned block surface."""


class DiffusionCachePolicy(str, Enum):
    """Whether approximate diffusion caching is disabled, optional, or
    required."""

    DISABLED = "disabled"
    AUTO = "auto"
    REQUIRED = "required"

    @classmethod
    def coerce(
        cls,
        value: DiffusionCachePolicy | str | bool,
    ) -> DiffusionCachePolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.REQUIRED if value else cls.DISABLED
        if not isinstance(value, str):
            raise TypeError("`diffusion_cache` must be a boolean, string, or "
                            "DiffusionCachePolicy.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "false": cls.DISABLED.value,
            "off": cls.DISABLED.value,
            "none": cls.DISABLED.value,
            "true": cls.REQUIRED.value,
            "on": cls.REQUIRED.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion-cache policy {value!r}; expected one of: "
                f"{choices}.") from error


class DiffusionCachePredictor(str, Enum):
    """Approximation used for a skipped middle-block residual."""

    REUSE = "reuse"
    TAYLOR = "taylor"

    @classmethod
    def coerce(
        cls,
        value: DiffusionCachePredictor | str,
    ) -> DiffusionCachePredictor:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`predictor` must be a string or DiffusionCachePredictor.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "dbcache": cls.REUSE.value,
            "taylorseer": cls.TAYLOR.value,
            "linear": cls.TAYLOR.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion-cache predictor {value!r}; expected one "
                f"of: {choices}.") from error


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer.")
    return value


def _bounded_step_limit(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError(f"`{name}` must be -1 or a non-negative integer.")
    return value


@dataclass(frozen=True, slots=True)
class DiffusionCacheConfig:
    """Serializable Cache-DiT-style settings for a VoiceHub DiT block list.

    ``front_blocks`` are always evaluated and provide the change probe.
    ``back_blocks`` are always evaluated after either computing or
    predicting the middle residual.  The first-order ``taylor``
    predictor uses only two previously *computed* residuals; it never
    chains predictions.

    The feature is approximate by construction.  Merely constructing
    this object does not enable it; callers must explicitly select
    :class:`DiffusionCachePolicy.AUTO` or ``REQUIRED`` in the universal
    TTS optimizer, or apply :class:`DiffusionCachePass` directly.
    """

    front_blocks: int = 1
    back_blocks: int = 0
    residual_diff_threshold: float = 0.08
    warmup_steps: int = 2
    max_cached_steps: int = -1
    max_consecutive_cached_steps: int = 3
    max_accumulated_relative_error: float | None = None
    predictor: DiffusionCachePredictor | str = DiffusionCachePredictor.REUSE
    compute_step_mask: tuple[bool, ...] = ()
    synchronize_distributed: bool = True
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        front = _non_negative_integer(self.front_blocks, name="front_blocks")
        if front == 0:
            raise ValueError("`front_blocks` must be at least one.")
        back = _non_negative_integer(self.back_blocks, name="back_blocks")
        warmup = _non_negative_integer(self.warmup_steps, name="warmup_steps")
        maximum = _bounded_step_limit(
            self.max_cached_steps,
            name="max_cached_steps",
        )
        consecutive = _bounded_step_limit(
            self.max_consecutive_cached_steps,
            name="max_consecutive_cached_steps",
        )
        if (isinstance(self.residual_diff_threshold, bool) or not isinstance(self.residual_diff_threshold,
                                                                             (int, float))):
            raise TypeError("`residual_diff_threshold` must be a real number.")
        threshold = float(self.residual_diff_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("`residual_diff_threshold` must be finite and non-negative.")
        accumulated = self.max_accumulated_relative_error
        if accumulated is not None:
            accumulated_is_real = isinstance(accumulated, (int, float))
            if isinstance(accumulated, bool) or not accumulated_is_real:
                raise TypeError("`max_accumulated_relative_error` must be a real number "
                                "or None.")
            accumulated = float(accumulated)
            if not math.isfinite(accumulated) or accumulated <= 0:
                raise ValueError("`max_accumulated_relative_error` must be finite and "
                                 "greater than zero.")
        predictor = DiffusionCachePredictor.coerce(self.predictor)
        if isinstance(self.compute_step_mask, (str, bytes)):
            raise TypeError("`compute_step_mask` must be an iterable of booleans.")
        try:
            step_mask = tuple(self.compute_step_mask)
        except TypeError as error:
            raise TypeError("`compute_step_mask` must be an iterable of booleans.") from error
        if any(not isinstance(item, bool) for item in step_mask):
            raise TypeError("`compute_step_mask` may contain only booleans.")
        if not isinstance(self.synchronize_distributed, bool):
            raise TypeError("`synchronize_distributed` must be a boolean.")
        if (isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float))):
            raise TypeError("`epsilon` must be a real number.")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("`epsilon` must be finite and greater than zero.")
        object.__setattr__(self, "front_blocks", front)
        object.__setattr__(self, "back_blocks", back)
        object.__setattr__(self, "residual_diff_threshold", threshold)
        object.__setattr__(self, "warmup_steps", warmup)
        object.__setattr__(self, "max_cached_steps", maximum)
        object.__setattr__(self, "max_consecutive_cached_steps", consecutive)
        object.__setattr__(
            self,
            "max_accumulated_relative_error",
            accumulated,
        )
        object.__setattr__(self, "predictor", predictor)
        object.__setattr__(self, "compute_step_mask", step_mask)
        object.__setattr__(self, "epsilon", epsilon)
        canonical_json_string(
            self.to_dict(),
            path="diffusion cache configuration",
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        **overrides: Any,
    ) -> DiffusionCacheConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Diffusion cache configuration must be a mapping.")
        output = dict(values)
        output.update(overrides)
        return cls(**output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "front_blocks": self.front_blocks,
            "back_blocks": self.back_blocks,
            "residual_diff_threshold": self.residual_diff_threshold,
            "warmup_steps": self.warmup_steps,
            "max_cached_steps": self.max_cached_steps,
            "max_consecutive_cached_steps": self.max_consecutive_cached_steps,
            "max_accumulated_relative_error": self.max_accumulated_relative_error,
            "predictor": self.predictor.value,
            "compute_step_mask": list(self.compute_step_mask),
            "synchronize_distributed": self.synchronize_distributed,
            "epsilon": self.epsilon,
        }


def coerce_diffusion_cache_config(
        value: DiffusionCacheConfig | Mapping[str, Any] | None) -> DiffusionCacheConfig:
    if value is None:
        return DiffusionCacheConfig()
    if isinstance(value, DiffusionCacheConfig):
        return value
    if isinstance(value, Mapping):
        return DiffusionCacheConfig.from_dict(value)
    raise TypeError("`diffusion_cache_config` must be a DiffusionCacheConfig, mapping, "
                    "or None.")


@dataclass(slots=True)
class _CacheLaneState:
    step: int = 0
    cached_steps: int = 0
    consecutive_cached_steps: int = 0
    accumulated_relative_error: float = 0.0
    probe: Tensor | None = None
    middle_residual: Tensor | None = None
    previous_middle_residual: Tensor | None = None
    computed_step: int | None = None
    previous_computed_step: int | None = None


@dataclass(slots=True)
class _MutableCacheStats:
    calls: int = 0
    computed_steps: int = 0
    cached_steps: int = 0
    bypassed_steps: int = 0
    warmup_misses: int = 0
    threshold_misses: int = 0
    limit_misses: int = 0
    invalidations: int = 0
    sessions: int = 0
    last_relative_difference: float | None = None
    maximum_relative_difference: float | None = None

    def snapshot(self) -> dict[str, int | float | None]:
        evaluated = self.computed_steps + self.cached_steps
        hit_rate = (self.cached_steps / evaluated if evaluated else 0.0)
        return {
            "calls": self.calls,
            "computed_steps": self.computed_steps,
            "cached_steps": self.cached_steps,
            "bypassed_steps": self.bypassed_steps,
            "warmup_misses": self.warmup_misses,
            "threshold_misses": self.threshold_misses,
            "limit_misses": self.limit_misses,
            "invalidations": self.invalidations,
            "sessions": self.sessions,
            "hit_rate": hit_rate,
            "last_relative_difference": self.last_relative_difference,
            "maximum_relative_difference": self.maximum_relative_difference,
        }


def _compatible_tensor(left: Tensor | None, right: Tensor) -> bool:
    return (
        left is not None and left.shape == right.shape and left.dtype == right.dtype and
        left.device == right.device)


def _expanded_valid_mask(mask: Tensor, value: Tensor) -> Tensor:
    if mask.device != value.device:
        mask = mask.to(device=value.device)
    mask = mask.to(dtype=torch.bool)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(mask, value.shape)
    except RuntimeError as error:
        raise ValueError(
            "Diffusion-cache valid mask is not broadcastable to hidden "
            f"states: mask={tuple(mask.shape)}, states={tuple(value.shape)}.") from error


def _relative_l1(
    previous: Tensor,
    current: Tensor,
    *,
    valid_mask: Tensor | None,
    epsilon: float,
) -> Tensor:
    difference = (previous.float() - current.float()).abs()
    denominator = previous.float().abs()
    if valid_mask is not None:
        expanded = _expanded_valid_mask(valid_mask, current)
        difference = difference.masked_fill(~expanded, 0)
        denominator = denominator.masked_fill(~expanded, 0)
        if current.ndim >= 3:
            reduce_dims = tuple(range(1, current.ndim))
            counts = expanded.sum(dim=reduce_dims).clamp_min(1)
            numerator = difference.sum(dim=reduce_dims) / counts
            baseline = denominator.sum(dim=reduce_dims) / counts
            return (numerator / baseline.clamp_min(epsilon)).amax()
        count_value = expanded.sum().clamp_min(1)
        return (difference.sum() / count_value / (denominator.sum() / count_value).clamp_min(epsilon))
    if current.ndim >= 3:
        reduce_dims = tuple(range(1, current.ndim))
        numerator = difference.mean(dim=reduce_dims)
        baseline = denominator.mean(dim=reduce_dims)
        return (numerator / baseline.clamp_min(epsilon)).amax()
    return difference.mean() / denominator.mean().clamp_min(epsilon)


def _synchronize_score(score: Tensor, *, enabled: bool) -> Tensor:
    if not enabled:
        return score
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return score
    is_available = getattr(distributed, "is_available", None)
    is_initialized = getattr(distributed, "is_initialized", None)
    if not callable(is_available) or not is_available():
        return score
    if not callable(is_initialized) or not is_initialized():
        return score
    synchronized = score.detach().clone()
    distributed.all_reduce(
        synchronized,
        op=distributed.ReduceOp.MAX,
    )
    return synchronized


class DiffusionBlockResidualCache:
    """Stateful executor for one or more request/CFG cache lanes."""

    def __init__(self, config: DiffusionCacheConfig):
        self.config = coerce_diffusion_cache_config(config)
        self._states: dict[tuple[str, str], _CacheLaneState] = {}
        self._lock = RLock()
        self._session_counter = count(1)
        self._session: ContextVar[str] = ContextVar(
            f"voicehub_diffusion_cache_session_{id(self)}",
            default="default",
        )
        self._stats = _MutableCacheStats()

    @staticmethod
    def _lane_name(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Diffusion-cache lane names must be non-empty strings.")
        return value.strip()

    @contextmanager
    def session(self) -> Iterator[DiffusionBlockResidualCache]:
        """Isolate cache tensors for one request and release them
        afterwards."""
        with self._lock:
            session_id = f"request-{next(self._session_counter)}"
            self._stats.sessions += 1
        token = self._session.set(session_id)
        try:
            yield self
        finally:
            with self._lock:
                keys = [key for key in self._states if key[0] == session_id]
                for key in keys:
                    del self._states[key]
            self._session.reset(token)

    def reset(self, *, lane: str | None = None) -> None:
        """Invalidate current request state, optionally for one CFG lane."""
        session_id = self._session.get()
        lane_name = None if lane is None else self._lane_name(lane)
        with self._lock:
            keys = [
                key for key in self._states
                if key[0] == session_id and (lane_name is None or key[1] == lane_name)
            ]
            for key in keys:
                del self._states[key]
            self._stats.invalidations += len(keys)

    def reset_all(self) -> None:
        with self._lock:
            invalidated = len(self._states)
            self._states.clear()
            self._stats.invalidations += invalidated

    def stats(self) -> dict[str, int | float | None]:
        with self._lock:
            return self._stats.snapshot()

    @staticmethod
    def _full_run(
        blocks: Sequence[Any],
        hidden_states: Tensor,
        block_call: Callable[[Any, Tensor], Tensor],
    ) -> Tensor:
        for block in blocks:
            hidden_states = block_call(block, hidden_states)
        return hidden_states

    def _predicted_residual(
        self,
        state: _CacheLaneState,
        *,
        current_step: int,
    ) -> Tensor:
        residual = state.middle_residual
        if residual is None:
            raise RuntimeError("A cache hit requires a middle residual.")
        if (self.config.predictor is DiffusionCachePredictor.TAYLOR and
                state.previous_middle_residual is not None and state.computed_step is not None and
                state.previous_computed_step is not None):
            interval = state.computed_step - state.previous_computed_step
            horizon = current_step - state.computed_step
            if interval > 0 and horizon > 0:
                slope = (residual - state.previous_middle_residual) / float(interval)
                return residual + slope * float(horizon)
        return residual

    def run(
        self,
        blocks: Sequence[Any],
        hidden_states: Tensor,
        block_call: Callable[[Any, Tensor], Tensor],
        *,
        lane: str = "default",
        valid_mask: Tensor | None = None,
        training: bool = False,
    ) -> Tensor:
        """Execute one block sequence, predicting only its middle residual."""
        try:
            block_sequence = tuple(blocks)
        except TypeError as error:
            raise TypeError("Diffusion cache blocks must be a finite sequence.") from error
        if not callable(block_call):
            raise TypeError("`block_call` must be callable.")
        if not isinstance(hidden_states, Tensor):
            raise TypeError("Diffusion cache hidden states must be a Tensor.")
        lane_name = self._lane_name(lane)
        with self._lock:
            self._stats.calls += 1
        middle_end = len(block_sequence) - self.config.back_blocks
        unsupported = (
            training or torch.is_grad_enabled() or
            len(block_sequence) <= self.config.front_blocks + self.config.back_blocks or
            self.config.residual_diff_threshold <= 0)
        if unsupported:
            with self._lock:
                self._stats.bypassed_steps += 1
            return self._full_run(
                block_sequence,
                hidden_states,
                block_call,
            )

        session_key = (self._session.get(), lane_name)
        with self._lock:
            state = self._states.setdefault(session_key, _CacheLaneState())
            current_step = state.step
            state.step += 1

        value = hidden_states
        for block in block_sequence[:self.config.front_blocks]:
            value = block_call(block, value)
        probe = value

        with self._lock:
            cached_probe = state.probe
            cached_residual = state.middle_residual
        cache_compatible = (
            _compatible_tensor(cached_probe, probe) and _compatible_tensor(cached_residual, probe))
        force_compute = not cache_compatible
        miss_kind = "invalid" if force_compute else ""
        if not force_compute and current_step < self.config.warmup_steps:
            force_compute = True
            miss_kind = "warmup"
        if (not force_compute and self.config.compute_step_mask and
                current_step < len(self.config.compute_step_mask) and
                self.config.compute_step_mask[current_step]):
            force_compute = True
            miss_kind = "mask"
        if (not force_compute and self.config.max_cached_steps >= 0 and
                state.cached_steps >= self.config.max_cached_steps):
            force_compute = True
            miss_kind = "limit"
        if (not force_compute and self.config.max_consecutive_cached_steps >= 0 and
                state.consecutive_cached_steps >= self.config.max_consecutive_cached_steps):
            force_compute = True
            miss_kind = "limit"

        relative_difference: float | None = None
        if not force_compute:
            score = _relative_l1(
                cached_probe,
                probe,
                valid_mask=valid_mask,
                epsilon=self.config.epsilon,
            )
            score = _synchronize_score(
                score,
                enabled=self.config.synchronize_distributed,
            )
            relative_difference = float(score.detach().item())
            if (not math.isfinite(relative_difference) or
                    relative_difference > self.config.residual_diff_threshold):
                force_compute = True
                miss_kind = "threshold"
            elif (self.config.max_accumulated_relative_error is not None and
                  state.accumulated_relative_error + relative_difference
                  > self.config.max_accumulated_relative_error):
                force_compute = True
                miss_kind = "limit"

        if force_compute:
            middle = probe
            for block in block_sequence[self.config.front_blocks:middle_end]:
                middle = block_call(block, middle)
            if middle.shape != probe.shape:
                # A shape-changing middle is valid model code, but it is not a
                # residual-cache surface.  Preserve eager behavior and leave
                # the lane empty so future calls continue to fail closed.
                with self._lock:
                    state.probe = None
                    state.middle_residual = None
                    state.previous_middle_residual = None
                    state.computed_step = None
                    state.previous_computed_step = None
                    state.consecutive_cached_steps = 0
                    self._stats.bypassed_steps += 1
                value = middle
            else:
                residual = (middle - probe).detach()
                with self._lock:
                    state.previous_middle_residual = state.middle_residual
                    state.previous_computed_step = state.computed_step
                    state.middle_residual = residual
                    state.computed_step = current_step
                    state.probe = probe.detach()
                    state.consecutive_cached_steps = 0
                    state.accumulated_relative_error = 0.0
                    self._stats.computed_steps += 1
                    if miss_kind == "warmup":
                        self._stats.warmup_misses += 1
                    elif miss_kind == "threshold":
                        self._stats.threshold_misses += 1
                    elif miss_kind == "limit":
                        self._stats.limit_misses += 1
                    elif miss_kind == "invalid" and current_step > 0:
                        self._stats.invalidations += 1
                value = middle
        else:
            with self._lock:
                predicted = self._predicted_residual(
                    state,
                    current_step=current_step,
                )
                state.cached_steps += 1
                state.consecutive_cached_steps += 1
                if relative_difference is not None:
                    state.accumulated_relative_error += relative_difference
                state.probe = probe.detach()
                self._stats.cached_steps += 1
            value = probe + predicted

        if relative_difference is not None:
            with self._lock:
                self._stats.last_relative_difference = relative_difference
                maximum = self._stats.maximum_relative_difference
                self._stats.maximum_relative_difference = (
                    relative_difference if maximum is None else max(maximum, relative_difference))
        for block in block_sequence[middle_end:]:
            value = block_call(block, value)
        return value


class DiffusionCacheMixin:
    """Non-module mixin for architecture-owned repeated block sequences."""

    _diffusion_cache_controller: DiffusionBlockResidualCache | None

    def _initialize_diffusion_cache(self) -> None:
        object.__setattr__(self, "_diffusion_cache_controller", None)

    @property
    def diffusion_cache_config(self) -> DiffusionCacheConfig | None:
        controller = getattr(self, "_diffusion_cache_controller", None)
        return None if controller is None else controller.config

    def enable_diffusion_cache(
        self,
        config: DiffusionCacheConfig | Mapping[str, Any] | None = None,
    ) -> DiffusionCacheConfig:
        resolved = coerce_diffusion_cache_config(config)
        object.__setattr__(
            self,
            "_diffusion_cache_controller",
            DiffusionBlockResidualCache(resolved),
        )
        return resolved

    def disable_diffusion_cache(self) -> DiffusionCacheConfig | None:
        previous = self.diffusion_cache_config
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is not None:
            controller.reset_all()
        object.__setattr__(self, "_diffusion_cache_controller", None)
        return previous

    def reset_diffusion_cache(self, *, lane: str | None = None) -> None:
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is not None:
            controller.reset(lane=lane)

    @contextmanager
    def diffusion_cache_session(self) -> Iterator[DiffusionCacheMixin]:
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is None:
            yield self
            return
        with controller.session():
            yield self

    def diffusion_cache_stats(self) -> dict[str, int | float | None]:
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is None:
            return {
                "calls": 0,
                "computed_steps": 0,
                "cached_steps": 0,
                "bypassed_steps": 0,
                "warmup_misses": 0,
                "threshold_misses": 0,
                "limit_misses": 0,
                "invalidations": 0,
                "sessions": 0,
                "hit_rate": 0.0,
                "last_relative_difference": None,
                "maximum_relative_difference": None,
            }
        return controller.stats()

    def _run_diffusion_blocks(
        self,
        blocks: Sequence[Any],
        hidden_states: Tensor,
        block_call: Callable[[Any, Tensor], Tensor],
        *,
        cache_lane: str = "default",
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is None:
            return DiffusionBlockResidualCache._full_run(
                tuple(blocks),
                hidden_states,
                block_call,
            )
        return controller.run(
            blocks,
            hidden_states,
            block_call,
            lane=cache_lane,
            valid_mask=valid_mask,
            training=bool(getattr(self, "training", False)),
        )


@dataclass(frozen=True, slots=True)
class _CacheTarget:
    module: Any
    label: str


@dataclass(frozen=True, slots=True)
class _CachePatch:
    target: _CacheTarget
    previous: DiffusionCacheConfig | None


def _adapter_component_roots(model: Any) -> tuple[tuple[str, nn.Module], ...]:
    candidates: list[tuple[str, Any]] = []
    primary = getattr(model, "primary_model", None)
    if primary is not None:
        candidates.append(("primary_model", primary))
    components = getattr(model, "_components", ())
    if isinstance(components, (tuple, list)):
        for entry in components:
            if (isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[0], str) and
                    entry[0]):
                candidates.append((f"component:{entry[0]}", entry[1]))
    attributes = vars(model) if hasattr(model, "__dict__") else {}
    for name in ("_model", "model", "codec_model"):
        candidates.append((name.removeprefix("_"), attributes.get(name)))
    return tuple((label, candidate) for label, candidate in candidates if isinstance(candidate, nn.Module))


def _module_roots(model: Any) -> tuple[tuple[str, nn.Module], ...]:
    provider = getattr(model, "optimization_module_roots", None)
    if callable(provider):
        declared = provider()
        entries = (tuple(declared.items()) if isinstance(declared, Mapping) else tuple(declared))
        output = []
        labels: set[str] = set()
        identities: set[int] = set()
        for entry in entries:
            if isinstance(entry, OptimizationModuleRoot):
                label, module = entry.label, entry.module
            elif isinstance(entry, (tuple, list)) and len(entry) == 2:
                label, module = entry
            else:
                raise TypeError("optimization_module_roots() entries must be "
                                "(label, module) pairs.")
            if not isinstance(label, str) or not label:
                raise ValueError("Optimization module-root labels must be non-empty strings.")
            if not isinstance(module, nn.Module):
                raise TypeError(f"Optimization module root {label!r} must be an nn.Module.")
            if label in labels or id(module) in identities:
                raise ValueError("Optimization module roots cannot contain duplicate "
                                 "labels or modules.")
            labels.add(label)
            identities.add(id(module))
            output.append((label, module))
        return tuple(output)
    if isinstance(model, nn.Module):
        return (("model", model), )
    return _adapter_component_roots(model)


def _cache_targets(model: Any) -> tuple[_CacheTarget, ...]:
    output = []
    seen: set[int] = set()
    for root_label, root in _module_roots(model):
        for path, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not (callable(getattr(module, "enable_diffusion_cache", None)) and
                    callable(getattr(module, "disable_diffusion_cache", None)) and
                    callable(getattr(module, "diffusion_cache_stats", None))):
                continue
            label = root_label if not path else f"{root_label}.{path}"
            output.append(_CacheTarget(module=module, label=label))
    return tuple(output)


_CACHE_PASS_CAPABILITIES = OptimizationCapabilities(
    modes=(OptimizationMode.INFERENCE, ),
    devices=("cpu", "cuda", "mps"),
    dtypes=("float32", "float16", "bfloat16"),
    streaming_safe=False,
    distributed_safe=True,
    persistent=False,
    reversible=True,
    changes_parameter_names=False,
    changes_topology=False,
)


class DiffusionCachePass(OptimizationPass):
    """Reversibly enable approximate block caching on declared DiT modules."""

    pass_id = "voicehub.diffusion-block-cache"
    pass_version = "1"
    optimization_kind = "diffusion-cache"
    capabilities = _CACHE_PASS_CAPABILITIES

    def __init__(
        self,
        config: DiffusionCacheConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = coerce_diffusion_cache_config(config)

    def manifest_configuration(self) -> Mapping[str, Any]:
        return {
            "algorithm": "first-last-block-residual-cache",
            "fidelity": "approximate",
            **self.config.to_dict(),
        }

    def validate(self, model: Any, context: OptimizationContext) -> None:
        super().validate(model, context)
        if not _cache_targets(model):
            raise OptimizationCompatibilityError(
                f"{type(model).__name__} exposes no architecture-owned "
                "enable_diffusion_cache()/disable_diffusion_cache() block "
                "surface.")

    def apply(self, model: Any, context: OptimizationContext) -> PassResult:
        del context
        targets = _cache_targets(model)
        patches: list[_CachePatch] = []
        try:
            for target in targets:
                previous = target.module.diffusion_cache_config
                target.module.enable_diffusion_cache(self.config)
                patches.append(_CachePatch(target=target, previous=previous))
        except BaseException:
            for patch in reversed(patches):
                patch.target.module.disable_diffusion_cache()
                if patch.previous is not None:
                    patch.target.module.enable_diffusion_cache(patch.previous)
            raise
        return PassResult(
            model=model,
            state={"patches": tuple(patches)},
            metadata={
                "targets": [target.label for target in targets],
                "fidelity": "approximate",
            },
        )

    def restore(
        self,
        model: Any,
        state: Mapping[str, Any],
        context: OptimizationContext,
    ) -> Any:
        del context
        patches = state.get("patches")
        if not isinstance(patches, tuple):
            raise DiffusionCacheError("Diffusion-cache restoration state is missing its target patches.")
        errors = []
        for patch in reversed(patches):
            try:
                patch.target.module.disable_diffusion_cache()
                if patch.previous is not None:
                    patch.target.module.enable_diffusion_cache(patch.previous)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise DiffusionCacheError(
                f"Could not restore {len(errors)} diffusion-cache target(s).") from errors[0]
        return model

    def runtime_manifest_status(
        self,
        result: PassResult,
    ) -> Mapping[str, Any]:
        patches = result.state.get("patches", ())
        return {patch.target.label: patch.target.module.diffusion_cache_stats() for patch in patches}


__all__ = [
    "DiffusionBlockResidualCache",
    "DiffusionCacheCompatibilityError",
    "DiffusionCacheConfig",
    "DiffusionCacheError",
    "DiffusionCacheMixin",
    "DiffusionCachePass",
    "DiffusionCachePolicy",
    "DiffusionCachePredictor",
    "coerce_diffusion_cache_config",
]
