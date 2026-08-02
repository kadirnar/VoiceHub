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
import statistics
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
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


class DiffusionCacheMethod(str, Enum):
    """Architecture-owned block-cache layouts."""

    DBCACHE = "dbcache"
    FIRST_BLOCK = "first_block"

    @classmethod
    def coerce(
        cls,
        value: DiffusionCacheMethod | str,
    ) -> DiffusionCacheMethod:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`method` must be a string or DiffusionCacheMethod.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "cache_dit": cls.DBCACHE.value,
            "fbcache": cls.FIRST_BLOCK.value,
            "first_block_cache": cls.FIRST_BLOCK.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion cache method {value!r}; expected one of: "
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


class DiffusionCacheStepPolicy(str, Enum):
    """How an explicit step-computation mask controls cache decisions."""

    DYNAMIC = "dynamic"
    STATIC = "static"

    @classmethod
    def coerce(
        cls,
        value: DiffusionCacheStepPolicy | str,
    ) -> DiffusionCacheStepPolicy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`compute_step_policy` must be a string or "
                            "DiffusionCacheStepPolicy.")
        normalized = value.strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion-cache step policy {value!r}; expected "
                f"one of: {choices}.") from error


class DiffusionCacheRefreshPolicy(str, Enum):
    """Whether a configured cache-refresh hint runs once or periodically."""

    ONCE = "once"
    REPEAT = "repeat"

    @classmethod
    def coerce(
        cls,
        value: DiffusionCacheRefreshPolicy | str,
    ) -> DiffusionCacheRefreshPolicy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                "`force_refresh_step_policy` must be a string or "
                "DiffusionCacheRefreshPolicy.")
        normalized = value.strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion-cache refresh policy {value!r}; expected "
                f"one of: {choices}.") from error


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer.")
    return value


def _bounded_step_limit(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < -1:
        raise ValueError(f"`{name}` must be -1 or a non-negative integer.")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


_CACHE_CONFIG_ALIASES = {
    "Fn_compute_blocks": "front_blocks",
    "Bn_compute_blocks": "back_blocks",
    "max_warmup_steps": "warmup_steps",
    "max_continuous_cached_steps": "max_consecutive_cached_steps",
    "max_accumulated_residual_diff_threshold": "max_accumulated_relative_error",
    "steps_computation_mask": "compute_step_mask",
    "steps_computation_policy": "compute_step_policy",
    "downsample_factor": "probe_downsample_factor",
    "taylorseer_order": "taylor_order",
}


def _normalize_cache_config_values(values: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(values)
    for alias, canonical in _CACHE_CONFIG_ALIASES.items():
        if alias not in output:
            continue
        if canonical in output:
            raise ValueError(
                f"Diffusion cache configuration cannot set both {alias!r} "
                f"and {canonical!r}.")
        output[canonical] = output.pop(alias)
    if "compute_step_mask" in output:
        raw_mask = output["compute_step_mask"]
        if not isinstance(raw_mask, (str, bytes)):
            try:
                mask_items = tuple(raw_mask)
            except TypeError:
                pass
            else:
                if all(isinstance(item, int) and not isinstance(item, bool) and item in {0, 1}
                       for item in mask_items):
                    output["compute_step_mask"] = tuple(bool(item) for item in mask_items)
    return output


@dataclass(frozen=True, slots=True)
class DiffusionCacheConfig:
    """Serializable Cache-DiT-style settings for a VoiceHub DiT block list.

    ``front_blocks`` are always evaluated and provide the change probe.
    ``back_blocks`` are always evaluated after either computing or
    predicting the middle residual. The ``taylor`` predictor supports
    orders one through three and uses only previously *computed*
    residuals; it never chains predictions.

    The feature is approximate by construction.  Merely constructing
    this object does not enable it; callers must explicitly select
    :class:`DiffusionCachePolicy.AUTO` or ``REQUIRED`` in the universal
    TTS optimizer, or apply :class:`DiffusionCachePass` directly.
    """

    method: DiffusionCacheMethod | str = DiffusionCacheMethod.DBCACHE
    front_blocks: int = 1
    back_blocks: int = 0
    residual_diff_threshold: float = 0.08
    warmup_steps: int = 2
    warmup_interval: int = 1
    max_cached_steps: int = -1
    max_consecutive_cached_steps: int = 3
    max_accumulated_relative_error: float | None = None
    predictor: DiffusionCachePredictor | str = DiffusionCachePredictor.REUSE
    taylor_order: int = 1
    compute_step_mask: tuple[bool, ...] = ()
    compute_step_policy: DiffusionCacheStepPolicy | str = DiffusionCacheStepPolicy.DYNAMIC
    num_inference_steps: int | None = None
    force_refresh_step_hint: int | None = None
    force_refresh_step_policy: DiffusionCacheRefreshPolicy | str = DiffusionCacheRefreshPolicy.ONCE
    probe_downsample_factor: int = 1
    metrics_history_size: int = 256
    synchronize_distributed: bool = True
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        method = DiffusionCacheMethod.coerce(self.method)
        front = _non_negative_integer(self.front_blocks, name="front_blocks")
        if front == 0:
            raise ValueError("`front_blocks` must be at least one.")
        back = _non_negative_integer(self.back_blocks, name="back_blocks")
        if (method is DiffusionCacheMethod.FIRST_BLOCK and (front != 1 or back != 0)):
            raise ValueError("First-block cache requires `front_blocks=1` and "
                             "`back_blocks=0`.")
        warmup = _non_negative_integer(self.warmup_steps, name="warmup_steps")
        warmup_interval = _positive_integer(
            self.warmup_interval,
            name="warmup_interval",
        )
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
        taylor_order = _positive_integer(
            self.taylor_order,
            name="taylor_order",
        )
        if taylor_order > 3:
            raise ValueError("`taylor_order` must be between 1 and 3.")
        if isinstance(self.compute_step_mask, (str, bytes)):
            raise TypeError("`compute_step_mask` must be an iterable of booleans.")
        try:
            step_mask = tuple(self.compute_step_mask)
        except TypeError as error:
            raise TypeError("`compute_step_mask` must be an iterable of booleans.") from error
        if any(not isinstance(item, bool) for item in step_mask):
            raise TypeError("`compute_step_mask` may contain only booleans.")
        step_policy = DiffusionCacheStepPolicy.coerce(self.compute_step_policy)
        num_inference_steps = self.num_inference_steps
        if num_inference_steps is not None:
            num_inference_steps = _positive_integer(
                num_inference_steps,
                name="num_inference_steps",
            )
            if step_mask and len(step_mask) != num_inference_steps:
                raise ValueError(
                    "`compute_step_mask` length must match "
                    "`num_inference_steps` when both are set.")
        refresh_hint = self.force_refresh_step_hint
        if refresh_hint is not None:
            refresh_hint = _positive_integer(
                refresh_hint,
                name="force_refresh_step_hint",
            )
        refresh_policy = DiffusionCacheRefreshPolicy.coerce(self.force_refresh_step_policy)
        downsample = _positive_integer(
            self.probe_downsample_factor,
            name="probe_downsample_factor",
        )
        metrics_history = _non_negative_integer(
            self.metrics_history_size,
            name="metrics_history_size",
        )
        if not isinstance(self.synchronize_distributed, bool):
            raise TypeError("`synchronize_distributed` must be a boolean.")
        if (isinstance(self.epsilon, bool) or not isinstance(self.epsilon, (int, float))):
            raise TypeError("`epsilon` must be a real number.")
        epsilon = float(self.epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("`epsilon` must be finite and greater than zero.")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "front_blocks", front)
        object.__setattr__(self, "back_blocks", back)
        object.__setattr__(self, "residual_diff_threshold", threshold)
        object.__setattr__(self, "warmup_steps", warmup)
        object.__setattr__(self, "warmup_interval", warmup_interval)
        object.__setattr__(self, "max_cached_steps", maximum)
        object.__setattr__(self, "max_consecutive_cached_steps", consecutive)
        object.__setattr__(
            self,
            "max_accumulated_relative_error",
            accumulated,
        )
        object.__setattr__(self, "predictor", predictor)
        object.__setattr__(self, "taylor_order", taylor_order)
        object.__setattr__(self, "compute_step_mask", step_mask)
        object.__setattr__(self, "compute_step_policy", step_policy)
        object.__setattr__(self, "num_inference_steps", num_inference_steps)
        object.__setattr__(self, "force_refresh_step_hint", refresh_hint)
        object.__setattr__(self, "force_refresh_step_policy", refresh_policy)
        object.__setattr__(self, "probe_downsample_factor", downsample)
        object.__setattr__(self, "metrics_history_size", metrics_history)
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
        output = _normalize_cache_config_values(values)
        output.update(_normalize_cache_config_values(overrides))
        return cls(**output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "front_blocks": self.front_blocks,
            "back_blocks": self.back_blocks,
            "residual_diff_threshold": self.residual_diff_threshold,
            "warmup_steps": self.warmup_steps,
            "warmup_interval": self.warmup_interval,
            "max_cached_steps": self.max_cached_steps,
            "max_consecutive_cached_steps": self.max_consecutive_cached_steps,
            "max_accumulated_relative_error": self.max_accumulated_relative_error,
            "predictor": self.predictor.value,
            "taylor_order": self.taylor_order,
            "compute_step_mask": list(self.compute_step_mask),
            "compute_step_policy": self.compute_step_policy.value,
            "num_inference_steps": self.num_inference_steps,
            "force_refresh_step_hint": self.force_refresh_step_hint,
            "force_refresh_step_policy": self.force_refresh_step_policy.value,
            "probe_downsample_factor": self.probe_downsample_factor,
            "metrics_history_size": self.metrics_history_size,
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
    segment_step: int = 0
    refreshes: int = 0
    forced_refreshes: int = 0
    cached_steps: int = 0
    consecutive_cached_steps: int = 0
    accumulated_relative_error: float = 0.0
    probe: Tensor | None = None
    middle_residual: Tensor | None = None
    block_signature: tuple[int, ...] | None = None
    residual_history: list[tuple[int, Tensor]] = field(default_factory=list)


@dataclass(slots=True)
class _MutableCacheStats:
    calls: int = 0
    computed_steps: int = 0
    cached_steps: int = 0
    bypassed_steps: int = 0
    cold_misses: int = 0
    warmup_misses: int = 0
    mask_misses: int = 0
    threshold_misses: int = 0
    cached_step_limit_misses: int = 0
    consecutive_step_limit_misses: int = 0
    accumulated_error_misses: int = 0
    invalidations: int = 0
    forced_refreshes: int = 0
    inference_refreshes: int = 0
    sessions: int = 0
    dynamic_hits: int = 0
    static_hits: int = 0
    reuse_predictions: int = 0
    taylor_predictions: int = 0
    taylor_fallbacks: int = 0
    maximum_taylor_order_used: int = 0
    total_block_evaluations: int = 0
    executed_block_evaluations: int = 0
    skipped_block_evaluations: int = 0
    maximum_consecutive_cached_steps_observed: int = 0
    peak_cache_entries: int = 0
    peak_cache_bytes: int = 0
    last_relative_difference: float | None = None
    maximum_relative_difference: float | None = None
    residual_differences: list[float] = field(default_factory=list)
    accepted_residual_differences: list[float] = field(default_factory=list)
    rejected_residual_differences: list[float] = field(default_factory=list)
    cached_step_indices: list[int] = field(default_factory=list)
    computed_step_indices: list[int] = field(default_factory=list)
    lanes: dict[str, _MutableCacheStats] = field(default_factory=dict)

    @staticmethod
    def _percentile(values: Sequence[float], quantile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def snapshot(
        self,
        *,
        active_cache_entries: int = 0,
        active_cache_bytes: int = 0,
        details: bool = False,
        include_lanes: bool = True,
    ) -> dict[str, Any]:
        evaluated = self.computed_steps + self.cached_steps
        hit_rate = (self.cached_steps / evaluated if evaluated else 0.0)
        total_blocks = self.total_block_evaluations
        executed_blocks = self.executed_block_evaluations
        output: dict[str, Any] = {
            "calls":
            self.calls,
            "evaluated_steps":
            evaluated,
            "computed_steps":
            self.computed_steps,
            "cached_steps":
            self.cached_steps,
            "bypassed_steps":
            self.bypassed_steps,
            "cache_misses":
            self.computed_steps,
            "cold_misses":
            self.cold_misses,
            "warmup_misses":
            self.warmup_misses,
            "mask_misses":
            self.mask_misses,
            "threshold_misses":
            self.threshold_misses,
            "cached_step_limit_misses":
            self.cached_step_limit_misses,
            "consecutive_step_limit_misses":
            self.consecutive_step_limit_misses,
            "accumulated_error_misses":
            self.accumulated_error_misses,
            "limit_misses": (
                self.cached_step_limit_misses + self.consecutive_step_limit_misses +
                self.accumulated_error_misses),
            "invalidations":
            self.invalidations,
            "forced_refreshes":
            self.forced_refreshes,
            "inference_refreshes":
            self.inference_refreshes,
            "sessions":
            self.sessions,
            "hit_rate":
            hit_rate,
            "miss_rate": (self.computed_steps / evaluated if evaluated else 0.0),
            "dynamic_hits":
            self.dynamic_hits,
            "static_hits":
            self.static_hits,
            "reuse_predictions":
            self.reuse_predictions,
            "taylor_predictions":
            self.taylor_predictions,
            "taylor_fallbacks":
            self.taylor_fallbacks,
            "maximum_taylor_order_used":
            self.maximum_taylor_order_used,
            "total_block_evaluations":
            total_blocks,
            "executed_block_evaluations":
            executed_blocks,
            "skipped_block_evaluations":
            self.skipped_block_evaluations,
            "block_compute_reduction":
            (self.skipped_block_evaluations / total_blocks if total_blocks else 0.0),
            "estimated_block_speedup": (total_blocks / executed_blocks if executed_blocks else 1.0),
            "maximum_consecutive_cached_steps":
            self.maximum_consecutive_cached_steps_observed,
            "active_cache_entries":
            active_cache_entries,
            "peak_cache_entries":
            self.peak_cache_entries,
            "active_cache_bytes":
            active_cache_bytes,
            "peak_cache_bytes":
            self.peak_cache_bytes,
            "last_relative_difference":
            self.last_relative_difference,
            "mean_relative_difference":
            (statistics.fmean(self.residual_differences) if self.residual_differences else None),
            "minimum_relative_difference":
            self._percentile(
                self.residual_differences,
                0.0,
            ),
            "p25_relative_difference":
            self._percentile(
                self.residual_differences,
                0.25,
            ),
            "p50_relative_difference":
            self._percentile(
                self.residual_differences,
                0.50,
            ),
            "p75_relative_difference":
            self._percentile(
                self.residual_differences,
                0.75,
            ),
            "p95_relative_difference":
            self._percentile(
                self.residual_differences,
                0.95,
            ),
            "maximum_relative_difference":
            self.maximum_relative_difference,
            "mean_accepted_relative_difference": (
                statistics.fmean(self.accepted_residual_differences)
                if self.accepted_residual_differences else None),
            "mean_rejected_relative_difference": (
                statistics.fmean(self.rejected_residual_differences)
                if self.rejected_residual_differences else None),
        }
        if details:
            output.update({
                "residual_differences": list(self.residual_differences),
                "accepted_residual_differences": list(self.accepted_residual_differences),
                "rejected_residual_differences": list(self.rejected_residual_differences),
                "cached_step_indices": list(self.cached_step_indices),
                "computed_step_indices": list(self.computed_step_indices),
            })
        if include_lanes:
            output["lanes"] = {
                name: lane.snapshot(
                    details=details,
                    include_lanes=False,
                )
                for name, lane in sorted(self.lanes.items())
            }
        return output


def _compatible_tensor(left: Tensor | None, right: Tensor) -> bool:
    return (
        left is not None and left.shape == right.shape and left.dtype == right.dtype and
        left.device == right.device)


def _tensor_bytes(value: Tensor) -> int:
    return value.numel() * value.element_size()


def _state_cache_bytes(states: Mapping[tuple[str, str], _CacheLaneState]) -> int:
    total = 0
    seen: set[int] = set()
    for state in states.values():
        tensors = [state.probe, state.middle_residual]
        tensors.extend(residual for _step, residual in state.residual_history)
        for value in tensors:
            if value is None or id(value) in seen:
                continue
            seen.add(id(value))
            total += _tensor_bytes(value)
    return total


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
    downsample_factor: int,
) -> Tensor:
    previous_values = previous
    current_values = current
    expanded = (None if valid_mask is None else _expanded_valid_mask(valid_mask, current))
    per_batch = current.ndim >= 3
    if downsample_factor > 1:
        if current.ndim >= 2:
            previous_values = previous.reshape(current.shape[0], -1)[:, ::downsample_factor]
            current_values = current.reshape(current.shape[0], -1)[:, ::downsample_factor]
            if expanded is not None:
                expanded = expanded.reshape(current.shape[0], -1)[:, ::downsample_factor]
        else:
            previous_values = previous[::downsample_factor]
            current_values = current[::downsample_factor]
            if expanded is not None:
                expanded = expanded[::downsample_factor]
    previous_float = previous_values.float()
    difference = (previous_float - current_values.float()).abs()
    denominator = previous_float.abs()
    if valid_mask is not None:
        if expanded is None:
            raise RuntimeError("Diffusion-cache valid-mask expansion failed.")
        difference = difference.masked_fill(~expanded, 0)
        denominator = denominator.masked_fill(~expanded, 0)
        if per_batch:
            reduce_dims = tuple(range(1, difference.ndim))
            counts = expanded.sum(dim=reduce_dims).clamp_min(1)
            numerator = difference.sum(dim=reduce_dims) / counts
            baseline = denominator.sum(dim=reduce_dims) / counts
            return (numerator / baseline.clamp_min(epsilon)).amax()
        count_value = expanded.sum().clamp_min(1)
        return (difference.sum() / count_value / (denominator.sum() / count_value).clamp_min(epsilon))
    if per_batch:
        reduce_dims = tuple(range(1, difference.ndim))
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
                self._increment_metric(key[1], "invalidations")

    def reset_all(self) -> None:
        with self._lock:
            lanes = [key[1] for key in self._states]
            self._states.clear()
            for lane_name in lanes:
                self._increment_metric(lane_name, "invalidations")

    def stats(self, *, details: bool = False) -> dict[str, Any]:
        with self._lock:
            active_entries = len(self._states)
            active_bytes = _state_cache_bytes(self._states)
            output = self._stats.snapshot(
                active_cache_entries=active_entries,
                active_cache_bytes=active_bytes,
                details=details,
            )
            lane_resources: dict[str, tuple[int, int]] = {}
            for (_session, lane), state in self._states.items():
                entries, size = lane_resources.get(lane, (0, 0))
                lane_resources[lane] = (
                    entries + 1,
                    size + _state_cache_bytes({("", lane): state}),
                )
            for lane, lane_output in output["lanes"].items():
                entries, size = lane_resources.get(lane, (0, 0))
                lane_output["active_cache_entries"] = entries
                lane_output["active_cache_bytes"] = size
            return output

    def reset_stats(self) -> None:
        """Clear telemetry without changing active cache tensors."""
        with self._lock:
            active_entries = len(self._states)
            active_bytes = _state_cache_bytes(self._states)
            self._stats = _MutableCacheStats(
                peak_cache_entries=active_entries,
                peak_cache_bytes=active_bytes,
            )

    def _metric_targets(self, lane: str) -> tuple[_MutableCacheStats, _MutableCacheStats]:
        lane_stats = self._stats.lanes.setdefault(lane, _MutableCacheStats())
        return self._stats, lane_stats

    def _increment_metric(self, lane: str, name: str, amount: int = 1) -> None:
        for target in self._metric_targets(lane):
            setattr(target, name, getattr(target, name) + amount)

    def _append_metric(self, lane: str, name: str, value: Any) -> None:
        maximum = self.config.metrics_history_size
        if maximum == 0:
            return
        for target in self._metric_targets(lane):
            values = getattr(target, name)
            values.append(value)
            if len(values) > maximum:
                del values[:len(values) - maximum]

    def _update_cache_resource_peaks(self, lane: str) -> None:
        entries = len(self._states)
        cache_bytes = _state_cache_bytes(self._states)
        self._stats.peak_cache_entries = max(
            self._stats.peak_cache_entries,
            entries,
        )
        self._stats.peak_cache_bytes = max(
            self._stats.peak_cache_bytes,
            cache_bytes,
        )
        lane_entries = {key: state for key, state in self._states.items() if key[1] == lane}
        lane_stats = self._stats.lanes.setdefault(lane, _MutableCacheStats())
        lane_stats.peak_cache_entries = max(
            lane_stats.peak_cache_entries,
            len(lane_entries),
        )
        lane_stats.peak_cache_bytes = max(
            lane_stats.peak_cache_bytes,
            _state_cache_bytes(lane_entries),
        )

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
    ) -> tuple[Tensor, int]:
        residual = state.middle_residual
        if residual is None:
            raise RuntimeError("A cache hit requires a middle residual.")
        if self.config.predictor is not DiffusionCachePredictor.TAYLOR:
            return residual, 0
        order = min(
            self.config.taylor_order,
            len(state.residual_history) - 1,
        )
        if order < 1:
            return residual, 0
        points = state.residual_history[-(order + 1):]
        predicted = torch.zeros_like(residual)
        for index, (point_step, point_value) in enumerate(points):
            coefficient = 1.0
            for other_index, (other_step, _other_value) in enumerate(points):
                if index == other_index:
                    continue
                denominator = point_step - other_step
                if denominator == 0:
                    return residual, 0
                coefficient *= (current_step - other_step) / denominator
            predicted = predicted + point_value * coefficient
        return predicted, order

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
        block_count = len(block_sequence)
        block_signature = tuple(id(block) for block in block_sequence)
        middle_end = block_count - self.config.back_blocks
        middle_block_count = max(
            0,
            middle_end - self.config.front_blocks,
        )
        with self._lock:
            self._increment_metric(lane_name, "calls")
            self._increment_metric(
                lane_name,
                "total_block_evaluations",
                block_count,
            )
        unsupported = (
            training or torch.is_grad_enabled() or
            block_count <= self.config.front_blocks + self.config.back_blocks or
            self.config.residual_diff_threshold <= 0)
        if unsupported:
            with self._lock:
                self._increment_metric(lane_name, "bypassed_steps")
                self._increment_metric(
                    lane_name,
                    "executed_block_evaluations",
                    block_count,
                )
            return self._full_run(
                block_sequence,
                hidden_states,
                block_call,
            )

        session_key = (self._session.get(), lane_name)
        with self._lock:
            state = self._states.setdefault(session_key, _CacheLaneState())
            current_step = state.step
            refresh_kind = None
            if (self.config.num_inference_steps is not None and
                    state.segment_step >= self.config.num_inference_steps):
                refresh_kind = "inference"
            hint = self.config.force_refresh_step_hint
            if hint is not None:
                should_force_refresh = (
                    current_step == hint and
                    self.config.force_refresh_step_policy is DiffusionCacheRefreshPolicy.ONCE and
                    state.forced_refreshes == 0)
                if self.config.force_refresh_step_policy is DiffusionCacheRefreshPolicy.REPEAT:
                    should_force_refresh = current_step > 0 and current_step % hint == 0
                if should_force_refresh:
                    refresh_kind = "forced"
            if refresh_kind is not None:
                state.segment_step = 0
                state.cached_steps = 0
                state.consecutive_cached_steps = 0
                state.accumulated_relative_error = 0.0
                state.probe = None
                state.middle_residual = None
                state.block_signature = None
                state.residual_history.clear()
                state.refreshes += 1
                if refresh_kind == "forced":
                    state.forced_refreshes += 1
                    self._increment_metric(lane_name, "forced_refreshes")
                else:
                    self._increment_metric(lane_name, "inference_refreshes")
            current_segment_step = state.segment_step
            state.step += 1
            state.segment_step += 1

        value = hidden_states
        for block in block_sequence[:self.config.front_blocks]:
            value = block_call(block, value)
        probe = value

        with self._lock:
            cached_probe = state.probe
            cached_residual = state.middle_residual
        cache_compatible = (
            state.block_signature == block_signature and _compatible_tensor(cached_probe, probe) and
            _compatible_tensor(cached_residual, probe))
        force_compute = not cache_compatible
        miss_kind = (
            "invalid" if force_compute and cached_probe is not None else "cold" if force_compute else "")
        warmup_compute = (
            current_segment_step < self.config.warmup_steps and
            current_segment_step % self.config.warmup_interval == 0)
        if not force_compute and warmup_compute:
            force_compute = True
            miss_kind = "warmup"
        mask_value = None
        if (self.config.compute_step_mask and current_segment_step < len(self.config.compute_step_mask)):
            mask_value = self.config.compute_step_mask[current_segment_step]
        if not force_compute and mask_value is True:
            force_compute = True
            miss_kind = "mask"
        if (not force_compute and self.config.max_cached_steps >= 0 and
                state.cached_steps >= self.config.max_cached_steps):
            force_compute = True
            miss_kind = "cached-step-limit"
        if (not force_compute and self.config.max_consecutive_cached_steps >= 0 and
                state.consecutive_cached_steps >= self.config.max_consecutive_cached_steps):
            force_compute = True
            miss_kind = "consecutive-step-limit"

        relative_difference: float | None = None
        static_cache = (
            mask_value is False and self.config.compute_step_policy is DiffusionCacheStepPolicy.STATIC)
        if not force_compute and not static_cache:
            score = _relative_l1(
                cached_probe,
                probe,
                valid_mask=valid_mask,
                epsilon=self.config.epsilon,
                downsample_factor=self.config.probe_downsample_factor,
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
                miss_kind = "accumulated-error"

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
                    state.block_signature = None
                    state.residual_history.clear()
                    state.consecutive_cached_steps = 0
                    self._increment_metric(lane_name, "bypassed_steps")
                    self._increment_metric(
                        lane_name,
                        "executed_block_evaluations",
                        block_count,
                    )
                value = middle
            else:
                residual = (middle - probe).detach()
                with self._lock:
                    state.middle_residual = residual
                    state.block_signature = block_signature
                    state.residual_history.append((current_step, residual))
                    del state.residual_history[:max(
                        0,
                        len(state.residual_history) - self.config.taylor_order - 1,
                    )]
                    state.probe = probe.detach()
                    state.consecutive_cached_steps = 0
                    state.accumulated_relative_error = 0.0
                    self._increment_metric(lane_name, "computed_steps")
                    self._increment_metric(
                        lane_name,
                        "executed_block_evaluations",
                        block_count,
                    )
                    self._append_metric(
                        lane_name,
                        "computed_step_indices",
                        current_step,
                    )
                    if miss_kind == "warmup":
                        self._increment_metric(lane_name, "warmup_misses")
                    elif miss_kind == "mask":
                        self._increment_metric(lane_name, "mask_misses")
                    elif miss_kind == "threshold":
                        self._increment_metric(lane_name, "threshold_misses")
                    elif miss_kind == "cached-step-limit":
                        self._increment_metric(
                            lane_name,
                            "cached_step_limit_misses",
                        )
                    elif miss_kind == "consecutive-step-limit":
                        self._increment_metric(
                            lane_name,
                            "consecutive_step_limit_misses",
                        )
                    elif miss_kind == "accumulated-error":
                        self._increment_metric(
                            lane_name,
                            "accumulated_error_misses",
                        )
                    elif miss_kind == "invalid" and current_step > 0:
                        self._increment_metric(lane_name, "invalidations")
                    elif miss_kind in {"cold", "invalid"}:
                        self._increment_metric(lane_name, "cold_misses")
                    self._update_cache_resource_peaks(lane_name)
                value = middle
        else:
            with self._lock:
                predicted, taylor_order = self._predicted_residual(
                    state,
                    current_step=current_step,
                )
                state.cached_steps += 1
                state.consecutive_cached_steps += 1
                if relative_difference is not None:
                    state.accumulated_relative_error += relative_difference
                state.probe = probe.detach()
                self._increment_metric(lane_name, "cached_steps")
                self._increment_metric(
                    lane_name,
                    "executed_block_evaluations",
                    self.config.front_blocks + self.config.back_blocks,
                )
                self._increment_metric(
                    lane_name,
                    "skipped_block_evaluations",
                    middle_block_count,
                )
                self._increment_metric(
                    lane_name,
                    "static_hits" if static_cache else "dynamic_hits",
                )
                self._append_metric(
                    lane_name,
                    "cached_step_indices",
                    current_step,
                )
                if self.config.predictor is DiffusionCachePredictor.TAYLOR:
                    if taylor_order:
                        self._increment_metric(lane_name, "taylor_predictions")
                        for target in self._metric_targets(lane_name):
                            target.maximum_taylor_order_used = max(
                                target.maximum_taylor_order_used,
                                taylor_order,
                            )
                    else:
                        self._increment_metric(lane_name, "taylor_fallbacks")
                else:
                    self._increment_metric(lane_name, "reuse_predictions")
                for target in self._metric_targets(lane_name):
                    target.maximum_consecutive_cached_steps_observed = max(
                        target.maximum_consecutive_cached_steps_observed,
                        state.consecutive_cached_steps,
                    )
                self._update_cache_resource_peaks(lane_name)
            value = probe + predicted

        if relative_difference is not None:
            with self._lock:
                self._append_metric(
                    lane_name,
                    "residual_differences",
                    relative_difference,
                )
                self._append_metric(
                    lane_name,
                    ("rejected_residual_differences" if force_compute else "accepted_residual_differences"),
                    relative_difference,
                )
                for target in self._metric_targets(lane_name):
                    target.last_relative_difference = relative_difference
                    maximum = target.maximum_relative_difference
                    target.maximum_relative_difference = (
                        relative_difference if maximum is None else max(
                            maximum,
                            relative_difference,
                        ))
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

    def reset_diffusion_cache_stats(self) -> None:
        """Clear cache telemetry without invalidating reusable tensors."""
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is not None:
            controller.reset_stats()

    def diffusion_cache_stats(self, *, details: bool = False) -> dict[str, Any]:
        controller = getattr(self, "_diffusion_cache_controller", None)
        if controller is None:
            return _MutableCacheStats().snapshot(details=details)
        return controller.stats(details=details)

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


def diffusion_cache_summary(
    model: Any,
    *,
    details: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return cache telemetry for every architecture-owned target in a
    model."""
    return {
        target.label: target.module.diffusion_cache_stats(details=details)
        for target in _cache_targets(model)
    }


def reset_diffusion_cache_metrics(model: Any) -> int:
    """Reset telemetry on every cache target and return the target count."""
    targets = _cache_targets(model)
    for target in targets:
        target.module.reset_diffusion_cache_stats()
    return len(targets)


@contextmanager
def diffusion_cache_request(model: Any) -> Iterator[Any]:
    """Isolate every cache target for one public generation request."""
    with ExitStack() as stack:
        for target in _cache_targets(model):
            stack.enter_context(target.module.diffusion_cache_session())
        yield model


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
            "algorithm": self.config.method.value,
            "fidelity": "approximate",
            **self.config.to_dict(),
        }

    def validate(self, model: Any, context: OptimizationContext) -> None:
        if not _cache_targets(model):
            return
        super().validate(model, context)

    def apply(self, model: Any, context: OptimizationContext) -> PassResult:
        del context
        targets = _cache_targets(model)
        if not targets:
            return self.not_applicable_result(
                model,
                reason=(
                    f"{type(model).__name__} exposes no architecture-owned "
                    "enable_diffusion_cache()/disable_diffusion_cache() "
                    "block surface"),
            )
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
                "outcome": "configured",
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
        if state.get("kind") == "not-applicable":
            return state.get("model", model)
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
        if result.state.get("kind") == "not-applicable":
            return {
                "outcome": "not-applicable",
                "reason": result.metadata["reason"],
            }
        patches = result.state.get("patches", ())
        return {patch.target.label: patch.target.module.diffusion_cache_stats() for patch in patches}


__all__ = [
    "DiffusionBlockResidualCache",
    "DiffusionCacheCompatibilityError",
    "DiffusionCacheConfig",
    "DiffusionCacheError",
    "DiffusionCacheMixin",
    "DiffusionCacheMethod",
    "DiffusionCachePass",
    "DiffusionCachePolicy",
    "DiffusionCachePredictor",
    "DiffusionCacheRefreshPolicy",
    "DiffusionCacheStepPolicy",
    "coerce_diffusion_cache_config",
    "diffusion_cache_request",
    "diffusion_cache_summary",
    "reset_diffusion_cache_metrics",
]
