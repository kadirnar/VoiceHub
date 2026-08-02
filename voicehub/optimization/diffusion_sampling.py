"""Sampler-level acceleration for diffusion and flow-matching TTS models.

This module is intentionally separate from codec kernel optimization.
Codec acceleration replaces tensor operations; the controller below
reduces diffusion model evaluations by rebuilding schedules, narrowing
classifier-free guidance, or predicting a denoiser/velocity output.

The approximations are opt-in, inference-only, request-scoped, and
architecture-owned.  A model must explicitly inherit
``DiffusionSamplingMixin`` and call the controller at its sampler
boundary.  This keeps solver stages, CFG lanes, and outer autoregressive
loops from accidentally sharing prediction state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
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
from voicehub.optimization.diffusion_solvers import STORK2FlowSolver, STORKFlowConfig
from voicehub.optimization.passes import (
    OptimizationCompatibilityError,
    OptimizationError,
    OptimizationPass,
    PassResult,
    canonical_json_string,
)
from voicehub.optimization.protocols import OptimizationModuleRoot


class DiffusionSamplingError(OptimizationError):
    """Base error for sampler-level diffusion acceleration."""


class DiffusionSamplingCompatibilityError(
        ValueError,
        DiffusionSamplingError,
):
    """A sampler cannot safely implement the requested optimization."""


class DiffusionSamplingPolicy(str, Enum):
    """Whether sampler-level acceleration is disabled, optional, or
    required."""

    DISABLED = "disabled"
    AUTO = "auto"
    REQUIRED = "required"

    @classmethod
    def coerce(
        cls,
        value: DiffusionSamplingPolicy | str | bool,
    ) -> DiffusionSamplingPolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.REQUIRED if value else cls.DISABLED
        if not isinstance(value, str):
            raise TypeError("`diffusion_sampling` must be a boolean, string, or "
                            "DiffusionSamplingPolicy.")
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
                f"Unknown diffusion-sampling policy {value!r}; expected one "
                f"of: {choices}.") from error


class DiffusionScheduleStrategy(str, Enum):
    """How a native schedule is rebuilt when ``target_steps`` is smaller."""

    NATIVE = "native"
    UNIFORM = "uniform"
    QUADRATIC = "quadratic"
    TRAILING = "trailing"

    @classmethod
    def coerce(
        cls,
        value: DiffusionScheduleStrategy | str,
    ) -> DiffusionScheduleStrategy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`schedule` must be a string or DiffusionScheduleStrategy.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "subsequence": cls.NATIVE.value,
            "native_subsequence": cls.NATIVE.value,
            "front_loaded": cls.QUADRATIC.value,
            "back_loaded": cls.TRAILING.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion schedule {value!r}; expected one of: "
                f"{choices}.") from error


class DiffusionGuidanceStrategy(str, Enum):
    """Classifier-free-guidance evaluation policy."""

    NATIVE = "native"
    LIMITED_INTERVAL = "limited_interval"
    ADAPTIVE = "adaptive"

    @classmethod
    def coerce(
        cls,
        value: DiffusionGuidanceStrategy | str,
    ) -> DiffusionGuidanceStrategy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`guidance` must be a string or DiffusionGuidanceStrategy.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "interval": cls.LIMITED_INTERVAL.value,
            "limited": cls.LIMITED_INTERVAL.value,
            "adaptive_guidance": cls.ADAPTIVE.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion guidance strategy {value!r}; expected "
                f"one of: {choices}.") from error


class DiffusionPredictionCacheMethod(str, Enum):
    """Whole-model prediction cache selected at the sampler boundary."""

    DISABLED = "disabled"
    FORA = "fora"
    TEACACHE = "teacache"
    SMOOTHCACHE = "smoothcache"
    TAYLOR = "taylor"

    @classmethod
    def coerce(
        cls,
        value: DiffusionPredictionCacheMethod | str | bool,
    ) -> DiffusionPredictionCacheMethod:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls.FORA if value else cls.DISABLED
        if not isinstance(value, str):
            raise TypeError(
                "`prediction_cache` must be a boolean, string, or "
                "DiffusionPredictionCacheMethod.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "none": cls.DISABLED.value,
            "off": cls.DISABLED.value,
            "fixed_interval": cls.FORA.value,
            "interval": cls.FORA.value,
            "tea_cache": cls.TEACACHE.value,
            "smooth_cache": cls.SMOOTHCACHE.value,
            "taylorseer": cls.TAYLOR.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion prediction cache {value!r}; expected one "
                f"of: {choices}.") from error


class DiffusionSolverStrategy(str, Enum):
    """Sampler integration rule for a supplied velocity prediction."""

    NATIVE = "native"
    STORK2 = "stork2"

    @classmethod
    def coerce(
        cls,
        value: DiffusionSolverStrategy | str,
    ) -> DiffusionSolverStrategy:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`solver` must be a string or DiffusionSolverStrategy.")
        normalized = value.strip().lower().replace("-", "")
        aliases = {
            "euler": cls.NATIVE.value,
            "stork": cls.STORK2.value,
            "stork_2": cls.STORK2.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion solver {value!r}; expected one of: "
                f"{choices}.") from error


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _non_negative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"`{name}` must be a non-negative integer.")
    return value


def _finite_float(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"`{name}` must be finite.")
    if minimum is not None:
        invalid = output <= minimum if strict_minimum else output < minimum
        if invalid:
            comparison = "greater than" if strict_minimum else "at least"
            raise ValueError(f"`{name}` must be {comparison} {minimum}.")
    if maximum is not None and output > maximum:
        raise ValueError(f"`{name}` must be at most {maximum}.")
    return output


@dataclass(frozen=True, slots=True)
class DiffusionSamplingConfig:
    """Serializable sampler-level acceleration configuration.

    ``target_steps`` compacts a caller's native schedule by rebuilding
    it before integration; the loop must never skip an already-created
    Euler/DPM step.  ``fora`` periodically reuses a full model output.
    ``teacache`` requires explicit, model-calibrated polynomial
    coefficients.  ``smoothcache`` requires an explicit compute mask.
    ``taylor`` extrapolates from computed outputs with a first- or
    second-order polynomial.
    """

    target_steps: int | None = None
    schedule: DiffusionScheduleStrategy | str = DiffusionScheduleStrategy.NATIVE
    solver: DiffusionSolverStrategy | str = DiffusionSolverStrategy.NATIVE
    stork_stages: int = 9
    guidance: DiffusionGuidanceStrategy | str = DiffusionGuidanceStrategy.NATIVE
    guidance_start: float = 0.0
    guidance_end: float = 1.0
    adaptive_guidance_threshold: float = 0.01
    adaptive_guidance_warmup_steps: int = 4
    adaptive_guidance_patience: int = 2
    prediction_cache: (DiffusionPredictionCacheMethod | str | bool) = DiffusionPredictionCacheMethod.DISABLED
    cache_interval: int = 2
    cache_warmup_steps: int = 2
    cache_max_consecutive_steps: int = 2
    cache_rel_l1_threshold: float = 0.08
    cache_error_budget: float = 0.20
    teacache_coefficients: tuple[float, ...] = ()
    smoothcache_compute_step_mask: tuple[bool, ...] = ()
    taylor_order: int = 1
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        target = self.target_steps
        if target is not None:
            target = _positive_integer(target, name="target_steps")
        schedule = DiffusionScheduleStrategy.coerce(self.schedule)
        solver = DiffusionSolverStrategy.coerce(self.solver)
        stork_stages = _positive_integer(
            self.stork_stages,
            name="stork_stages",
        )
        if stork_stages < 2:
            raise ValueError("`stork_stages` must be at least two.")
        guidance = DiffusionGuidanceStrategy.coerce(self.guidance)
        guidance_start = _finite_float(
            self.guidance_start,
            name="guidance_start",
            minimum=0.0,
            maximum=1.0,
        )
        guidance_end = _finite_float(
            self.guidance_end,
            name="guidance_end",
            minimum=0.0,
            maximum=1.0,
        )
        if guidance_start > guidance_end:
            raise ValueError("`guidance_start` cannot exceed `guidance_end`.")
        adaptive_threshold = _finite_float(
            self.adaptive_guidance_threshold,
            name="adaptive_guidance_threshold",
            minimum=0.0,
        )
        adaptive_warmup = _non_negative_integer(
            self.adaptive_guidance_warmup_steps,
            name="adaptive_guidance_warmup_steps",
        )
        adaptive_patience = _positive_integer(
            self.adaptive_guidance_patience,
            name="adaptive_guidance_patience",
        )
        prediction_cache = DiffusionPredictionCacheMethod.coerce(self.prediction_cache)
        cache_interval = _positive_integer(
            self.cache_interval,
            name="cache_interval",
        )
        cache_warmup = _non_negative_integer(
            self.cache_warmup_steps,
            name="cache_warmup_steps",
        )
        cache_max_consecutive = _positive_integer(
            self.cache_max_consecutive_steps,
            name="cache_max_consecutive_steps",
        )
        cache_threshold = _finite_float(
            self.cache_rel_l1_threshold,
            name="cache_rel_l1_threshold",
            minimum=0.0,
        )
        cache_budget = _finite_float(
            self.cache_error_budget,
            name="cache_error_budget",
            minimum=0.0,
            strict_minimum=True,
        )
        if isinstance(self.teacache_coefficients, (str, bytes)):
            raise TypeError("`teacache_coefficients` must be an iterable of floats.")
        coefficients = tuple(
            _finite_float(value, name="teacache_coefficients item") for value in self.teacache_coefficients)
        if prediction_cache is DiffusionPredictionCacheMethod.TEACACHE and not coefficients:
            raise ValueError(
                "`teacache_coefficients` are required for TeaCache because "
                "its rescaling polynomial is model-specific.")
        if isinstance(self.smoothcache_compute_step_mask, (str, bytes)):
            raise TypeError("`smoothcache_compute_step_mask` must be an iterable of booleans.")
        mask = tuple(self.smoothcache_compute_step_mask)
        if any(not isinstance(value, bool) for value in mask):
            raise TypeError("`smoothcache_compute_step_mask` may contain only booleans.")
        if prediction_cache is DiffusionPredictionCacheMethod.SMOOTHCACHE and not mask:
            raise ValueError(
                "`smoothcache_compute_step_mask` is required for SmoothCache "
                "because its reuse schedule is calibrated.")
        taylor_order = _positive_integer(
            self.taylor_order,
            name="taylor_order",
        )
        if taylor_order not in {1, 2}:
            raise ValueError("`taylor_order` must be 1 or 2.")
        if (solver is DiffusionSolverStrategy.STORK2 and guidance is not DiffusionGuidanceStrategy.NATIVE):
            raise ValueError(
                "STORK-2 cannot be combined with adaptive/limited guidance "
                "until derivative-history resets are declared by the sampler.")
        if (solver is DiffusionSolverStrategy.STORK2 and
                prediction_cache is not DiffusionPredictionCacheMethod.DISABLED):
            raise ValueError(
                "STORK-2 cannot be combined with whole-model prediction "
                "caching because both approximate velocity history.")
        epsilon = _finite_float(
            self.epsilon,
            name="epsilon",
            minimum=0.0,
            strict_minimum=True,
        )
        object.__setattr__(self, "target_steps", target)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "solver", solver)
        object.__setattr__(self, "stork_stages", stork_stages)
        object.__setattr__(self, "guidance", guidance)
        object.__setattr__(self, "guidance_start", guidance_start)
        object.__setattr__(self, "guidance_end", guidance_end)
        object.__setattr__(
            self,
            "adaptive_guidance_threshold",
            adaptive_threshold,
        )
        object.__setattr__(
            self,
            "adaptive_guidance_warmup_steps",
            adaptive_warmup,
        )
        object.__setattr__(
            self,
            "adaptive_guidance_patience",
            adaptive_patience,
        )
        object.__setattr__(self, "prediction_cache", prediction_cache)
        object.__setattr__(self, "cache_interval", cache_interval)
        object.__setattr__(self, "cache_warmup_steps", cache_warmup)
        object.__setattr__(
            self,
            "cache_max_consecutive_steps",
            cache_max_consecutive,
        )
        object.__setattr__(
            self,
            "cache_rel_l1_threshold",
            cache_threshold,
        )
        object.__setattr__(self, "cache_error_budget", cache_budget)
        object.__setattr__(self, "teacache_coefficients", coefficients)
        object.__setattr__(self, "smoothcache_compute_step_mask", mask)
        object.__setattr__(self, "taylor_order", taylor_order)
        object.__setattr__(self, "epsilon", epsilon)
        canonical_json_string(
            self.to_dict(),
            path="diffusion sampling configuration",
        )

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
        **overrides: Any,
    ) -> DiffusionSamplingConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Diffusion sampling configuration must be a mapping.")
        output = dict(values)
        output.update(overrides)
        return cls(**output)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_steps": self.target_steps,
            "schedule": self.schedule.value,
            "solver": self.solver.value,
            "stork_stages": self.stork_stages,
            "guidance": self.guidance.value,
            "guidance_start": self.guidance_start,
            "guidance_end": self.guidance_end,
            "adaptive_guidance_threshold": self.adaptive_guidance_threshold,
            "adaptive_guidance_warmup_steps": self.adaptive_guidance_warmup_steps,
            "adaptive_guidance_patience": self.adaptive_guidance_patience,
            "prediction_cache": self.prediction_cache.value,
            "cache_interval": self.cache_interval,
            "cache_warmup_steps": self.cache_warmup_steps,
            "cache_max_consecutive_steps": self.cache_max_consecutive_steps,
            "cache_rel_l1_threshold": self.cache_rel_l1_threshold,
            "cache_error_budget": self.cache_error_budget,
            "teacache_coefficients": list(self.teacache_coefficients),
            "smoothcache_compute_step_mask": list(self.smoothcache_compute_step_mask),
            "taylor_order": self.taylor_order,
            "epsilon": self.epsilon,
        }


def coerce_diffusion_sampling_config(
        value: DiffusionSamplingConfig | Mapping[str, Any] | None) -> DiffusionSamplingConfig:
    if value is None:
        return DiffusionSamplingConfig()
    if isinstance(value, DiffusionSamplingConfig):
        return value
    if isinstance(value, Mapping):
        return DiffusionSamplingConfig.from_dict(value)
    raise TypeError("`diffusion_sampling_config` must be a DiffusionSamplingConfig, "
                    "mapping, or None.")


@dataclass(frozen=True, slots=True)
class DiffusionStepContext:
    """Identity and solver state for one model evaluation."""

    index: int
    total_steps: int
    timestep: Tensor | float
    next_timestep: Tensor | float
    lane: str = "default"
    solver: str = "euler"
    stage: str = "main"
    outer_step: int | None = None

    def __post_init__(self) -> None:
        index = _non_negative_integer(self.index, name="index")
        total = _positive_integer(self.total_steps, name="total_steps")
        if index >= total:
            raise ValueError("`index` must be smaller than `total_steps`.")
        for name in ("lane", "solver", "stage"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty string.")
        outer = self.outer_step
        if outer is not None:
            outer = _non_negative_integer(outer, name="outer_step")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "total_steps", total)
        object.__setattr__(self, "lane", self.lane.strip())
        object.__setattr__(self, "solver", self.solver.strip())
        object.__setattr__(self, "stage", self.stage.strip())
        object.__setattr__(self, "outer_step", outer)

    @property
    def progress(self) -> float:
        if self.total_steps == 1:
            return 0.0
        return self.index / float(self.total_steps - 1)

    @property
    def state_lane(self) -> str:
        outer = "none" if self.outer_step is None else str(self.outer_step)
        return f"{outer}:{self.solver}:{self.stage}:{self.lane}"


@dataclass(slots=True)
class _PredictionLaneState:
    last_probe: Tensor | None = None
    outputs: list[tuple[int, Tensor]] = field(default_factory=list)
    accumulated_change: float = 0.0
    consecutive_cached_steps: int = 0
    guidance_small_steps: int = 0
    guidance_disabled: bool = False


@dataclass(slots=True)
class _MutableSamplingStats:
    schedules: int = 0
    native_steps: int = 0
    prepared_steps: int = 0
    model_calls: int = 0
    predicted_calls: int = 0
    guidance_calls: int = 0
    guidance_skips: int = 0
    resets: int = 0
    sessions: int = 0
    solver_steps: int = 0
    solver_startup_steps: int = 0
    solver_stabilized_steps: int = 0
    last_change: float | None = None

    def snapshot(self) -> dict[str, int | float | None]:
        evaluated = self.model_calls + self.predicted_calls
        return {
            "schedules": self.schedules,
            "native_steps": self.native_steps,
            "prepared_steps": self.prepared_steps,
            "model_calls": self.model_calls,
            "predicted_calls": self.predicted_calls,
            "prediction_hit_rate": (self.predicted_calls / evaluated if evaluated else 0.0),
            "guidance_calls": self.guidance_calls,
            "guidance_skips": self.guidance_skips,
            "resets": self.resets,
            "sessions": self.sessions,
            "solver_steps": self.solver_steps,
            "solver_startup_steps": self.solver_startup_steps,
            "solver_stabilized_steps": self.solver_stabilized_steps,
            "last_change": self.last_change,
        }


def _relative_l1(previous: Tensor, current: Tensor, *, epsilon: float) -> float:
    if (previous.shape != current.shape or previous.dtype != current.dtype or
            previous.device != current.device):
        return math.inf
    difference = (current.float() - previous.float()).abs().mean()
    baseline = previous.float().abs().mean().clamp_min(epsilon)
    return float((difference / baseline).item())


def _polynomial(value: float, coefficients: tuple[float, ...]) -> float:
    output = 0.0
    for coefficient in reversed(coefficients):
        output = output * value + coefficient
    return output


def _lagrange_prediction(
    samples: list[tuple[int, Tensor]],
    *,
    step: int,
    order: int,
) -> Tensor:
    points = samples[-(order + 1):]
    if len(points) == 1:
        return points[0][1]
    output = torch.zeros_like(points[-1][1])
    for position, (sample_step, sample) in enumerate(points):
        weight = 1.0
        for other_position, (other_step, _other_sample) in enumerate(points):
            if position == other_position:
                continue
            denominator = sample_step - other_step
            if denominator == 0:
                return points[-1][1]
            weight *= (step - other_step) / float(denominator)
        output = output + sample * weight
    return output


class DiffusionSamplingController:
    """Request-scoped schedule, guidance, and prediction-cache controller."""

    def __init__(self, config: DiffusionSamplingConfig):
        self.config = coerce_diffusion_sampling_config(config)
        self._states: dict[tuple[str, str], _PredictionLaneState] = {}
        self._solvers: dict[tuple[str, str], STORK2FlowSolver] = {}
        self._session_counter = count(1)
        self._session: ContextVar[str] = ContextVar(
            f"voicehub_diffusion_sampling_session_{id(self)}",
            default="default",
        )
        self._lock = RLock()
        self._stats = _MutableSamplingStats()

    @contextmanager
    def session(self) -> Iterator[DiffusionSamplingController]:
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
                solver_keys = [key for key in self._solvers if key[0] == session_id]
                for key in solver_keys:
                    del self._solvers[key]
            self._session.reset(token)

    def reset(self, *, lane: str | None = None) -> None:
        session_id = self._session.get()
        with self._lock:
            keys = [
                key for key in self._states
                if key[0] == session_id and (lane is None or key[1].endswith(f":{lane}"))
            ]
            for key in keys:
                del self._states[key]
            solver_keys = [
                key for key in self._solvers
                if key[0] == session_id and (lane is None or key[1].endswith(f":{lane}"))
            ]
            for key in solver_keys:
                del self._solvers[key]
            self._stats.resets += 1

    def reset_all(self) -> None:
        with self._lock:
            self._states.clear()
            self._solvers.clear()
            self._stats.resets += 1

    def stats(self) -> dict[str, int | float | None]:
        with self._lock:
            return self._stats.snapshot()

    def _state(self, context: DiffusionStepContext) -> _PredictionLaneState:
        key = (self._session.get(), context.state_lane)
        with self._lock:
            return self._states.setdefault(key, _PredictionLaneState())

    @staticmethod
    def _validate_schedule(values: Tensor) -> int:
        if not isinstance(values, Tensor):
            raise TypeError("Diffusion schedule must be a torch.Tensor.")
        if values.ndim != 1 or values.numel() < 2:
            raise ValueError("Diffusion schedule must be one-dimensional with at least two values.")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("Diffusion schedule must contain only finite values.")
        differences = values[1:] - values[:-1]
        increasing = bool(torch.all(differences > 0).item())
        decreasing = bool(torch.all(differences < 0).item())
        if not increasing and not decreasing:
            raise ValueError("Diffusion schedule must be strictly monotonic.")
        return values.numel() - 1

    def prepare_schedule(self, values: Tensor) -> Tensor:
        """Rebuild a monotonic schedule before a solver starts."""
        native_steps = self._validate_schedule(values)
        target = self.config.target_steps
        if target is None or target >= native_steps:
            prepared = values
        elif self.config.schedule is DiffusionScheduleStrategy.NATIVE:
            positions = torch.linspace(
                0,
                native_steps,
                target + 1,
                device=values.device,
                dtype=torch.float64,
            ).round().to(dtype=torch.long)
            positions[0] = 0
            positions[-1] = native_steps
            prepared = values.index_select(0, positions)
        else:
            if not values.is_floating_point():
                raise DiffusionSamplingCompatibilityError(
                    "Uniform, quadratic, and trailing diffusion schedules "
                    "require a floating-point native schedule.")
            unit = torch.linspace(
                0,
                1,
                target + 1,
                device=values.device,
                dtype=values.dtype,
            )
            if self.config.schedule is DiffusionScheduleStrategy.QUADRATIC:
                unit = unit.square()
            elif self.config.schedule is DiffusionScheduleStrategy.TRAILING:
                unit = 1 - (1 - unit).square()
            prepared = values[0] + (values[-1] - values[0]) * unit
        prepared_steps = self._validate_schedule(prepared)
        with self._lock:
            self._stats.schedules += 1
            self._stats.native_steps += native_steps
            self._stats.prepared_steps += prepared_steps
        return prepared

    def should_use_guidance(
        self,
        context: DiffusionStepContext,
        *,
        native: bool = True,
    ) -> bool:
        """Only narrow a sampler's native CFG decision; never enable CFG."""
        if not native:
            return False
        strategy = self.config.guidance
        if strategy is DiffusionGuidanceStrategy.NATIVE:
            with self._lock:
                self._stats.guidance_calls += 1
            return True
        in_interval = (self.config.guidance_start <= context.progress <= self.config.guidance_end)
        state = self._state(context)
        use_guidance = in_interval and not (
            strategy is DiffusionGuidanceStrategy.ADAPTIVE and state.guidance_disabled)
        with self._lock:
            if use_guidance:
                self._stats.guidance_calls += 1
            else:
                self._stats.guidance_skips += 1
        return use_guidance

    def observe_guidance(
        self,
        context: DiffusionStepContext,
        conditional: Tensor,
        unconditional: Tensor,
    ) -> float | None:
        """Observe conditional convergence for adaptive guidance."""
        if self.config.guidance is not DiffusionGuidanceStrategy.ADAPTIVE:
            return None
        state = self._state(context)
        relative = _relative_l1(
            conditional,
            unconditional,
            epsilon=self.config.epsilon,
        )
        if context.index < self.config.adaptive_guidance_warmup_steps:
            state.guidance_small_steps = 0
            return relative
        if relative <= self.config.adaptive_guidance_threshold:
            state.guidance_small_steps += 1
            if state.guidance_small_steps >= self.config.adaptive_guidance_patience:
                state.guidance_disabled = True
        else:
            state.guidance_small_steps = 0
        return relative

    def _compute_required(
        self,
        context: DiffusionStepContext,
        state: _PredictionLaneState,
        probe: Tensor,
    ) -> bool:
        method = self.config.prediction_cache
        if method is DiffusionPredictionCacheMethod.DISABLED:
            return True
        previous_probe = state.last_probe
        if (previous_probe is not None and
            (previous_probe.shape != probe.shape or previous_probe.dtype != probe.dtype or
             previous_probe.device != probe.device)):
            return True
        if method is DiffusionPredictionCacheMethod.SMOOTHCACHE:
            mask = self.config.smoothcache_compute_step_mask
            if len(mask) != context.total_steps:
                raise DiffusionSamplingCompatibilityError(
                    "SmoothCache compute mask length must equal the prepared "
                    f"step count ({len(mask)} != {context.total_steps}).")
        if (context.index < self.config.cache_warmup_steps or not state.outputs or
                state.consecutive_cached_steps >= self.config.cache_max_consecutive_steps):
            return True
        if method in {
                DiffusionPredictionCacheMethod.FORA,
                DiffusionPredictionCacheMethod.TAYLOR,
        }:
            return context.index % self.config.cache_interval == 0
        if method is DiffusionPredictionCacheMethod.SMOOTHCACHE:
            return mask[context.index]
        if method is DiffusionPredictionCacheMethod.TEACACHE:
            previous = state.last_probe
            if previous is None:
                return True
            relative = _relative_l1(
                previous,
                probe,
                epsilon=self.config.epsilon,
            )
            if not math.isfinite(relative):
                return True
            rescaled = abs(_polynomial(
                relative,
                self.config.teacache_coefficients,
            ))
            if not math.isfinite(rescaled):
                return True
            state.accumulated_change += rescaled
            with self._lock:
                self._stats.last_change = rescaled
            return (
                rescaled >= self.config.cache_rel_l1_threshold or
                state.accumulated_change >= self.config.cache_error_budget)
        raise AssertionError(f"Unhandled diffusion cache method: {method}")

    def evaluate(
        self,
        context: DiffusionStepContext,
        probe: Tensor,
        compute: Callable[[], Tensor],
    ) -> Tensor:
        """Compute or predict one raw denoiser/velocity output."""
        if not isinstance(context, DiffusionStepContext):
            raise TypeError("`context` must be a DiffusionStepContext.")
        if not isinstance(probe, Tensor):
            raise TypeError("`probe` must be a torch.Tensor.")
        if not callable(compute):
            raise TypeError("`compute` must be callable.")
        state = self._state(context)
        required = self._compute_required(context, state, probe)
        if required:
            output = compute()
            if not isinstance(output, Tensor):
                raise TypeError("Diffusion model evaluation must return a torch.Tensor.")
            detached = output.detach()
            state.last_probe = probe.detach()
            state.outputs.append((context.index, detached))
            history = 3 if self.config.taylor_order == 2 else 2
            if len(state.outputs) > history:
                del state.outputs[:-history]
            state.accumulated_change = 0.0
            state.consecutive_cached_steps = 0
            with self._lock:
                self._stats.model_calls += 1
            return output
        state.consecutive_cached_steps += 1
        method = self.config.prediction_cache
        if method is DiffusionPredictionCacheMethod.TAYLOR:
            output = _lagrange_prediction(
                state.outputs,
                step=context.index,
                order=self.config.taylor_order,
            )
        else:
            output = state.outputs[-1][1]
        # TeaCache calibrates accumulated drift from adjacent modulated-input
        # probes, including steps whose denoiser output was reused.  Retaining
        # only the last fully computed probe over-counts change across a reuse
        # run and makes model-specific thresholds fire too early.
        state.last_probe = probe.detach()
        with self._lock:
            self._stats.predicted_calls += 1
        return output

    def advance(
        self,
        context: DiffusionStepContext,
        state: Tensor,
        velocity: Tensor,
        *,
        discontinuity: bool = False,
    ) -> Tensor:
        """Apply the configured native Euler or specialized STORK-2 step."""
        if not isinstance(context, DiffusionStepContext):
            raise TypeError("`context` must be a DiffusionStepContext.")
        if self.config.solver is DiffusionSolverStrategy.NATIVE:
            step_size = (
                context.next_timestep - context.timestep if isinstance(context.next_timestep, Tensor) else
                float(context.next_timestep) - float(context.timestep))
            return state + step_size * velocity
        key = (self._session.get(), context.state_lane)
        with self._lock:
            solver = self._solvers.get(key)
            if solver is None:
                solver = STORK2FlowSolver(STORKFlowConfig(stages=self.config.stork_stages), )
                self._solvers[key] = solver
        before = solver.stats()
        output = solver.advance(
            state,
            velocity,
            timestep=context.timestep,
            next_timestep=context.next_timestep,
            discontinuity=discontinuity,
        )
        after = solver.stats()
        with self._lock:
            self._stats.solver_steps += after["steps"] - before["steps"]
            self._stats.solver_startup_steps += (after["startup_steps"] - before["startup_steps"])
            self._stats.solver_stabilized_steps += (after["stabilized_steps"] - before["stabilized_steps"])
        return output


class DiffusionSamplingMixin:
    """Non-module mixin for architecture-owned sampler integration."""

    _diffusion_sampling_controller: DiffusionSamplingController | None
    diffusion_sampling_capabilities = frozenset({
        "schedule",
        "guidance",
        "prediction-cache",
    })

    def _initialize_diffusion_sampling(self) -> None:
        object.__setattr__(self, "_diffusion_sampling_controller", None)

    @property
    def diffusion_sampling_config(self) -> DiffusionSamplingConfig | None:
        controller = getattr(self, "_diffusion_sampling_controller", None)
        return None if controller is None else controller.config

    @property
    def diffusion_sampling_controller(self, ) -> DiffusionSamplingController | None:
        return getattr(self, "_diffusion_sampling_controller", None)

    def enable_diffusion_sampling(
        self,
        config: DiffusionSamplingConfig | Mapping[str, Any] | None = None,
    ) -> DiffusionSamplingConfig:
        resolved = coerce_diffusion_sampling_config(config)
        requested = []
        if (resolved.target_steps is not None and not {
                "schedule",
                "discrete-step-count",
        }.intersection(self.diffusion_sampling_capabilities)):
            requested.append("schedule")
        if resolved.guidance is not DiffusionGuidanceStrategy.NATIVE:
            requested.append("guidance")
        if (resolved.prediction_cache is not DiffusionPredictionCacheMethod.DISABLED):
            requested.append("prediction-cache")
        if resolved.solver is DiffusionSolverStrategy.STORK2:
            requested.append("stork2")
        missing = tuple(
            technique for technique in requested if technique not in self.diffusion_sampling_capabilities)
        if missing:
            raise DiffusionSamplingCompatibilityError(
                f"{type(self).__name__} does not declare diffusion sampling "
                f"technique(s): {', '.join(missing)}.")
        object.__setattr__(
            self,
            "_diffusion_sampling_controller",
            DiffusionSamplingController(resolved),
        )
        return resolved

    def disable_diffusion_sampling(self) -> DiffusionSamplingConfig | None:
        previous = self.diffusion_sampling_config
        controller = self.diffusion_sampling_controller
        if controller is not None:
            controller.reset_all()
        object.__setattr__(self, "_diffusion_sampling_controller", None)
        return previous

    def reset_diffusion_sampling(self, *, lane: str | None = None) -> None:
        controller = self.diffusion_sampling_controller
        if controller is not None:
            controller.reset(lane=lane)

    @contextmanager
    def diffusion_sampling_session(self) -> Iterator[DiffusionSamplingMixin]:
        controller = self.diffusion_sampling_controller
        if controller is None:
            yield self
            return
        with controller.session():
            yield self

    def diffusion_sampling_stats(self) -> dict[str, int | float | None]:
        controller = self.diffusion_sampling_controller
        if controller is None:
            return _MutableSamplingStats().snapshot()
        return controller.stats()


@dataclass(frozen=True, slots=True)
class _SamplingTarget:
    module: Any
    label: str


@dataclass(frozen=True, slots=True)
class _SamplingPatch:
    target: _SamplingTarget
    previous: DiffusionSamplingConfig | None


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
    for name in ("_model", "model"):
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


def _sampling_targets(model: Any) -> tuple[_SamplingTarget, ...]:
    output = []
    seen: set[int] = set()
    for root_label, root in _module_roots(model):
        for path, module in root.named_modules():
            if id(module) in seen:
                continue
            seen.add(id(module))
            if not (callable(getattr(module, "enable_diffusion_sampling", None)) and
                    callable(getattr(module, "disable_diffusion_sampling", None)) and
                    callable(getattr(module, "diffusion_sampling_stats", None))):
                continue
            label = root_label if not path else f"{root_label}.{path}"
            output.append(_SamplingTarget(module=module, label=label))
    return tuple(output)


_SAMPLING_PASS_CAPABILITIES = OptimizationCapabilities(
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


class DiffusionSamplingPass(OptimizationPass):
    """Reversibly enable sampler-level diffusion acceleration."""

    pass_id = "voicehub.diffusion-sampling"
    pass_version = "1"
    optimization_kind = "diffusion-sampling"
    capabilities = _SAMPLING_PASS_CAPABILITIES

    def __init__(
        self,
        config: DiffusionSamplingConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = coerce_diffusion_sampling_config(config)

    def manifest_configuration(self) -> Mapping[str, Any]:
        return {
            "fidelity": "approximate",
            **self.config.to_dict(),
        }

    def validate(self, model: Any, context: OptimizationContext) -> None:
        if not _sampling_targets(model):
            return
        super().validate(model, context)

    def apply(self, model: Any, context: OptimizationContext) -> PassResult:
        del context
        targets = _sampling_targets(model)
        if not targets:
            return self.not_applicable_result(
                model,
                reason=(
                    f"{type(model).__name__} exposes no architecture-owned "
                    "diffusion sampling surface"),
            )
        patches: list[_SamplingPatch] = []
        try:
            for target in targets:
                previous = target.module.diffusion_sampling_config
                target.module.enable_diffusion_sampling(self.config)
                patches.append(_SamplingPatch(target=target, previous=previous))
        except BaseException:
            for patch in reversed(patches):
                patch.target.module.disable_diffusion_sampling()
                if patch.previous is not None:
                    patch.target.module.enable_diffusion_sampling(patch.previous)
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
            raise DiffusionSamplingError("Diffusion-sampling restoration state is missing target patches.")
        errors = []
        for patch in reversed(patches):
            try:
                patch.target.module.disable_diffusion_sampling()
                if patch.previous is not None:
                    patch.target.module.enable_diffusion_sampling(patch.previous)
            except BaseException as error:
                errors.append(error)
        if errors:
            raise DiffusionSamplingError(
                f"Could not restore {len(errors)} diffusion sampling target(s).") from errors[0]
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
        return {patch.target.label: patch.target.module.diffusion_sampling_stats() for patch in patches}


__all__ = [
    "DiffusionGuidanceStrategy",
    "DiffusionPredictionCacheMethod",
    "DiffusionSamplingCompatibilityError",
    "DiffusionSamplingConfig",
    "DiffusionSamplingController",
    "DiffusionSamplingError",
    "DiffusionSamplingMixin",
    "DiffusionSamplingPass",
    "DiffusionSamplingPolicy",
    "DiffusionScheduleStrategy",
    "DiffusionSolverStrategy",
    "DiffusionStepContext",
    "coerce_diffusion_sampling_config",
]
