"""Specialized deterministic ODE solvers for diffusion/flow TTS sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class STORKFlowConfig:
    """STORK-2 settings for a direct deterministic velocity field.

    The model is evaluated once per outer step.  Stabilized virtual
    stages use a first-order Taylor approximation of the velocity along
    the sampled trajectory and therefore add tensor arithmetic but no
    neural-network evaluations.
    """

    stages: int = 9
    accumulator_dtype: str = "float32"

    def __post_init__(self) -> None:
        if (isinstance(self.stages, bool) or not isinstance(self.stages, int) or self.stages < 2):
            raise ValueError("`stages` must be an integer of at least two.")
        if self.accumulator_dtype != "float32":
            raise ValueError("STORK currently requires `accumulator_dtype='float32'`.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver_order": 2,
            "taylor_order": 1,
            "stages": self.stages,
            "accumulator_dtype": self.accumulator_dtype,
        }


@dataclass(slots=True)
class _STORKHistory:
    time: Tensor | None = None
    velocity: Tensor | None = None


def _time_scalar(
    value: Tensor | float,
    *,
    device: torch.device,
) -> Tensor:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("STORK timesteps must be scalar values.")
        output = value.detach().to(device=device, dtype=torch.float32).reshape(())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output = torch.tensor(float(value), device=device, dtype=torch.float32)
    else:
        raise TypeError("STORK timesteps must be scalar tensors or real numbers.")
    if not bool(torch.isfinite(output).item()):
        raise ValueError("STORK timesteps must be finite.")
    return output


def _b_coefficient(index: int) -> float:
    if index == 0:
        return 1.0
    if index == 1:
        return 1.0 / 3.0
    numerator = 4.0 * (index - 1) * (index + 4)
    denominator = 3.0 * index * (index + 1) * (index + 2) * (index + 3)
    return numerator / denominator


class STORK2FlowSolver:
    """Stateful STORK-2/Taylor-1 integrator for one request and solver lane."""

    def __init__(self, config: STORKFlowConfig | None = None):
        self.config = STORKFlowConfig() if config is None else config
        if not isinstance(self.config, STORKFlowConfig):
            raise TypeError("`config` must be a STORKFlowConfig.")
        self._history = _STORKHistory()
        self.steps = 0
        self.startup_steps = 0
        self.stabilized_steps = 0

    def reset(self) -> None:
        self._history = _STORKHistory()

    def stats(self) -> dict[str, int]:
        return {
            "steps": self.steps,
            "startup_steps": self.startup_steps,
            "stabilized_steps": self.stabilized_steps,
            "model_evaluations": self.steps,
            "virtual_stage_evaluations": (self.stabilized_steps * max(0, self.config.stages - 1)),
        }

    def _compatible_history(self, velocity: Tensor) -> bool:
        previous = self._history.velocity
        return (
            previous is not None and previous.shape == velocity.shape and previous.device == velocity.device)

    def advance(
        self,
        state: Tensor,
        velocity: Tensor,
        *,
        timestep: Tensor | float,
        next_timestep: Tensor | float,
        discontinuity: bool = False,
    ) -> Tensor:
        """Advance one signed step with exactly one supplied model
        evaluation."""
        if not isinstance(state, Tensor) or not isinstance(velocity, Tensor):
            raise TypeError("STORK state and velocity must be torch.Tensor values.")
        if state.shape != velocity.shape:
            raise ValueError(
                "STORK state and velocity must have identical shapes "
                f"({tuple(state.shape)} != {tuple(velocity.shape)}).")
        if state.device != velocity.device:
            raise ValueError("STORK state and velocity must use the same device.")
        if discontinuity:
            self.reset()
        current_time = _time_scalar(timestep, device=state.device)
        following_time = _time_scalar(next_timestep, device=state.device)
        step_size = following_time - current_time
        if bool((step_size == 0).item()):
            raise ValueError("STORK requires distinct current and next timesteps.")

        state_fp32 = state.float()
        velocity_fp32 = velocity.float()
        previous_time = self._history.time
        compatible = self._compatible_history(velocity_fp32)
        self.steps += 1
        if previous_time is None or not compatible:
            output = state_fp32 + step_size * velocity_fp32
            self.startup_steps += 1
        else:
            history_delta = current_time - previous_time
            if bool((history_delta == 0).item()):
                output = state_fp32 + step_size * velocity_fp32
                self.startup_steps += 1
            else:
                derivative = (velocity_fp32 - self._history.velocity) / history_delta
                output = self._stabilized_step(
                    state_fp32,
                    velocity_fp32,
                    derivative,
                    step_size,
                )
                self.stabilized_steps += 1
        self._history.time = current_time
        self._history.velocity = velocity_fp32.detach()
        return output.to(dtype=state.dtype)

    def _stabilized_step(
        self,
        state: Tensor,
        velocity: Tensor,
        derivative: Tensor,
        step_size: Tensor,
    ) -> Tensor:
        stages = self.config.stages
        omega = 6.0 / ((stages + 4) * (stages - 1))
        initial = state
        previous_previous = initial
        previous = initial + step_size * omega * velocity
        denominator = stages * stages + stages - 2
        for stage in range(2, stages + 1):
            current_b = _b_coefficient(stage)
            previous_b = _b_coefficient(stage - 1)
            previous_previous_b = _b_coefficient(stage - 2)
            mu = ((2 * stage + 1) * current_b / (stage * previous_b))
            nu = (-(stage + 1) * current_b / (stage * previous_previous_b))
            mu_tilde = mu * omega
            gamma_tilde = -mu_tilde * (1 - stage * (stage + 1) * previous_b / 2)
            if stage == 2:
                stage_fraction = 4.0 / (3.0 * denominator)
            else:
                prior = stage - 1
                stage_fraction = (prior * prior + prior - 2) / denominator
            virtual_velocity = (velocity + step_size * stage_fraction * derivative)
            current = (
                mu * previous + nu * previous_previous + (1 - mu - nu) * initial +
                step_size * mu_tilde * virtual_velocity + step_size * gamma_tilde * velocity)
            previous_previous, previous = previous, current
        return previous


__all__ = [
    "STORK2FlowSolver",
    "STORKFlowConfig",
]
