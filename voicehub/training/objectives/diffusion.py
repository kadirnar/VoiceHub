"""Noising, targets, and regression losses for diffusion and flow TTS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from voicehub.training.objectives._shared import (
    active_selection_mask,
    masked_reduction,
    require_floating_tensor,
    require_tensor,
    torch_module,
)


@dataclass(frozen=True)
class DiffusionTrainingPair:
    """Noisy model input and supervised target for a diffusion recipe."""

    noisy_inputs: Any
    targets: Any
    timesteps: Any
    noise: Any
    alpha: Any
    sigma: Any


def _sample_noise(
    samples: Any,
    *,
    noise: Any | None,
    generator: Any | None,
    noise_sampler: Callable[..., Any] | None,
    torch: Any,
):
    if noise is not None and noise_sampler is not None:
        raise ValueError("Pass either `noise` or `noise_sampler`, not both.")
    if noise is None:
        if noise_sampler is None:
            noise = torch.randn(
                samples.shape,
                device=samples.device,
                dtype=samples.dtype,
                generator=generator,
            )
        else:
            noise = noise_sampler(samples, generator=generator)
    noise = require_floating_tensor(noise, name="noise", torch=torch)
    if tuple(noise.shape) != tuple(samples.shape):
        raise ValueError(
            "`noise` must exactly match samples; received "
            f"{tuple(noise.shape)} and {tuple(samples.shape)}.")
    return noise.to(device=samples.device, dtype=samples.dtype)


def _sample_timesteps(
    samples: Any,
    *,
    timesteps: Any | None,
    generator: Any | None,
    timestep_sampler: Callable[..., Any] | None,
    num_train_timesteps: int | None,
    continuous: bool,
    torch: Any,
):
    if timesteps is not None and timestep_sampler is not None:
        raise ValueError("Pass either `timesteps` or `timestep_sampler`, not both.")
    batch_size = int(samples.shape[0])
    if timesteps is None:
        if timestep_sampler is not None:
            timesteps = timestep_sampler(
                batch_size,
                device=samples.device,
                generator=generator,
            )
        elif continuous:
            timesteps = torch.rand(
                batch_size,
                device=samples.device,
                dtype=samples.dtype,
                generator=generator,
            )
        else:
            if (isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, int) or
                    num_train_timesteps <= 0):
                raise ValueError(
                    "Discrete diffusion sampling requires positive "
                    "`num_train_timesteps`, explicit `timesteps`, or a "
                    "`timestep_sampler`.")
            timesteps = torch.randint(
                0,
                num_train_timesteps,
                (batch_size, ),
                device=samples.device,
                generator=generator,
            )
    timesteps = require_tensor(
        timesteps,
        name="timesteps",
        torch=torch,
    )
    if tuple(timesteps.shape) != (batch_size, ):
        raise ValueError(
            "`timesteps` must have shape [batch]; received "
            f"{tuple(timesteps.shape)} for batch size {batch_size}.")
    if timesteps.dtype == torch.bool or timesteps.is_complex():
        raise TypeError("`timesteps` must use a real numeric dtype.")
    if not continuous:
        if timesteps.is_floating_point():
            raise TypeError("Discrete diffusion `timesteps` must use an integer dtype.")
        if bool((timesteps < 0).any().item()):
            raise ValueError("Discrete diffusion `timesteps` must be non-negative.")
        if num_train_timesteps is not None:
            if (isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, int) or
                    num_train_timesteps <= 0):
                raise ValueError("`num_train_timesteps` must be a positive integer.")
            if bool((timesteps >= num_train_timesteps).any().item()):
                raise ValueError(
                    "Discrete diffusion `timesteps` must be smaller than "
                    "`num_train_timesteps`.")
    return timesteps.to(device=samples.device)


def _coefficient_tensor(
    value: Any,
    samples: Any,
    *,
    name: str,
    torch: Any,
):
    if not torch.is_tensor(value):
        try:
            value = torch.as_tensor(
                value,
                device=samples.device,
                dtype=samples.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(f"`coefficient_fn` must return tensor-like {name}.") from exc
    value = require_floating_tensor(value, name=name, torch=torch)
    if value.ndim == 1 and value.shape[0] == samples.shape[0]:
        value = value.reshape((value.shape[0], ) + (1, ) * (samples.ndim - 1))
    if value.ndim > samples.ndim:
        raise ValueError(f"`{name}` has more dimensions than samples.")
    while value.ndim < samples.ndim:
        value = value.unsqueeze(-1)
    for index, (actual, expected) in enumerate(zip(value.shape, samples.shape)):
        if actual not in (1, expected):
            raise ValueError(
                f"`{name}` shape {tuple(value.shape)} cannot scale samples "
                f"shape {tuple(samples.shape)} at dimension {index}.")
    value = value.to(
        device=samples.device,
        dtype=samples.dtype,
    )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"`{name}` must contain only finite values.")
    return value


def _prediction_target(
    prediction_type: str,
    *,
    samples: Any,
    noise: Any,
    alpha: Any,
    sigma: Any,
    flow_velocity: bool,
):
    normalized = str(prediction_type).strip().lower().replace("-", "_")
    aliases = {
        "noise": "epsilon",
        "eps": "epsilon",
        "v_prediction": "velocity",
        "x0": "sample",
        "original_sample": "sample",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "epsilon":
        return noise
    if normalized == "sample":
        return samples
    if normalized == "velocity":
        # Linear flow matching uses d((1-t)x + t*noise)/dt. Diffusion
        # v-prediction instead uses alpha*noise - sigma*x.
        return noise - samples if flow_velocity else alpha * noise - sigma * samples
    raise ValueError("`prediction_type` must be 'epsilon', 'velocity', or 'sample'.")


def build_diffusion_training_pair(
    samples: Any,
    *,
    coefficient_fn: Callable[[Any, Any], tuple[Any, Any]],
    prediction_type: str = "epsilon",
    timesteps: Any | None = None,
    noise: Any | None = None,
    generator: Any | None = None,
    timestep_sampler: Callable[..., Any] | None = None,
    noise_sampler: Callable[..., Any] | None = None,
    num_train_timesteps: int | None = None,
) -> DiffusionTrainingPair:
    """Construct a discrete diffusion input and epsilon/v/sample target.

    ``coefficient_fn(timesteps, samples)`` must return broadcastable
    ``(alpha, sigma)`` coefficients used as
    ``noisy = alpha * samples + sigma * noise``.  Velocity prediction uses
    the common diffusion convention ``alpha * noise - sigma * samples``.

    Randomness is fully injectable.  Pass concrete ``timesteps``/``noise``,
    one or both sampler hooks, and optionally a ``torch.Generator``.  The
    default timestep sampler uses integer values in
    ``[0, num_train_timesteps)``; default noise is standard normal.
    """
    torch = torch_module()
    samples = require_floating_tensor(
        samples,
        name="samples",
        torch=torch,
    )
    if samples.ndim < 1 or samples.shape[0] <= 0:
        raise ValueError("`samples` must have a non-empty batch dimension.")
    if not callable(coefficient_fn):
        raise TypeError("`coefficient_fn` must be callable.")
    timesteps = _sample_timesteps(
        samples,
        timesteps=timesteps,
        generator=generator,
        timestep_sampler=timestep_sampler,
        num_train_timesteps=num_train_timesteps,
        continuous=False,
        torch=torch,
    )
    noise = _sample_noise(
        samples,
        noise=noise,
        generator=generator,
        noise_sampler=noise_sampler,
        torch=torch,
    )
    coefficients = coefficient_fn(timesteps, samples)
    if not isinstance(coefficients, (tuple, list)) or len(coefficients) != 2:
        raise TypeError("`coefficient_fn` must return an (alpha, sigma) pair.")
    alpha = _coefficient_tensor(
        coefficients[0],
        samples,
        name="alpha",
        torch=torch,
    )
    sigma = _coefficient_tensor(
        coefficients[1],
        samples,
        name="sigma",
        torch=torch,
    )
    noisy_inputs = alpha * samples + sigma * noise
    targets = _prediction_target(
        prediction_type,
        samples=samples,
        noise=noise,
        alpha=alpha,
        sigma=sigma,
        flow_velocity=False,
    )
    return DiffusionTrainingPair(
        noisy_inputs=noisy_inputs,
        targets=targets,
        timesteps=timesteps,
        noise=noise,
        alpha=alpha,
        sigma=sigma,
    )


def build_flow_matching_training_pair(
    samples: Any,
    *,
    prediction_type: str = "velocity",
    timesteps: Any | None = None,
    noise: Any | None = None,
    generator: Any | None = None,
    timestep_sampler: Callable[..., Any] | None = None,
    noise_sampler: Callable[..., Any] | None = None,
) -> DiffusionTrainingPair:
    """Construct a linear flow-matching path and its supervised target.

    The path is ``(1 - t) * samples + t * noise`` for continuous
    ``t in [0, 1]``.  Its velocity target is therefore ``noise - samples``.
    Epsilon and sample prediction return ``noise`` and ``samples``
    respectively.
    """
    torch = torch_module()
    samples = require_floating_tensor(
        samples,
        name="samples",
        torch=torch,
    )
    if samples.ndim < 1 or samples.shape[0] <= 0:
        raise ValueError("`samples` must have a non-empty batch dimension.")
    timesteps = _sample_timesteps(
        samples,
        timesteps=timesteps,
        generator=generator,
        timestep_sampler=timestep_sampler,
        num_train_timesteps=None,
        continuous=True,
        torch=torch,
    ).to(dtype=samples.dtype)
    finite_timesteps = bool(torch.isfinite(timesteps).all().item())
    out_of_range = bool(((timesteps < 0) | (timesteps > 1)).any().item())
    if not finite_timesteps or out_of_range:
        raise ValueError("Flow-matching `timesteps` must be finite and in [0, 1].")
    noise = _sample_noise(
        samples,
        noise=noise,
        generator=generator,
        noise_sampler=noise_sampler,
        torch=torch,
    )
    alpha = _coefficient_tensor(
        1 - timesteps,
        samples,
        name="alpha",
        torch=torch,
    )
    sigma = _coefficient_tensor(
        timesteps,
        samples,
        name="sigma",
        torch=torch,
    )
    noisy_inputs = alpha * samples + sigma * noise
    targets = _prediction_target(
        prediction_type,
        samples=samples,
        noise=noise,
        alpha=alpha,
        sigma=sigma,
        flow_velocity=True,
    )
    return DiffusionTrainingPair(
        noisy_inputs=noisy_inputs,
        targets=targets,
        timesteps=timesteps,
        noise=noise,
        alpha=alpha,
        sigma=sigma,
    )


def masked_diffusion_regression_loss(
    predictions: Any,
    targets: Any,
    *,
    mask: Any | None = None,
    weights: Any | None = None,
    loss_type: str = "mse",
    reduction: str = "mean",
):
    """Compute exact-shape masked diffusion or flow regression.

    Predictions and targets must have identical floating-point shapes.
    Masks and non-negative weights may match that shape, use singleton
    dimensions, or omit trailing feature dimensions.  Padding is
    therefore excluded explicitly instead of contributing zero-valued
    targets.
    """
    torch = torch_module()
    predictions = require_floating_tensor(
        predictions,
        name="predictions",
        torch=torch,
    )
    targets = require_floating_tensor(
        targets,
        name="targets",
        torch=torch,
    )
    if tuple(predictions.shape) != tuple(targets.shape):
        raise ValueError(
            "`predictions` and `targets` must have identical shapes; "
            f"received {tuple(predictions.shape)} and {tuple(targets.shape)}.")
    targets = targets.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )
    _, _, active = active_selection_mask(
        predictions,
        mask=mask,
        weights=weights,
        torch=torch,
    )
    zero = torch.zeros(
        (),
        device=predictions.device,
        dtype=predictions.dtype,
    )
    predictions = torch.where(active, predictions, zero)
    targets = torch.where(active, targets, zero)
    normalized = str(loss_type).strip().lower().replace("-", "_")
    if normalized in ("mse", "l2"):
        values = (predictions - targets).square()
    elif normalized in ("l1", "mae"):
        values = (predictions - targets).abs()
    elif normalized in ("smooth_l1", "huber"):
        values = torch.nn.functional.smooth_l1_loss(
            predictions,
            targets,
            reduction="none",
        )
    else:
        raise ValueError("`loss_type` must be 'mse', 'l1', or 'smooth_l1'.")
    return masked_reduction(
        values,
        mask=mask,
        weights=weights,
        reduction=reduction,
        torch=torch,
    )


__all__ = [
    "DiffusionTrainingPair",
    "build_diffusion_training_pair",
    "build_flow_matching_training_pair",
    "masked_diffusion_regression_loss",
]
