"""Dependency-free diagnostics used by native Vocos training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spectrogram_to_rgb(
    spectrogram: torch.Tensor,
    *,
    height: int = 256,
) -> torch.Tensor:
    """Render a spectrogram as a compact ``uint8`` RGB tensor.

    This intentionally returns a tensor instead of importing Matplotlib and
    NumPy into the training runtime.  Experiment trackers commonly accept
    ``[channels, height, width]`` tensors directly.
    """
    values = torch.as_tensor(spectrogram).detach().float().cpu().squeeze()
    if values.ndim != 2:
        raise ValueError("Spectrogram rendering expects a two-dimensional tensor.")
    if (
        isinstance(height, bool)
        or not isinstance(height, int)
        or height <= 0
    ):
        raise ValueError("`height` must be a positive integer.")
    finite = torch.isfinite(values)
    if not bool(finite.any().item()):
        values = torch.zeros_like(values)
    else:
        minimum = values[finite].amin()
        maximum = values[finite].amax()
        values = torch.where(finite, values, minimum)
        scale = (maximum - minimum).clamp_min(torch.finfo(values.dtype).eps)
        values = (values - minimum) / scale
    if values.shape[0] != height:
        width = max(1, round(values.shape[1] * height / values.shape[0]))
        values = F.interpolate(
            values[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    # A small perceptually ordered blue/cyan/yellow map.
    red = (1.8 * values - 0.8).clamp(0.0, 1.0)
    green = (1.6 * values).clamp(0.0, 1.0)
    blue = (1.2 - 1.4 * values).clamp(0.0, 1.0)
    return (torch.stack((red, green, blue)) * 255.0).round().to(torch.uint8)


def plot_spectrogram_to_numpy(spectrogram: torch.Tensor) -> torch.Tensor:
    """Compatibility alias returning a native RGB tensor.

    The historical name is retained for callers, but no NumPy object is
    created.  New code should use :func:`spectrogram_to_rgb`.
    """
    return spectrogram_to_rgb(spectrogram)


def gradient_norm(
    model: torch.nn.Module,
    norm_type: float = 2.0,
) -> torch.Tensor:
    """Compute one finite aggregate gradient norm."""
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        reference = next(model.parameters(), None)
        return torch.zeros(
            (),
            device=None if reference is None else reference.device,
        )
    norms = torch.stack(
        [torch.linalg.vector_norm(gradient, ord=norm_type) for gradient in gradients]
    )
    return torch.linalg.vector_norm(norms, ord=norm_type)


class GradNormCallback:
    """Tracker-neutral callback compatible with VoiceHub's trainer hooks."""

    def on_after_backward(self, trainer, model) -> None:
        del trainer
        log = getattr(model, "log", None)
        if callable(log):
            log("grad_norm", gradient_norm(model))


__all__ = [
    "GradNormCallback",
    "gradient_norm",
    "plot_spectrogram_to_numpy",
    "spectrogram_to_rgb",
]
