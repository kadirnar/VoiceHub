"""Stable token-level objectives for encoder-decoder speech models."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional

_REDUCTIONS = frozenset({"none", "mean", "sum"})


def _validate_reduction(reduction: str) -> None:
    if reduction not in _REDUCTIONS:
        choices = ", ".join(sorted(_REDUCTIONS))
        raise ValueError(f"`reduction` must be one of {choices}; found {reduction!r}.")


def _validate_integer_targets(targets: Tensor) -> None:
    if targets.dtype == torch.bool or targets.is_floating_point() or targets.is_complex():
        raise TypeError("`targets` must use an integer dtype.")


def sequence_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    *,
    attention_mask: Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> Tensor:
    """Compute masked sequence cross entropy with safe mixed-precision math.

    Half-precision and bfloat16 logits are promoted to float32 for the
    log-softmax calculation. ``attention_mask`` and ``ignore_index`` are
    combined, and a fully masked batch returns a differentiable zero instead
    of ``NaN``.
    """
    if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("`logits` and `targets` must be PyTorch tensors.")
    if logits.ndim < 2:
        raise ValueError("`logits` must have shape [..., vocabulary].")
    if logits.shape[-1] == 0:
        raise ValueError("The logits vocabulary cannot be empty.")
    if tuple(targets.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            "`targets` must match every non-vocabulary logits dimension: "
            f"expected {tuple(logits.shape[:-1])}, found {tuple(targets.shape)}.")
    if not logits.is_floating_point():
        raise TypeError("`logits` must use a floating-point dtype.")
    _validate_integer_targets(targets)
    if logits.device != targets.device:
        raise ValueError("`logits` and `targets` must be on the same device.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("`ignore_index` must be an integer.")
    if not isinstance(label_smoothing, (int, float)) or isinstance(label_smoothing, bool):
        raise TypeError("`label_smoothing` must be a real number.")
    label_smoothing = float(label_smoothing)
    if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
        raise ValueError("`label_smoothing` must be finite and in the interval [0, 1).")
    _validate_reduction(reduction)

    long_targets = targets.long()
    valid = long_targets != ignore_index
    if attention_mask is not None:
        if not isinstance(attention_mask, Tensor):
            raise TypeError("`attention_mask` must be a PyTorch tensor.")
        if attention_mask.shape != targets.shape:
            raise ValueError(
                "`attention_mask` must have the same shape as `targets`: "
                f"expected {tuple(targets.shape)}, found {tuple(attention_mask.shape)}.")
        if attention_mask.device != targets.device:
            raise ValueError("`attention_mask` and `targets` must be on the same device.")
        valid &= attention_mask.to(dtype=torch.bool)

    valid_targets = long_targets[valid]
    if valid_targets.numel():
        if (valid_targets < 0).any() or (valid_targets >= logits.shape[-1]).any():
            raise ValueError("A non-ignored target ID is outside the logits vocabulary.")

    working_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    flat_logits = working_logits.reshape(-1, working_logits.shape[-1])
    effective_targets = long_targets.masked_fill(~valid, ignore_index)
    flat_targets = effective_targets.reshape(-1)
    losses = functional.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    ).reshape(targets.shape)
    losses = losses.masked_fill(~valid, 0.0)

    if reduction == "none":
        return losses
    total = losses.sum()
    if reduction == "sum":
        return total
    count = valid.sum().clamp_min(1).to(dtype=total.dtype)
    return total / count


class Seq2SeqCrossEntropyLoss(nn.Module):
    """Module wrapper around :func:`sequence_cross_entropy`."""

    def __init__(
        self,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
            raise TypeError("`ignore_index` must be an integer.")
        if not isinstance(label_smoothing, (int, float)) or isinstance(label_smoothing, bool):
            raise TypeError("`label_smoothing` must be a real number.")
        label_smoothing = float(label_smoothing)
        if not math.isfinite(label_smoothing) or not 0.0 <= label_smoothing < 1.0:
            raise ValueError("`label_smoothing` must be finite and in the interval [0, 1).")
        _validate_reduction(reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return sequence_cross_entropy(
            logits,
            targets,
            attention_mask=attention_mask,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
