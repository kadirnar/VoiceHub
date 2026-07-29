"""Numerically stable connectionist temporal classification objective."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional

_REDUCTIONS = frozenset({"none", "mean", "sum"})


def _require_integer_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor.")
    if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
        raise TypeError(f"`{name}` must use an integer dtype.")


def _validate_lengths(
    input_lengths: Tensor,
    target_lengths: Tensor,
    *,
    batch_size: int,
    input_steps: int,
    targets: Tensor,
) -> None:
    for name, value in (
        ("input_lengths", input_lengths),
        ("target_lengths", target_lengths),
    ):
        _require_integer_tensor(name, value)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError(f"`{name}` must have shape [{batch_size}].")
        if (value < 0).any():
            raise ValueError(f"`{name}` cannot contain negative lengths.")

    if (input_lengths > input_steps).any():
        raise ValueError("`input_lengths` exceeds the available logit time steps.")
    if targets.ndim == 1:
        if int(target_lengths.sum().item()) != targets.numel():
            raise ValueError(
                "For concatenated one-dimensional targets, `target_lengths` "
                "must sum to the number of target tokens.")
    elif (target_lengths > targets.shape[1]).any():
        raise ValueError("`target_lengths` exceeds the padded target width.")


def _valid_targets(targets: Tensor, target_lengths: Tensor) -> Tensor:
    if targets.ndim == 1:
        return targets
    positions = torch.arange(targets.shape[1], device=target_lengths.device)
    mask = positions[None, :] < target_lengths[:, None]
    if mask.device != targets.device:
        mask = mask.to(targets.device)
    return targets[mask]


def ctc_loss(
    logits: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    *,
    blank: int = 0,
    reduction: str = "mean",
    zero_infinity: bool = True,
    time_major: bool = False,
) -> Tensor:
    """Compute CTC loss from unnormalized logits.

    Log-softmax is always evaluated in float32 for half and bfloat16 inputs.
    Both padded targets shaped ``[batch, target_time]`` and concatenated
    one-dimensional targets are supported.
    """
    if not isinstance(logits, Tensor):
        raise TypeError("`logits` must be a PyTorch tensor.")
    if logits.ndim != 3:
        raise ValueError("`logits` must have shape [batch, time, classes] or "
                         "[time, batch, classes].")
    if not logits.is_floating_point():
        raise TypeError("`logits` must use a floating-point dtype.")
    if not isinstance(time_major, bool):
        raise TypeError("`time_major` must be a boolean.")
    if not isinstance(zero_infinity, bool):
        raise TypeError("`zero_infinity` must be a boolean.")
    if reduction not in _REDUCTIONS:
        choices = ", ".join(sorted(_REDUCTIONS))
        raise ValueError(f"`reduction` must be one of {choices}; found {reduction!r}.")

    batch_axis = 1 if time_major else 0
    time_axis = 0 if time_major else 1
    batch_size = logits.shape[batch_axis]
    input_steps = logits.shape[time_axis]
    class_count = logits.shape[-1]
    if batch_size == 0 or input_steps == 0 or class_count < 2:
        raise ValueError("CTC logits require a non-empty batch/time axis and at least two classes.")
    if isinstance(blank, bool) or not isinstance(blank, int):
        raise TypeError("`blank` must be an integer.")
    if not 0 <= blank < class_count:
        raise ValueError("`blank` must identify a class in the logits vocabulary.")

    _require_integer_tensor("targets", targets)
    if targets.ndim not in (1, 2):
        raise ValueError("`targets` must be a concatenated vector or a padded matrix.")
    if targets.ndim == 2 and targets.shape[0] != batch_size:
        raise ValueError(f"Padded `targets` must have batch size {batch_size}.")
    if targets.device != logits.device:
        raise ValueError("`targets` and `logits` must be on the same device.")
    _validate_lengths(
        input_lengths,
        target_lengths,
        batch_size=batch_size,
        input_steps=input_steps,
        targets=targets,
    )

    valid_targets = _valid_targets(targets, target_lengths)
    if valid_targets.numel():
        if (valid_targets < 0).any() or (valid_targets >= class_count).any():
            raise ValueError("A target ID is outside the CTC class vocabulary.")
        if (valid_targets == blank).any():
            raise ValueError("CTC targets cannot contain the blank token.")

    working_logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    if not time_major:
        working_logits = working_logits.transpose(0, 1)
    log_probabilities = functional.log_softmax(working_logits, dim=-1)
    return functional.ctc_loss(
        log_probabilities,
        targets.long(),
        input_lengths.long(),
        target_lengths.long(),
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )


class CTCLoss(nn.Module):
    """Module wrapper around :func:`ctc_loss`."""

    def __init__(
        self,
        *,
        blank: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = True,
        time_major: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(blank, bool) or not isinstance(blank, int):
            raise TypeError("`blank` must be an integer.")
        if blank < 0:
            raise ValueError("`blank` cannot be negative.")
        if reduction not in _REDUCTIONS:
            choices = ", ".join(sorted(_REDUCTIONS))
            raise ValueError(f"`reduction` must be one of {choices}; found {reduction!r}.")
        if not isinstance(zero_infinity, bool):
            raise TypeError("`zero_infinity` must be a boolean.")
        if not isinstance(time_major, bool):
            raise TypeError("`time_major` must be a boolean.")
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.time_major = time_major

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
    ) -> Tensor:
        return ctc_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
            blank=self.blank,
            reduction=self.reduction,
            zero_infinity=self.zero_infinity,
            time_major=self.time_major,
        )
