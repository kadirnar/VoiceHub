"""Framework-lazy validation and reduction helpers for TTS objectives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from voicehub.dependencies import import_optional


def torch_module():
    """Resolve PyTorch only when objective math is actually requested."""
    return import_optional(
        "torch",
        model_type="TTS training objectives",
        install_extra="training",
    )


def require_tensor(value: Any, *, name: str, torch: Any):
    if not torch.is_tensor(value):
        raise TypeError(f"`{name}` must be a PyTorch tensor.")
    return value


def require_floating_tensor(value: Any, *, name: str, torch: Any):
    value = require_tensor(value, name=name, torch=torch)
    if not value.is_floating_point():
        raise TypeError(f"`{name}` must use a floating-point dtype.")
    return value


def normalize_dim(dimension: int, rank: int, *, name: str) -> int:
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError(f"`{name}` must be an integer.")
    normalized = dimension + rank if dimension < 0 else dimension
    if not 0 <= normalized < rank:
        raise ValueError(f"`{name}`={dimension} is invalid for a rank-{rank} tensor.")
    return normalized


def slice_dimension(
    value: Any,
    dimension: int,
    start: int | None,
    stop: int | None,
):
    slices = [slice(None)] * value.ndim
    slices[dimension] = slice(start, stop)
    return value[tuple(slices)]


def expand_mask(mask: Any, target: Any, *, name: str, torch: Any):
    """Expand a mask without permitting arbitrary broadcasting.

    A mask may match the target, use singleton dimensions at target
    positions, or omit trailing target dimensions.  For example,
    ``[batch, time]`` can mask ``[batch, time, channels]`` and ``[batch,
    1, time]`` can mask ``[batch, channels, time]``.
    """
    mask = require_tensor(mask, name=name, torch=torch)
    if mask.ndim > target.ndim:
        raise ValueError(f"`{name}` rank {mask.ndim} exceeds target rank {target.ndim}.")
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    for index, (actual, expected) in enumerate(zip(mask.shape, target.shape)):
        if actual not in (1, expected):
            raise ValueError(
                f"`{name}` shape {tuple(mask.shape)} cannot mask target "
                f"shape {tuple(target.shape)}; dimension {index} must be "
                f"1 or {expected}.")
    return mask.to(device=target.device, dtype=torch.bool).expand_as(target)


def expand_weights(weights: Any, target: Any, *, torch: Any):
    """Validate and expand non-negative weights like a trailing mask."""
    weights = require_floating_tensor(
        weights,
        name="weights",
        torch=torch,
    )
    if weights.ndim > target.ndim:
        raise ValueError(f"`weights` rank {weights.ndim} exceeds target rank {target.ndim}.")
    while weights.ndim < target.ndim:
        weights = weights.unsqueeze(-1)
    for index, (actual, expected) in enumerate(zip(weights.shape, target.shape)):
        if actual not in (1, expected):
            raise ValueError(
                f"`weights` shape {tuple(weights.shape)} cannot weight target "
                f"shape {tuple(target.shape)}; dimension {index} must be "
                f"1 or {expected}.")
    weights = weights.to(
        device=target.device,
        dtype=target.dtype,
    ).expand_as(target)
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("`weights` must contain only finite values.")
    if bool((weights < 0).any().item()):
        raise ValueError("`weights` must be non-negative.")
    return weights


def masked_reduction(
    values: Any,
    *,
    mask: Any | None,
    weights: Any | None,
    reduction: str,
    torch: Any,
):
    """Apply strict mask/weight expansion and a sum, mean, or no reduction."""
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("`reduction` must be 'mean', 'sum', or 'none'.")

    valid, effective_weights, active = active_selection_mask(
        values,
        mask=mask,
        weights=weights,
        torch=torch,
    )
    selected = torch.where(
        active,
        values,
        torch.zeros((), device=values.device, dtype=values.dtype),
    )
    if effective_weights is not None:
        selected = selected * effective_weights
    if reduction == "none":
        return selected
    if not bool(valid.any().item()):
        raise ValueError("The loss mask does not select any elements.")
    if reduction == "sum":
        return selected.sum()

    if effective_weights is None:
        denominator = valid.sum().to(dtype=values.dtype)
    else:
        denominator = (effective_weights * valid.to(dtype=values.dtype)).sum()
        if not bool((denominator > 0).item()):
            raise ValueError("The selected loss weights must sum to a positive value.")
    return selected.sum() / denominator


def active_selection_mask(
    target: Any,
    *,
    mask: Any | None,
    weights: Any | None,
    torch: Any,
):
    """Return validated reduction masks before objective math is evaluated."""
    valid = (
        torch.ones_like(target, dtype=torch.bool) if mask is None else expand_mask(
            mask,
            target,
            name="mask",
            torch=torch,
        ))
    effective_weights = (None if weights is None else expand_weights(
        weights,
        target,
        torch=torch,
    ))
    active = (valid if effective_weights is None else valid & effective_weights.ne(0))
    return valid, effective_weights, active


def tensor_sequence(value: Any, *, name: str, torch: Any) -> tuple[Any, ...]:
    """Normalize one tensor or a non-empty tensor sequence."""
    if torch.is_tensor(value):
        values = (value, )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        raise TypeError(f"`{name}` must be a tensor or sequence of tensors.")
    if not values:
        raise ValueError(f"`{name}` cannot be empty.")
    return tuple(
        require_floating_tensor(
            item,
            name=f"{name}[{index}]",
            torch=torch,
        ) for index, item in enumerate(values))


def mask_sequence(
    masks: Any | None,
    count: int,
    *,
    name: str,
    torch: Any,
) -> tuple[Any | None, ...]:
    """Normalize one mask per discriminator output."""
    if masks is None:
        return (None, ) * count
    if torch.is_tensor(masks):
        if count != 1:
            raise ValueError(f"A single `{name}` tensor can only mask one output tensor.")
        return (masks, )
    if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
        raise TypeError(f"`{name}` must be a tensor, sequence, or None.")
    masks = tuple(masks)
    if len(masks) != count:
        raise ValueError(f"`{name}` must contain one entry per output ({len(masks)} != "
                         f"{count}).")
    return masks
