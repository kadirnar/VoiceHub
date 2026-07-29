"""Token-level objectives for autoregressive and multi-codebook TTS."""

from __future__ import annotations

from typing import Any

from voicehub.training.objectives._shared import (
    masked_reduction,
    normalize_dim,
    require_floating_tensor,
    require_tensor,
    slice_dimension,
    torch_module,
)


def multi_codebook_cross_entropy(
    logits: Any,
    labels: Any,
    *,
    loss_mask: Any | None = None,
    ignore_index: int = -100,
    causal_shift: bool = False,
    sequence_dim: int = -1,
    codebook_weights: Any | None = None,
    codebook_dim: int = 1,
    reduction: str = "mean",
):
    """Compute exact-shape token cross entropy for one or more codebooks.

    ``logits`` must have shape ``labels.shape + (vocabulary_size,)``.
    This supports layouts such as ``[batch, time, vocab]``, ``[batch,
    codebook, time, vocab]``, or ``[batch, time, codebook, vocab]``
    without guessing the codebook axis.

    When ``causal_shift`` is true, logits at positions ``[:-1]`` predict
    labels at positions ``[1:]`` along ``sequence_dim``.  ``loss_mask``
    must exactly match the unshifted labels.  Ignored labels and false
    mask entries are excluded from the reduction.

    Optional ``codebook_weights`` must be a one-dimensional floating
    tensor. ``codebook_dim`` identifies its axis in ``labels``.  A
    weighted mean is normalized by the sum of selected weights, not by
    the number of tokens.
    """
    torch = torch_module()
    logits = require_floating_tensor(
        logits,
        name="logits",
        torch=torch,
    )
    labels = require_tensor(labels, name="labels", torch=torch)
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise TypeError("`labels` must use a non-boolean integer dtype.")
    labels = labels.to(device=logits.device)
    if logits.ndim < 2:
        raise ValueError("`logits` must have rank two or greater.")
    if tuple(logits.shape[:-1]) != tuple(labels.shape):
        raise ValueError(
            "`logits` must have shape labels.shape + (vocabulary_size,); "
            f"received {tuple(logits.shape)} and {tuple(labels.shape)}.")
    if logits.shape[-1] <= 1:
        raise ValueError("The logits vocabulary dimension must exceed one.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("`ignore_index` must be an integer.")

    if loss_mask is not None:
        loss_mask = require_tensor(
            loss_mask,
            name="loss_mask",
            torch=torch,
        )
        if tuple(loss_mask.shape) != tuple(labels.shape):
            raise ValueError(
                "`loss_mask` must exactly match labels; received "
                f"{tuple(loss_mask.shape)} and {tuple(labels.shape)}.")
        loss_mask = loss_mask.to(device=labels.device, dtype=torch.bool)

    sequence_dim = normalize_dim(
        sequence_dim,
        labels.ndim,
        name="sequence_dim",
    )
    if causal_shift:
        if labels.shape[sequence_dim] < 2:
            raise ValueError("Causal cross entropy requires at least two sequence positions.")
        logits = slice_dimension(logits, sequence_dim, None, -1)
        labels = slice_dimension(labels, sequence_dim, 1, None)
        if loss_mask is not None:
            loss_mask = slice_dimension(
                loss_mask,
                sequence_dim,
                1,
                None,
            )

    valid = labels.ne(ignore_index)
    if loss_mask is not None:
        valid = valid & loss_mask

    weights = None
    if codebook_weights is not None:
        codebook_weights = require_floating_tensor(
            codebook_weights,
            name="codebook_weights",
            torch=torch,
        )
        if codebook_weights.ndim != 1:
            raise ValueError("`codebook_weights` must be one-dimensional.")
        codebook_dim = normalize_dim(
            codebook_dim,
            labels.ndim,
            name="codebook_dim",
        )
        if codebook_dim == sequence_dim:
            raise ValueError("`codebook_dim` and `sequence_dim` must identify different axes.")
        if codebook_weights.shape[0] != labels.shape[codebook_dim]:
            raise ValueError(
                "`codebook_weights` length must match the selected codebook "
                f"axis ({codebook_weights.shape[0]} != "
                f"{labels.shape[codebook_dim]}).")
        shape = [1] * labels.ndim
        shape[codebook_dim] = int(codebook_weights.shape[0])
        weights = codebook_weights.reshape(shape).to(
            device=logits.device,
            dtype=logits.dtype,
        ).expand(labels.shape)

    active = valid.to(device=logits.device)
    if weights is not None:
        active = active & weights.ne(0)
    safe_logits = torch.where(
        active.unsqueeze(-1),
        logits,
        torch.zeros((), device=logits.device, dtype=logits.dtype),
    )
    safe_labels = torch.where(
        active.to(device=labels.device),
        labels,
        torch.zeros((), device=labels.device, dtype=labels.dtype),
    )
    per_token = torch.nn.functional.cross_entropy(
        safe_logits.reshape(-1, safe_logits.shape[-1]),
        safe_labels.reshape(-1).long(),
        reduction="none",
    ).reshape(labels.shape)

    return masked_reduction(
        per_token,
        mask=valid,
        weights=weights,
        reduction=reduction,
        torch=torch,
    )


__all__ = ["multi_codebook_cross_entropy"]
