from importlib import import_module
from typing import Optional

import torch

_SGL_TOPK_UNSET = object()
_sgl_topk_softmax = _SGL_TOPK_UNSET


def _load_sgl_topk():
    global _sgl_topk_softmax
    if _sgl_topk_softmax is not _SGL_TOPK_UNSET:
        return _sgl_topk_softmax
    try:
        module = import_module("sgl_kernel")
        _sgl_topk_softmax = module.topk_softmax
    except (AttributeError, ImportError, OSError, RuntimeError):
        _sgl_topk_softmax = None
    return _sgl_topk_softmax


def _torch_topk(
    gating_output: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(
        gating_output.float(),
        dim=-1,
    )
    topk_weights, topk_ids = torch.topk(
        probabilities,
        k=topk,
        dim=-1,
    )
    return topk_weights, topk_ids.to(dtype=torch.int32)


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if hidden_states.ndim != 2 or gating_output.ndim != 2:
        raise ValueError(
            "MoE hidden states and router logits must both be rank two."
        )
    if hidden_states.shape[0] != gating_output.shape[0]:
        raise ValueError("MoE hidden states and router logits must have one token count.")
    if hidden_states.device != gating_output.device:
        raise ValueError(
            "MoE hidden states and router logits must be on the same device."
        )
    if not gating_output.is_floating_point():
        raise TypeError("MoE router logits must use a floating-point dtype.")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("MoE `topk` must be a positive integer.")
    if topk > gating_output.shape[1]:
        raise ValueError(
            "MoE `topk` cannot exceed the number of experts."
        )

    M, _ = hidden_states.shape

    sgl_topk_softmax = (
        _load_sgl_topk()
        if hidden_states.device.type == "cuda"
        else None
    )
    if sgl_topk_softmax is not None:
        topk_weights = torch.empty(
            M,
            topk,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        sgl_topk_softmax(
            topk_weights,
            topk_ids,
            gating_output.float(),
            renormalize,
        )
    else:
        topk_weights, topk_ids = _torch_topk(
            gating_output,
            topk,
        )

    return _fused_topk_postprocess(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        renormalize=renormalize,
        num_token_non_padded=num_token_non_padded,
    )


def _mask_topk_ids_padded_region(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return

    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    padded = indices >= num_token_non_padded
    topk_ids[padded, :] = -1
    topk_weights[padded, :] = 0


def _fused_topk_postprocess(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor],
):
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)

    _mask_topk_ids_padded_region(
        topk_weights,
        topk_ids,
        num_token_non_padded,
    )
    return topk_weights, topk_ids


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    num_token_non_padded: Optional[torch.Tensor] = None,
    renormalize: bool = True,
):
    topk_weights, topk_ids = fused_topk(
        hidden_states=hidden_states,
        gating_output=router_logits,
        topk=top_k,
        renormalize=renormalize,
        num_token_non_padded=num_token_non_padded,
    )
    return topk_weights, topk_ids
