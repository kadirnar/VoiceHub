import functools
from dataclasses import dataclass
from importlib import import_module
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe.topk import select_experts


@dataclass(frozen=True)
class _OptimizedKernels:
    """Optional CUDA-only kernels used by the released Zonos2 runtime."""

    triton: object
    triton_language: object
    gelu_and_mul: object
    silu_and_mul: object
    moe_align_block_size: object
    fused_moe_kernel: object
    moe_sum_reduce: object


_OPTIMIZED_KERNELS_UNSET = object()
_optimized_kernels: _OptimizedKernels | None | object = (
    _OPTIMIZED_KERNELS_UNSET
)


def _load_optimized_kernels() -> _OptimizedKernels | None:
    """Load Triton/SGL kernels only when an accelerated call needs them.

    Triton and ``sgl_kernel`` are not available on the supported macOS and
    Windows CPU runtimes. Importing them at module scope made the entire
    Zonos2 model unavailable before a device could be selected.
    """
    global _optimized_kernels
    if _optimized_kernels is not _OPTIMIZED_KERNELS_UNSET:
        return _optimized_kernels
    try:
        triton = import_module("triton")
        triton_language = import_module("triton.language")
        sgl_kernel = import_module("sgl_kernel")
        moe_impl = import_module(
            "voicehub.models.zonos2.source.zonos2.kernel.moe_impl"
        )
        triton_moe = import_module(
            "voicehub.models.zonos2.source.zonos2.kernel.triton.fused_moe"
        )
        _optimized_kernels = _OptimizedKernels(
            triton=triton,
            triton_language=triton_language,
            gelu_and_mul=sgl_kernel.gelu_and_mul,
            silu_and_mul=sgl_kernel.silu_and_mul,
            moe_align_block_size=sgl_kernel.moe_align_block_size,
            fused_moe_kernel=moe_impl.fused_moe_kernel_triton,
            moe_sum_reduce=triton_moe.moe_sum_reduce_triton,
        )
    except (AttributeError, ImportError, OSError, RuntimeError):
        _optimized_kernels = None
    return _optimized_kernels


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)


def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    kernels: _OptimizedKernels,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = kernels.triton.cdiv(
        max_num_tokens_padded,
        block_size,
    )
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)

    kernels.moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    is_marlin: bool,
) -> Dict[str, int]:

    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    # A heuristic: fused marlin works faster with this config for small M
    if M <= E or (is_marlin and M <= 32):
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
    is_marlin: bool = False,
):
    E, _, N = w2_shape

    config = get_default_config(M, E, N, w1_shape[2], top_k, is_marlin)
    return config


def _fused_experts_optimized(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    *,
    kernels: _OptimizedKernels,
):
    if no_combine:
        raise ValueError(
            "Uncombined expert outputs use the portable PyTorch path."
        )
    padded_size = 0
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    num_tokens, _ = hidden_states.shape
    E, N, _ = w1.shape

    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )

    compute_type = (
        kernels.triton_language.bfloat16
        if hidden_states.dtype == torch.bfloat16
        else kernels.triton_language.float16
    )
    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[: tokens_in_chunk * topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            curr_topk_ids,
            config["BLOCK_SIZE_M"],
            E,
            kernels=kernels,
        )

        kernels.fused_moe_kernel(
            curr_hidden_states,
            w1,
            intermediate_cache1,
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            config,
            compute_type=compute_type,
        )

        if activation == "silu":

            kernels.silu_and_mul(
                intermediate_cache1.view(-1, N),
                intermediate_cache2,
            )

        elif activation == "gelu":

            kernels.gelu_and_mul(
                intermediate_cache1.view(-1, N),
                intermediate_cache2,
            )

        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        kernels.fused_moe_kernel(
            intermediate_cache2,
            w2,
            (
                intermediate_cache3
                if topk_ids.shape[1] != 1
                else out_hidden_states[begin_chunk_idx:end_chunk_idx].unsqueeze(0)
            ),
            curr_topk_weights,
            curr_topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if topk_ids.shape[1] == 1:
            # GEMM2 writes the single route directly into the output rather
            # than intermediate_cache3.
            if routed_scaling_factor != 1.0:
                out_hidden_states[
                    begin_chunk_idx:end_chunk_idx
                ].mul_(routed_scaling_factor)
        elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
            torch.add(
                intermediate_cache3[:, 0],
                intermediate_cache3[:, 1],
                out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
            ).squeeze(dim=1)
        else:
            # According to micro benchmark results, torch.compile can get better performance for small token.
            if tokens_in_chunk <= 32:

                moe_sum_reduce_torch_compile(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )
            else:
                kernels.moe_sum_reduce(
                    intermediate_cache3,
                    out_hidden_states[begin_chunk_idx:end_chunk_idx],
                    routed_scaling_factor,
                )
    return out_hidden_states


def _validate_expert_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    inplace: bool,
    no_combine: bool,
) -> None:
    if hidden_states.ndim != 2:
        raise ValueError(
            "MoE hidden states must have shape [tokens, hidden_size]."
        )
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("MoE expert weights must be rank-three tensors.")
    if w1.shape[0] != w2.shape[0]:
        raise ValueError("MoE projections must contain the same expert count.")
    if hidden_states.shape[1] != w1.shape[2]:
        raise ValueError("MoE input and first-projection hidden sizes differ.")
    if w1.shape[1] % 2:
        raise ValueError(
            "MoE gated first projection must have an even output size."
        )
    if w2.shape[2] != w1.shape[1] // 2:
        raise ValueError(
            "MoE second-projection input must match the gated intermediate "
            "size."
        )
    if topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("MoE routes must have shape [tokens, top_k].")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("MoE route weights and expert IDs must have one shape.")
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError("MoE routes and hidden states must have one token count.")
    if topk_ids.shape[1] == 0:
        raise ValueError("MoE routing must select at least one expert.")
    if topk_ids.shape[1] > w1.shape[0]:
        raise ValueError(
            "MoE routing cannot select more routes than there are experts."
        )
    if activation not in {"silu", "gelu"}:
        raise ValueError(f"Unsupported activation: activation={activation!r}")
    if no_combine and inplace:
        raise ValueError("`no_combine=True` cannot be used in place.")
    if inplace and w2.shape[1] != hidden_states.shape[1]:
        raise ValueError(
            "In-place MoE requires its output and hidden sizes to match."
        )
    devices = {
        hidden_states.device,
        w1.device,
        w2.device,
        topk_weights.device,
        topk_ids.device,
    }
    if len(devices) != 1:
        raise ValueError("Every MoE tensor must be on the same device.")
    if hidden_states.dtype not in {
        torch.float32,
        torch.float16,
        torch.bfloat16,
    }:
        raise TypeError(
            "MoE hidden states must use float32, float16, or bfloat16."
        )
    if w1.dtype != hidden_states.dtype or w2.dtype != hidden_states.dtype:
        raise TypeError(
            "MoE projection weights must use the hidden-state dtype."
        )
    if not topk_weights.is_floating_point():
        raise TypeError("MoE route weights must use a floating-point dtype.")
    if topk_ids.dtype not in {
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("MoE expert IDs must use an integer dtype.")
    if topk_ids.numel():
        lowest = int(topk_ids.min().item())
        highest = int(topk_ids.max().item())
        if lowest < -1 or highest >= w1.shape[0]:
            raise ValueError(
                "MoE expert IDs must be -1 for padding or index an existing "
                "expert."
            )


def _activate_and_multiply(projected: torch.Tensor, activation: str):
    gate, values = projected.chunk(2, dim=-1)
    if activation == "silu":
        return F.silu(gate) * values
    return F.gelu(gate, approximate="none") * values


def _torch_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    activation: str,
    apply_router_weight_on_input: bool,
    no_combine: bool,
    routed_scaling_factor: Optional[float],
) -> torch.Tensor:
    """Portable expert dispatch with the same route weighting as the kernels."""
    token_count, top_k = topk_ids.shape
    output_size = w2.shape[1]
    routed = hidden_states.new_zeros(
        (token_count, top_k, output_size),
    )

    for expert_index in range(w1.shape[0]):
        token_indices, route_indices = torch.where(
            topk_ids == expert_index
        )
        if token_indices.numel() == 0:
            continue
        selected = hidden_states.index_select(0, token_indices)
        first_projection = F.linear(selected, w1[expert_index])
        route_weights = topk_weights[
            token_indices,
            route_indices,
        ].to(
            device=first_projection.device,
            dtype=first_projection.dtype,
        )
        if apply_router_weight_on_input:
            first_projection = first_projection * route_weights.unsqueeze(-1)
        activated = _activate_and_multiply(
            first_projection,
            activation,
        )
        expert_output = F.linear(activated, w2[expert_index])
        if not apply_router_weight_on_input:
            expert_output = expert_output * route_weights.unsqueeze(-1)
        routed[token_indices, route_indices] = expert_output

    scale = 1.0 if routed_scaling_factor is None else float(
        routed_scaling_factor
    )
    if no_combine:
        return routed
    return routed.sum(dim=1).mul(scale)


def _optimized_kernels_for(
    hidden_states: torch.Tensor,
) -> _OptimizedKernels | None:
    if hidden_states.device.type != "cuda" or not torch.version.cuda:
        return None
    return _load_optimized_kernels()


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):
    """Dispatch to optional CUDA kernels or a portable PyTorch implementation."""
    _validate_expert_inputs(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=activation,
        inplace=inplace,
        no_combine=no_combine,
    )
    kernels = _optimized_kernels_for(hidden_states)
    optimized_layout = (
        hidden_states.shape[0] > 0
        and hidden_states.is_contiguous()
        and w1.is_contiguous()
        and w2.is_contiguous()
        and topk_weights.is_contiguous()
        and topk_ids.is_contiguous()
        and w2.shape[1] == hidden_states.shape[1]
        and not bool((topk_ids < 0).any().item())
    )
    if kernels is not None and not no_combine and optimized_layout:
        return _fused_experts_optimized(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=inplace,
            activation=activation,
            apply_router_weight_on_input=(
                apply_router_weight_on_input
            ),
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
            kernels=kernels,
        )

    output = _torch_experts(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
    )
    if inplace:
        hidden_states.copy_(output)
        return hidden_states
    return output


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> None:

    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        True,
        activation,
        apply_router_weight_on_input,
        False,
        routed_scaling_factor,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
) -> torch.Tensor:
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        False,
        activation,
        apply_router_weight_on_input,
        no_combine=no_combine,
        routed_scaling_factor=routed_scaling_factor,
    )


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):

    if inplace:
        assert not no_combine, "no combine + inplace makes no sense"
        inplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            routed_scaling_factor,
        )
        return hidden_states
    else:
        return outplace_fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input,
            no_combine=no_combine,
            routed_scaling_factor=routed_scaling_factor,
        )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    inplace: bool = False,
    activation: str = "silu",
    no_combine: bool = False,
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:

    topk_weights, topk_ids = select_experts(
        hidden_states=hidden_states,
        router_logits=gating_output,
        top_k=topk,
        renormalize=renormalize,
    )
    return fused_experts(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        no_combine=no_combine,
    )
