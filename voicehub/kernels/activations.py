"""Fused activation kernels shared by native TTS architecture families."""

from __future__ import annotations

import sys
from functools import cache
from importlib import import_module
from pathlib import Path
from threading import RLock

import torch
from torch.nn import functional as F

from voicehub.kernel_operations import (
    AUDIO_CODEC_SNAKE,
    AUDIO_CODEC_SNAKE_BETA,
    DIFFUSION_FUSED_BIAS_GELU,
    DIFFUSION_FUSED_MODULATE,
    LLM_GATED_SILU,
    VITS_FUSED_ADD_TANH_SIGMOID,
    VITS_TANH_SIGMOID_GATE,
)
from voicehub.kernels.capabilities import triton_capability
from voicehub.kernels.cuda_extensions import (
    CudaExtensionSpec,
    LoadedCudaExtension,
    load_cuda_extension,
    register_cuda_extension,
)
from voicehub.kernels.registry import KernelBackend, KernelSupport, dispatch_kernel, register_kernel

ACTIVATION_CUDA_EXTENSION_NAME = "voicehub_kernels_activations"

_TRITON_DTYPES = frozenset({
    torch.float16,
    torch.bfloat16,
    torch.float32,
})
_CSRC_ROOT = Path(__file__).with_name("csrc")
_ACTIVATION_CUDA_SPEC = CudaExtensionSpec(
    name=ACTIVATION_CUDA_EXTENSION_NAME,
    sources=(
        _CSRC_ROOT / "activations.cpp",
        _CSRC_ROOT / "activations.cu",
    ),
    operators=(
        "voicehub_kernels::gated_silu",
        "voicehub_kernels::tanh_sigmoid_gate",
        "voicehub_kernels::fused_add_tanh_sigmoid",
        "voicehub_kernels::fused_bias_gelu",
        "voicehub_kernels::fused_modulate",
        "voicehub_kernels::codec_snake",
        "voicehub_kernels::codec_snake_beta",
    ),
    extra_cflags=("-O3", ),
    extra_cuda_cflags=("-O3", "--use_fast_math"),
)
_CUDA_REGISTRATION_LIBRARY: torch.library.Library | None = None
_CUDA_REGISTRATION_LOCK = RLock()


@cache
def _cached_triton_capability(device: str):
    """Avoid repeating CUDA/package discovery in every eager FFN block."""
    return triton_capability(torch.device(device))


def _validate_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    operation: str,
    left_name: str,
    right_name: str,
) -> None:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError(f"`{left_name}` and `{right_name}` must be torch.Tensor values.")
    if left.shape != right.shape:
        raise ValueError(f"{operation} expects tensors with identical shapes.")
    if left.device != right.device:
        raise ValueError(f"{operation} expects tensors on the same device.")
    if left.dtype != right.dtype:
        raise ValueError(f"{operation} expects tensors with the same dtype.")
    if not left.is_floating_point():
        raise TypeError(f"{operation} expects floating-point tensors.")


def _validate_bias_gelu(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> None:
    if not isinstance(inputs, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("`inputs` and `bias` must be torch.Tensor values.")
    if inputs.ndim < 1 or bias.ndim != 1:
        raise ValueError("fused_bias_gelu expects inputs[..., hidden] and bias[hidden].")
    if inputs.shape[-1] != bias.shape[0]:
        raise ValueError("fused_bias_gelu bias size must match the input's last dimension.")
    if inputs.device != bias.device:
        raise ValueError("fused_bias_gelu expects tensors on the same device.")
    if inputs.dtype != bias.dtype:
        raise ValueError("fused_bias_gelu expects tensors with the same dtype.")
    if not inputs.is_floating_point():
        raise TypeError("fused_bias_gelu expects floating-point tensors.")


def _validate_fused_modulate(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    values = (hidden_states, shift, scale)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("`hidden_states`, `shift`, and `scale` must be torch.Tensor values.")
    if hidden_states.ndim < 1:
        raise ValueError("fused_modulate expects at least one tensor dimension.")
    if hidden_states.device != shift.device or hidden_states.device != scale.device:
        raise ValueError("fused_modulate expects tensors on the same device.")
    if hidden_states.dtype != shift.dtype or hidden_states.dtype != scale.dtype:
        raise ValueError("fused_modulate expects tensors with the same dtype.")
    if not hidden_states.is_floating_point():
        raise TypeError("fused_modulate expects floating-point tensors.")
    try:
        output_shape = torch.broadcast_shapes(
            hidden_states.shape,
            shift.shape,
            scale.shape,
        )
    except RuntimeError as error:
        raise ValueError("fused_modulate expects shift and scale to broadcast to hidden_states.") from error
    if output_shape != hidden_states.shape:
        raise ValueError("fused_modulate cannot expand the hidden-state output shape.")


def _validate_codec_snake(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> None:
    if not isinstance(inputs, torch.Tensor) or not isinstance(alpha, torch.Tensor):
        raise TypeError("`inputs` and `alpha` must be torch.Tensor values.")
    if inputs.ndim < 2:
        raise ValueError("codec_snake expects inputs with shape [batch, channels, ...].")
    if alpha.numel() != inputs.shape[1]:
        raise ValueError("codec_snake expects one alpha value per input channel.")
    if inputs.device != alpha.device:
        raise ValueError("codec_snake expects tensors on the same device.")
    if inputs.dtype != alpha.dtype:
        raise ValueError("codec_snake expects tensors with the same dtype.")
    if not inputs.is_floating_point():
        raise TypeError("codec_snake expects floating-point tensors.")


def _validate_codec_snake_beta(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    values = (inputs, alpha, beta)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise TypeError("`inputs`, `alpha`, and `beta` must be torch.Tensor values.")
    if inputs.ndim < 2:
        raise ValueError("codec_snake_beta expects inputs with shape [batch, channels, ...].")
    if alpha.numel() != inputs.shape[1] or beta.numel() != inputs.shape[1]:
        raise ValueError("codec_snake_beta expects one alpha and beta value per input channel.")
    if inputs.device != alpha.device or inputs.device != beta.device:
        raise ValueError("codec_snake_beta expects tensors on the same device.")
    if inputs.dtype != alpha.dtype or inputs.dtype != beta.dtype:
        raise ValueError("codec_snake_beta expects tensors with the same dtype.")
    if not inputs.is_floating_point():
        raise TypeError("codec_snake_beta expects floating-point tensors.")


def _validate_vits_fused_gate(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> None:
    if not isinstance(input_a, torch.Tensor) or not isinstance(input_b, torch.Tensor):
        raise TypeError("`input_a` and `input_b` must be torch.Tensor values.")
    if input_a.ndim != 3 or input_b.ndim != 3:
        raise ValueError("fused_add_tanh_sigmoid expects [batch, 2 * channels, frames] tensors.")
    if input_a.device != input_b.device:
        raise ValueError("fused_add_tanh_sigmoid expects tensors on the same device.")
    if input_a.dtype != input_b.dtype:
        raise ValueError("fused_add_tanh_sigmoid expects tensors with the same dtype.")
    if not input_a.is_floating_point():
        raise TypeError("fused_add_tanh_sigmoid expects floating-point tensors.")
    if isinstance(channels, bool) or not isinstance(channels, int):
        raise TypeError("`channels` must be an integer.")
    if (channels < 1 or input_a.shape[1] != 2 * channels or input_b.shape[1] != 2 * channels):
        raise ValueError("fused_add_tanh_sigmoid expects input channel size to equal "
                         "2 * `channels`.")
    for dimension, name in ((0, "batch"), (2, "frame")):
        left = input_a.shape[dimension]
        right = input_b.shape[dimension]
        if left != right and left != 1 and right != 1:
            raise ValueError(
                "fused_add_tanh_sigmoid input "
                f"{name} dimensions must be equal or broadcastable.")


def _vits_fused_gate_contract_supported(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> bool:
    return bool(
        isinstance(input_a, torch.Tensor) and isinstance(input_b, torch.Tensor) and input_a.ndim == 3 and
        input_b.ndim == 3 and not isinstance(channels, bool) and isinstance(channels, int) and
        channels > 0 and input_a.shape[1] == 2 * channels and input_b.shape[1] == 2 * channels and
        (input_a.shape[0] == input_b.shape[0] or input_a.shape[0] == 1 or input_b.shape[0] == 1) and
        (input_a.shape[2] == input_b.shape[2] or input_a.shape[2] == 1 or input_b.shape[2] == 1))


def _pair_triton_support(
    left: torch.Tensor,
    right: torch.Tensor,
) -> KernelSupport:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return KernelSupport(False, "arguments are not tensors")
    if left.device.type != "cuda" or right.device != left.device:
        return KernelSupport(False, "Triton activation kernels require one CUDA device")
    if left.dtype not in _TRITON_DTYPES or right.dtype != left.dtype:
        return KernelSupport(
            False,
            "Triton activation kernels require matching float16, bfloat16, or float32 tensors",
        )
    if "voicehub.kernels.triton_activations" in sys.modules:
        return KernelSupport(True)
    capability = _cached_triton_capability(str(left.device))
    return KernelSupport(capability.available, capability.reason)


def _bias_gelu_triton_support(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> KernelSupport:
    support = _pair_triton_support(inputs, bias)
    if not support.available:
        return support
    if inputs.ndim < 1 or bias.ndim != 1 or inputs.shape[-1] != bias.shape[0]:
        return KernelSupport(False, "bias must match the input's last dimension")
    return support


def _fused_modulate_contract_supported(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> bool:
    if not all(isinstance(value, torch.Tensor) for value in (hidden_states, shift, scale)):
        return False
    # Accelerated implementations specialize the vector, matrix, and
    # [batch, time, hidden] contracts used by diffusion TTS. Higher-rank
    # tensors retain the exact PyTorch fallback instead of materializing
    # expanded shift/scale tensors.
    if not 1 <= hidden_states.ndim <= 3:
        return False
    try:
        return torch.broadcast_shapes(
            hidden_states.shape,
            shift.shape,
            scale.shape,
        ) == hidden_states.shape
    except RuntimeError:
        return False


def _codec_snake_contract_supported(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> bool:
    return bool(
        isinstance(inputs, torch.Tensor) and isinstance(alpha, torch.Tensor) and inputs.ndim >= 2 and
        alpha.numel() == inputs.shape[1])


def _codec_snake_beta_contract_supported(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> bool:
    return bool(
        isinstance(inputs, torch.Tensor) and isinstance(alpha, torch.Tensor) and
        isinstance(beta, torch.Tensor) and inputs.ndim >= 2 and alpha.numel() == inputs.shape[1] and
        beta.numel() == inputs.shape[1])


def _triton_tensor_group_support(
    tensors: tuple[torch.Tensor, ...],
    *,
    operation: str,
) -> KernelSupport:
    if not tensors or any(not isinstance(value, torch.Tensor) for value in tensors):
        return KernelSupport(False, f"{operation} arguments are not tensors")
    first = tensors[0]
    if first.device.type != "cuda" or any(value.device != first.device for value in tensors[1:]):
        return KernelSupport(
            False,
            f"Triton {operation} kernels require one CUDA device",
        )
    if first.dtype not in _TRITON_DTYPES or any(value.dtype != first.dtype for value in tensors[1:]):
        return KernelSupport(
            False,
            f"Triton {operation} kernels require matching float16, "
            "bfloat16, or float32 tensors",
        )
    if "voicehub.kernels.triton_activations" in sys.modules:
        return KernelSupport(True)
    capability = _cached_triton_capability(str(first.device))
    return KernelSupport(capability.available, capability.reason)


def _fused_modulate_triton_support(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> KernelSupport:
    if not _fused_modulate_contract_supported(hidden_states, shift, scale):
        return KernelSupport(
            False,
            "shift and scale must broadcast to the hidden-state shape",
        )
    return _triton_tensor_group_support(
        (hidden_states, shift, scale),
        operation="fused modulation",
    )


def _codec_snake_triton_support(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> KernelSupport:
    if not _codec_snake_contract_supported(inputs, alpha):
        return KernelSupport(
            False,
            "inputs must have [batch, channels, ...] shape and one alpha per channel",
        )
    return _triton_tensor_group_support(
        (inputs, alpha),
        operation="codec Snake",
    )


def _codec_snake_beta_triton_support(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> KernelSupport:
    if not _codec_snake_beta_contract_supported(inputs, alpha, beta):
        return KernelSupport(
            False,
            "inputs must have [batch, channels, ...] shape and one alpha/beta pair per channel",
        )
    return _triton_tensor_group_support(
        (inputs, alpha, beta),
        operation="codec SnakeBeta",
    )


def _vits_fused_gate_triton_support(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> KernelSupport:
    if not _vits_fused_gate_contract_supported(input_a, input_b, channels):
        return KernelSupport(
            False,
            "inputs must have broadcastable [batch, 2 * channels, frames] shapes",
        )
    if input_a.device.type != "cuda" or input_b.device != input_a.device:
        return KernelSupport(False, "Triton activation kernels require one CUDA device")
    if input_a.dtype not in _TRITON_DTYPES or input_b.dtype != input_a.dtype:
        return KernelSupport(
            False,
            "Triton activation kernels require matching float16, bfloat16, or float32 tensors",
        )
    if "voicehub.kernels.triton_activations" in sys.modules:
        return KernelSupport(True)
    capability = _cached_triton_capability(str(input_a.device))
    return KernelSupport(capability.available, capability.reason)


def _pair_cuda_extension_support(
    left: torch.Tensor,
    right: torch.Tensor,
) -> KernelSupport:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return KernelSupport(False, "arguments are not tensors")
    if left.device.type != "cuda" or right.device != left.device:
        return KernelSupport(False, "the CUDA extension requires one CUDA device")
    if left.dtype not in _TRITON_DTYPES or right.dtype != left.dtype:
        return KernelSupport(
            False,
            "the CUDA extension requires matching float16, bfloat16, or float32 tensors",
        )
    return KernelSupport(True)


def _bias_gelu_cuda_extension_support(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> KernelSupport:
    support = _pair_cuda_extension_support(inputs, bias)
    if not support.available:
        return support
    if inputs.ndim < 1 or bias.ndim != 1 or inputs.shape[-1] != bias.shape[0]:
        return KernelSupport(False, "bias must match the input's last dimension")
    return support


def _cuda_tensor_group_support(
    tensors: tuple[torch.Tensor, ...],
    *,
    operation: str,
) -> KernelSupport:
    if not tensors or any(not isinstance(value, torch.Tensor) for value in tensors):
        return KernelSupport(False, f"{operation} arguments are not tensors")
    first = tensors[0]
    if first.device.type != "cuda" or any(value.device != first.device for value in tensors[1:]):
        return KernelSupport(
            False,
            f"the CUDA {operation} extension requires one CUDA device",
        )
    if first.dtype not in _TRITON_DTYPES or any(value.dtype != first.dtype for value in tensors[1:]):
        return KernelSupport(
            False,
            f"the CUDA {operation} extension requires matching float16, "
            "bfloat16, or float32 tensors",
        )
    return KernelSupport(True)


def _fused_modulate_cuda_extension_support(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> KernelSupport:
    if not _fused_modulate_contract_supported(hidden_states, shift, scale):
        return KernelSupport(
            False,
            "shift and scale must broadcast to the hidden-state shape",
        )
    return _cuda_tensor_group_support(
        (hidden_states, shift, scale),
        operation="fused modulation",
    )


def _codec_snake_cuda_extension_support(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> KernelSupport:
    if not _codec_snake_contract_supported(inputs, alpha):
        return KernelSupport(
            False,
            "inputs must have [batch, channels, ...] shape and one alpha per channel",
        )
    return _cuda_tensor_group_support(
        (inputs, alpha),
        operation="codec Snake",
    )


def _codec_snake_beta_cuda_extension_support(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> KernelSupport:
    if not _codec_snake_beta_contract_supported(inputs, alpha, beta):
        return KernelSupport(
            False,
            "inputs must have [batch, channels, ...] shape and one alpha/beta pair per channel",
        )
    return _cuda_tensor_group_support(
        (inputs, alpha, beta),
        operation="codec SnakeBeta",
    )


def _vits_fused_gate_cuda_extension_support(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> KernelSupport:
    if not _vits_fused_gate_contract_supported(input_a, input_b, channels):
        return KernelSupport(
            False,
            "inputs must have broadcastable [batch, 2 * channels, frames] shapes",
        )
    if input_a.device.type != "cuda" or input_b.device != input_a.device:
        return KernelSupport(False, "the CUDA extension requires one CUDA device")
    if input_a.dtype not in _TRITON_DTYPES or input_b.dtype != input_a.dtype:
        return KernelSupport(
            False,
            "the CUDA extension requires matching float16, bfloat16, or float32 tensors",
        )
    return KernelSupport(True)


def gated_silu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for Qwen/Conversation-style SwiGLU gating."""
    return F.silu(gate) * up


def tanh_sigmoid_gate_reference(
    activation: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for the VITS/WaveNet gated activation unit."""
    return torch.tanh(activation) * torch.sigmoid(gate)


def fused_add_tanh_sigmoid_reference(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    """Reference for the complete VITS/WaveNet gated activation.

    Keeping the addition and channel split inside the logical operation
    lets accelerator implementations avoid two materialized channel
    views and the intermediate ``input_a + input_b`` tensor.
    """
    combined = input_a + input_b
    activation = combined[:, :channels, :]
    gate = combined[:, channels:, :]
    return torch.tanh(activation) * torch.sigmoid(gate)


def fused_bias_gelu_reference(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for F5 feed-forward bias + approximate GELU."""
    return F.gelu(inputs + bias, approximate="tanh")


def fused_modulate_reference(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference for DiT adaptive-normalization modulation."""
    return hidden_states * (1.0 + scale) + shift


def codec_snake_reference(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference for DAC/SNAC periodic Snake activation."""
    shape = inputs.shape
    flattened = inputs.reshape(shape[0], shape[1], -1)
    channel_alpha = alpha.reshape(1, shape[1], 1)
    output = (flattened + (channel_alpha + 1e-9).reciprocal() * torch.sin(channel_alpha * flattened).square())
    return output.reshape(shape)


def codec_snake_beta_reference(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Exact PyTorch reference for independent-frequency SnakeBeta."""
    shape = inputs.shape
    flattened = inputs.reshape(shape[0], shape[1], -1)
    channel_alpha = alpha.reshape(1, shape[1], 1)
    channel_beta = beta.reshape(1, shape[1], 1)
    output = (flattened + (channel_beta + 1e-9).reciprocal() * torch.sin(channel_alpha * flattened).square())
    return output.reshape(shape)


def _gated_silu_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.gated_silu_triton(gate, up)


def _tanh_sigmoid_gate_triton(
    activation: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.tanh_sigmoid_gate_triton(activation, gate)


def _fused_add_tanh_sigmoid_triton(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.fused_add_tanh_sigmoid_triton(input_a, input_b, channels)


def _fused_bias_gelu_triton(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.fused_bias_gelu_triton(inputs, bias)


def _fused_modulate_triton(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.fused_modulate_triton(hidden_states, shift, scale)


def _codec_snake_triton(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.codec_snake_triton(inputs, alpha)


def _codec_snake_beta_triton(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.codec_snake_beta_triton(inputs, alpha, beta)


def _gated_silu_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.gated_silu(gate, up)


def _tanh_sigmoid_gate_cuda(
    activation: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.tanh_sigmoid_gate(activation, gate)


def _fused_add_tanh_sigmoid_cuda(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.fused_add_tanh_sigmoid(
        input_a,
        input_b,
        channels,
    )


def _fused_bias_gelu_cuda(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.fused_bias_gelu(inputs, bias)


def _fused_modulate_cuda(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.fused_modulate(
        hidden_states,
        shift,
        scale,
    )


def _codec_snake_cuda(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.codec_snake(inputs, alpha)


def _codec_snake_beta_cuda(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.codec_snake_beta(
        inputs,
        alpha,
        beta,
    )


def _register_builtin_kernels() -> None:
    register_cuda_extension(_ACTIVATION_CUDA_SPEC)
    for operation, reference, triton_implementation, support_check in (
        (
            LLM_GATED_SILU,
            gated_silu_reference,
            _gated_silu_triton,
            _pair_triton_support,
        ),
        (
            VITS_TANH_SIGMOID_GATE,
            tanh_sigmoid_gate_reference,
            _tanh_sigmoid_gate_triton,
            _pair_triton_support,
        ),
        (
            DIFFUSION_FUSED_BIAS_GELU,
            fused_bias_gelu_reference,
            _fused_bias_gelu_triton,
            _bias_gelu_triton_support,
        ),
        (
            DIFFUSION_FUSED_MODULATE,
            fused_modulate_reference,
            _fused_modulate_triton,
            _fused_modulate_triton_support,
        ),
        (
            AUDIO_CODEC_SNAKE,
            codec_snake_reference,
            _codec_snake_triton,
            _codec_snake_triton_support,
        ),
        (
            AUDIO_CODEC_SNAKE_BETA,
            codec_snake_beta_reference,
            _codec_snake_beta_triton,
            _codec_snake_beta_triton_support,
        ),
    ):
        register_kernel(
            operation,
            KernelBackend.TORCH,
            reference,
            priority=0,
            replace=True,
        )
        register_kernel(
            operation,
            KernelBackend.TRITON,
            triton_implementation,
            priority=200,
            support_check=support_check,
            replace=True,
        )
    register_kernel(
        VITS_FUSED_ADD_TANH_SIGMOID,
        KernelBackend.TORCH,
        fused_add_tanh_sigmoid_reference,
        priority=0,
        replace=True,
    )
    register_kernel(
        VITS_FUSED_ADD_TANH_SIGMOID,
        KernelBackend.TRITON,
        _fused_add_tanh_sigmoid_triton,
        priority=200,
        support_check=_vits_fused_gate_triton_support,
        replace=True,
    )


def _register_cuda_kernels() -> None:
    for operation, implementation, support_check in (
        (
            LLM_GATED_SILU,
            _gated_silu_cuda,
            _pair_cuda_extension_support,
        ),
        (
            VITS_TANH_SIGMOID_GATE,
            _tanh_sigmoid_gate_cuda,
            _pair_cuda_extension_support,
        ),
        (
            VITS_FUSED_ADD_TANH_SIGMOID,
            _fused_add_tanh_sigmoid_cuda,
            _vits_fused_gate_cuda_extension_support,
        ),
        (
            DIFFUSION_FUSED_BIAS_GELU,
            _fused_bias_gelu_cuda,
            _bias_gelu_cuda_extension_support,
        ),
        (
            DIFFUSION_FUSED_MODULATE,
            _fused_modulate_cuda,
            _fused_modulate_cuda_extension_support,
        ),
        (
            AUDIO_CODEC_SNAKE,
            _codec_snake_cuda,
            _codec_snake_cuda_extension_support,
        ),
        (
            AUDIO_CODEC_SNAKE_BETA,
            _codec_snake_beta_cuda,
            _codec_snake_beta_cuda_extension_support,
        ),
    ):
        register_kernel(
            operation,
            KernelBackend.CUDA_EXTENSION,
            implementation,
            priority=300,
            support_check=support_check,
            replace=True,
        )


def _register_cuda_autograd() -> None:
    """Attach Python dispatcher registrations exactly once per process."""
    with _CUDA_REGISTRATION_LOCK:
        _register_cuda_autograd_locked()


def _register_cuda_autograd_locked() -> None:
    """Attach fake-tensor and autograd formulas after schemas are loaded."""
    global _CUDA_REGISTRATION_LIBRARY
    if _CUDA_REGISTRATION_LIBRARY is not None:
        return

    def check_same_tensor_contract(left, right, operation: str) -> None:
        torch._check(
            left.ndim == right.ndim,
            lambda: f"{operation} expects tensors with identical ranks",
        )
        for dimension in range(left.ndim):
            torch._check(
                left.shape[dimension] == right.shape[dimension],
                lambda: f"{operation} expects tensors with identical shapes",
            )
        torch._check(
            left.device == right.device,
            lambda: f"{operation} expects tensors on the same device",
        )
        torch._check(
            left.dtype == right.dtype,
            lambda: f"{operation} expects tensors with the same dtype",
        )
        torch._check(
            left.dtype in _TRITON_DTYPES,
            lambda: (f"{operation} supports float16, bfloat16, and float32 tensors"),
        )

    def pair_setup_context(ctx, inputs, output) -> None:
        del output
        ctx.save_for_backward(*inputs)

    def gated_silu_backward(ctx, gradient):
        gate, up = ctx.saved_tensors
        sigmoid = torch.sigmoid(gate)
        silu = gate * sigmoid
        silu_derivative = sigmoid + gate * sigmoid * (1.0 - sigmoid)
        return gradient * up * silu_derivative, gradient * silu

    def tanh_sigmoid_backward(ctx, gradient):
        activation, gate = ctx.saved_tensors
        tanh_activation = torch.tanh(activation)
        sigmoid_gate = torch.sigmoid(gate)
        return (
            gradient * sigmoid_gate * (1.0 - tanh_activation.square()),
            gradient * tanh_activation * sigmoid_gate * (1.0 - sigmoid_gate),
        )

    def fused_add_tanh_sigmoid_setup_context(ctx, inputs, output) -> None:
        del output
        input_a, input_b, channels = inputs
        ctx.channels = channels
        ctx.save_for_backward(input_a, input_b)

    def fused_add_tanh_sigmoid_backward(ctx, gradient):
        input_a, input_b = ctx.saved_tensors
        channels = ctx.channels
        combined = input_a + input_b
        activation = combined[:, :channels, :]
        gate = combined[:, channels:, :]
        tanh_activation = torch.tanh(activation)
        sigmoid_gate = torch.sigmoid(gate)
        activation_gradient = (gradient * sigmoid_gate * (1.0 - tanh_activation.square()))
        gate_gradient = (gradient * tanh_activation * sigmoid_gate * (1.0 - sigmoid_gate))
        combined_gradient = torch.cat(
            (activation_gradient, gate_gradient),
            dim=1,
        )
        return (
            combined_gradient.sum_to_size(input_a.shape),
            combined_gradient.sum_to_size(input_b.shape),
            None,
        )

    def fused_bias_gelu_backward(ctx, gradient):
        inputs, bias = ctx.saved_tensors
        value = inputs + bias
        value_squared = value.square()
        coefficient = 0.7978845608028654
        cubic = 0.044715
        inner = coefficient * (value + cubic * value * value_squared)
        tanh_inner = torch.tanh(inner)
        inner_derivative = coefficient * (1.0 + 3.0 * cubic * value_squared)
        derivative = (0.5 * (1.0 + tanh_inner) + 0.5 * value * (1.0 - tanh_inner.square()) * inner_derivative)
        input_gradient = gradient * derivative
        reduction_dimensions = tuple(range(input_gradient.ndim - 1))
        bias_gradient = (
            input_gradient if not reduction_dimensions else input_gradient.sum(dim=reduction_dimensions))
        return input_gradient, bias_gradient

    def fused_modulate_backward(ctx, gradient):
        hidden_states, shift, scale = ctx.saved_tensors
        return (
            gradient * (1.0 + scale),
            gradient.sum_to_size(shift.shape),
            (gradient * hidden_states).sum_to_size(scale.shape),
        )

    def codec_snake_backward(ctx, gradient):
        inputs, alpha = ctx.saved_tensors
        shape = inputs.shape
        flattened = inputs.reshape(shape[0], shape[1], -1)
        channel_alpha = alpha.reshape(1, shape[1], 1)
        denominator = channel_alpha + 1e-9
        angles = channel_alpha * flattened
        sine = torch.sin(angles)
        double_sine = torch.sin(2.0 * angles)
        input_derivative = 1.0 + (channel_alpha / denominator) * double_sine
        alpha_derivative = (flattened * double_sine / denominator - sine.square() / denominator.square())
        flat_gradient = gradient.reshape_as(flattened)
        return (
            (flat_gradient * input_derivative).reshape(shape),
            (flat_gradient * alpha_derivative).sum(dim=(0, 2), ).reshape(alpha.shape),
        )

    def codec_snake_beta_backward(ctx, gradient):
        inputs, alpha, beta = ctx.saved_tensors
        shape = inputs.shape
        flattened = inputs.reshape(shape[0], shape[1], -1)
        channel_alpha = alpha.reshape(1, shape[1], 1)
        channel_beta = beta.reshape(1, shape[1], 1)
        denominator = channel_beta + 1e-9
        angles = channel_alpha * flattened
        sine = torch.sin(angles)
        double_sine = torch.sin(2.0 * angles)
        flat_gradient = gradient.reshape_as(flattened)
        input_gradient = (flat_gradient * (1.0 + channel_alpha * double_sine / denominator))
        alpha_gradient = (flat_gradient * flattened * double_sine / denominator)
        beta_gradient = (-flat_gradient * sine.square() / denominator.square())
        return (
            input_gradient.reshape(shape),
            alpha_gradient.sum(dim=(0, 2)).reshape(alpha.shape),
            beta_gradient.sum(dim=(0, 2)).reshape(beta.shape),
        )

    def fake_gated_silu(gate, up):
        check_same_tensor_contract(
            gate,
            up,
            "voicehub_kernels::gated_silu",
        )
        return torch.empty_like(
            gate,
            memory_format=torch.contiguous_format,
        )

    def fake_tanh_sigmoid_gate(activation, gate):
        check_same_tensor_contract(
            activation,
            gate,
            "voicehub_kernels::tanh_sigmoid_gate",
        )
        return torch.empty_like(
            activation,
            memory_format=torch.contiguous_format,
        )

    def fake_fused_add_tanh_sigmoid(input_a, input_b, channels):
        torch._check(
            input_a.device == input_b.device,
            lambda: ("voicehub_kernels::fused_add_tanh_sigmoid expects tensors "
                     "on the same device"),
        )
        torch._check(
            input_a.dtype == input_b.dtype,
            lambda: ("voicehub_kernels::fused_add_tanh_sigmoid expects tensors "
                     "with the same dtype"),
        )
        torch._check(
            input_a.ndim == 3 and input_b.ndim == 3,
            lambda: ("voicehub_kernels::fused_add_tanh_sigmoid expects "
                     "[batch, 2 * channels, frames]"),
        )
        torch._check(
            channels > 0 and input_a.shape[1] == 2 * channels and input_b.shape[1] == 2 * channels,
            lambda:
            ("voicehub_kernels::fused_add_tanh_sigmoid input channel "
             "size must equal 2 * channels"),
        )
        torch._check(
            input_a.shape[0] == input_b.shape[0] or input_a.shape[0] == 1 or input_b.shape[0] == 1,
            lambda: ("voicehub_kernels::fused_add_tanh_sigmoid batch dimensions "
                     "must be broadcastable"),
        )
        torch._check(
            input_a.shape[2] == input_b.shape[2] or input_a.shape[2] == 1 or input_b.shape[2] == 1,
            lambda: ("voicehub_kernels::fused_add_tanh_sigmoid frame dimensions "
                     "must be broadcastable"),
        )
        return torch.empty(
            (
                (input_b.shape[0] if input_a.shape[0] == 1 else input_a.shape[0]),
                channels,
                (input_b.shape[2] if input_a.shape[2] == 1 else input_a.shape[2]),
            ),
            device=input_a.device,
            dtype=input_a.dtype,
        )

    def fake_fused_bias_gelu(inputs, bias):
        torch._check(
            inputs.ndim >= 1,
            lambda: "voicehub_kernels::fused_bias_gelu expects input[..., hidden]",
        )
        torch._check(
            bias.ndim == 1,
            lambda: "voicehub_kernels::fused_bias_gelu expects bias[hidden]",
        )
        torch._check(
            inputs.shape[-1] == bias.shape[0],
            lambda: ("voicehub_kernels::fused_bias_gelu bias size must match "
                     "the input's last dimension"),
        )
        torch._check(
            inputs.device == bias.device,
            lambda: ("voicehub_kernels::fused_bias_gelu expects tensors on "
                     "the same device"),
        )
        torch._check(
            inputs.dtype == bias.dtype,
            lambda: ("voicehub_kernels::fused_bias_gelu expects tensors with "
                     "the same dtype"),
        )
        torch._check(
            inputs.dtype in _TRITON_DTYPES,
            lambda: ("voicehub_kernels::fused_bias_gelu supports float16, "
                     "bfloat16, and float32 tensors"),
        )
        return torch.empty_like(
            inputs,
            memory_format=torch.contiguous_format,
        )

    def fake_fused_modulate(hidden_states, shift, scale):
        torch._check(
            hidden_states.device == shift.device and hidden_states.device == scale.device,
            lambda: ("voicehub_kernels::fused_modulate expects tensors on "
                     "the same device"),
        )
        torch._check(
            hidden_states.dtype == shift.dtype and hidden_states.dtype == scale.dtype,
            lambda: ("voicehub_kernels::fused_modulate expects tensors with "
                     "the same dtype"),
        )
        torch._check(
            hidden_states.ndim >= 1,
            lambda: ("voicehub_kernels::fused_modulate expects at least one "
                     "dimension"),
        )
        output_shape = torch.broadcast_shapes(
            hidden_states.shape,
            shift.shape,
            scale.shape,
        )
        torch._check(
            output_shape == hidden_states.shape,
            lambda: ("voicehub_kernels::fused_modulate shift and scale must "
                     "broadcast to hidden_states"),
        )
        return torch.empty_like(
            hidden_states,
            memory_format=torch.contiguous_format,
        )

    def fake_codec_snake(inputs, alpha):
        torch._check(
            inputs.ndim >= 2,
            lambda: ("voicehub_kernels::codec_snake expects "
                     "[batch, channels, ...] inputs"),
        )
        torch._check(
            alpha.numel() == inputs.shape[1],
            lambda: ("voicehub_kernels::codec_snake expects one alpha per "
                     "channel"),
        )
        torch._check(
            inputs.device == alpha.device,
            lambda: ("voicehub_kernels::codec_snake expects tensors on the "
                     "same device"),
        )
        torch._check(
            inputs.dtype == alpha.dtype,
            lambda: ("voicehub_kernels::codec_snake expects tensors with "
                     "the same dtype"),
        )
        return torch.empty_like(
            inputs,
            memory_format=torch.contiguous_format,
        )

    def fake_codec_snake_beta(inputs, alpha, beta):
        torch._check(
            inputs.ndim >= 2,
            lambda: ("voicehub_kernels::codec_snake_beta expects "
                     "[batch, channels, ...] inputs"),
        )
        torch._check(
            alpha.numel() == inputs.shape[1] and beta.numel() == inputs.shape[1],
            lambda: ("voicehub_kernels::codec_snake_beta expects one "
                     "alpha/beta pair per channel"),
        )
        torch._check(
            inputs.device == alpha.device and inputs.device == beta.device,
            lambda: ("voicehub_kernels::codec_snake_beta expects tensors "
                     "on the same device"),
        )
        torch._check(
            inputs.dtype == alpha.dtype and inputs.dtype == beta.dtype,
            lambda: ("voicehub_kernels::codec_snake_beta expects tensors "
                     "with the same dtype"),
        )
        return torch.empty_like(
            inputs,
            memory_format=torch.contiguous_format,
        )

    registration_library = torch.library.Library(
        "voicehub_kernels",
        "FRAGMENT",
    )
    torch.library.register_fake(
        "voicehub_kernels::gated_silu",
        fake_gated_silu,
        lib=registration_library,
    )
    torch.library.register_fake(
        "voicehub_kernels::tanh_sigmoid_gate",
        fake_tanh_sigmoid_gate,
        lib=registration_library,
    )
    torch.library.register_fake(
        "voicehub_kernels::fused_add_tanh_sigmoid",
        fake_fused_add_tanh_sigmoid,
        lib=registration_library,
    )
    torch.library.register_fake(
        "voicehub_kernels::fused_bias_gelu",
        fake_fused_bias_gelu,
        lib=registration_library,
    )
    torch.library.register_fake(
        "voicehub_kernels::fused_modulate",
        fake_fused_modulate,
        lib=registration_library,
    )
    torch.library.register_fake(
        "voicehub_kernels::codec_snake",
        fake_codec_snake,
        lib=registration_library,
    )
    torch.library.register_fake(
        "voicehub_kernels::codec_snake_beta",
        fake_codec_snake_beta,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::gated_silu",
        gated_silu_backward,
        setup_context=pair_setup_context,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::tanh_sigmoid_gate",
        tanh_sigmoid_backward,
        setup_context=pair_setup_context,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::fused_add_tanh_sigmoid",
        fused_add_tanh_sigmoid_backward,
        setup_context=fused_add_tanh_sigmoid_setup_context,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::fused_bias_gelu",
        fused_bias_gelu_backward,
        setup_context=pair_setup_context,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::fused_modulate",
        fused_modulate_backward,
        setup_context=pair_setup_context,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::codec_snake",
        codec_snake_backward,
        setup_context=pair_setup_context,
        lib=registration_library,
    )
    torch.library.register_autograd(
        "voicehub_kernels::codec_snake_beta",
        codec_snake_beta_backward,
        setup_context=pair_setup_context,
        lib=registration_library,
    )
    _CUDA_REGISTRATION_LIBRARY = registration_library


def load_tts_activation_cuda_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> LoadedCudaExtension:
    """Build and register the optional compile/autograd-aware CUDA kernels."""
    loaded = load_cuda_extension(
        ACTIVATION_CUDA_EXTENSION_NAME,
        build_directory=build_directory,
        verbose=verbose,
        require_runtime=True,
    )
    _register_cuda_autograd()
    _register_cuda_kernels()
    return loaded


def gated_silu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Fuse SiLU(gate) * up for LLM-based TTS feed-forward blocks."""
    _validate_pair(
        gate,
        up,
        operation="gated_silu",
        left_name="gate",
        right_name="up",
    )
    return dispatch_kernel(
        LLM_GATED_SILU,
        gate,
        up,
        backend=backend,
    )


def tanh_sigmoid_gate(
    activation: torch.Tensor,
    gate: torch.Tensor,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Fuse tanh(activation) * sigmoid(gate) for VITS WaveNet blocks."""
    _validate_pair(
        activation,
        gate,
        operation="tanh_sigmoid_gate",
        left_name="activation",
        right_name="gate",
    )
    return dispatch_kernel(
        VITS_TANH_SIGMOID_GATE,
        activation,
        gate,
        backend=backend,
    )


def fused_add_tanh_sigmoid(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Fuse the complete add/split/tanh/sigmoid VITS WaveNet gate."""
    _validate_vits_fused_gate(input_a, input_b, channels)
    return dispatch_kernel(
        VITS_FUSED_ADD_TANH_SIGMOID,
        input_a,
        input_b,
        channels,
        backend=backend,
    )


def fused_bias_gelu(
    inputs: torch.Tensor,
    bias: torch.Tensor,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Fuse bias addition with F5's tanh-approximate GELU projection."""
    _validate_bias_gelu(inputs, bias)
    return dispatch_kernel(
        DIFFUSION_FUSED_BIAS_GELU,
        inputs,
        bias,
        backend=backend,
    )


def fused_modulate(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Fuse DiT ``hidden * (1 + scale) + shift`` with broadcasting."""
    _validate_fused_modulate(hidden_states, shift, scale)
    return dispatch_kernel(
        DIFFUSION_FUSED_MODULATE,
        hidden_states,
        shift,
        scale,
        backend=backend,
    )


def codec_snake(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Run the exact periodic Snake activation used by native audio codecs."""
    _validate_codec_snake(inputs, alpha)
    return dispatch_kernel(
        AUDIO_CODEC_SNAKE,
        inputs,
        alpha,
        backend=backend,
    )


def codec_snake_beta(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    backend: KernelBackend | str = KernelBackend.AUTO,
) -> torch.Tensor:
    """Run exact SnakeBeta with independent frequency and magnitude."""
    _validate_codec_snake_beta(inputs, alpha, beta)
    return dispatch_kernel(
        AUDIO_CODEC_SNAKE_BETA,
        inputs,
        alpha,
        beta,
        backend=backend,
    )


_register_builtin_kernels()
