"""Fused activation kernels shared by native TTS architecture families."""

from __future__ import annotations

import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from threading import RLock

import torch
from torch.nn import functional as F

from voicehub.kernels.capabilities import triton_capability
from voicehub.kernels.cuda_extensions import (
    CudaExtensionSpec,
    LoadedCudaExtension,
    load_cuda_extension,
    register_cuda_extension,
)
from voicehub.kernels.registry import KernelBackend, KernelSupport, dispatch_kernel, register_kernel

LLM_GATED_SILU = "tts.llm.gated_silu"
VITS_TANH_SIGMOID_GATE = "tts.vits.tanh_sigmoid_gate"
DIFFUSION_FUSED_BIAS_GELU = "tts.diffusion.fused_bias_gelu"
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
        "voicehub_kernels::fused_bias_gelu",
    ),
    extra_cflags=("-O3", ),
    extra_cuda_cflags=("-O3", "--use_fast_math"),
)
_CUDA_REGISTRATION_LIBRARY: torch.library.Library | None = None
_CUDA_REGISTRATION_LOCK = RLock()


@lru_cache(maxsize=None)
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


def fused_bias_gelu_reference(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for F5 feed-forward bias + approximate GELU."""
    return F.gelu(inputs + bias, approximate="tanh")


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


def _fused_bias_gelu_triton(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.triton_activations")
    return module.fused_bias_gelu_triton(inputs, bias)


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


def _fused_bias_gelu_cuda(
    inputs: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.fused_bias_gelu(inputs, bias)


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
            DIFFUSION_FUSED_BIAS_GELU,
            _fused_bias_gelu_cuda,
            _bias_gelu_cuda_extension_support,
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
        "voicehub_kernels::fused_bias_gelu",
        fake_fused_bias_gelu,
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
        "voicehub_kernels::fused_bias_gelu",
        fused_bias_gelu_backward,
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


_register_builtin_kernels()
