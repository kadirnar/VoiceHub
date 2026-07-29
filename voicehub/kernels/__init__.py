"""Optional custom kernels with portable PyTorch fallbacks.

Importing this package never imports Triton, initializes CUDA, or
invokes a compiler. Accelerator backends are selected lazily from
concrete tensor arguments, and CUDA source compilation requires an
explicit loader call.
"""

from voicehub.kernels.activations import (
    ACTIVATION_CUDA_EXTENSION_NAME,
    DIFFUSION_FUSED_BIAS_GELU,
    LLM_GATED_SILU,
    VITS_TANH_SIGMOID_GATE,
    fused_bias_gelu,
    fused_bias_gelu_reference,
    gated_silu,
    gated_silu_reference,
    load_tts_activation_cuda_extension,
    tanh_sigmoid_gate,
    tanh_sigmoid_gate_reference,
)
from voicehub.kernels.capabilities import (
    CapabilityStatus,
    KernelCapabilities,
    cuda_extension_capability,
    cuda_runtime_capability,
    get_kernel_capabilities,
    triton_capability,
)
from voicehub.kernels.cuda_extensions import (
    CUDA_EXTENSIONS,
    CudaExtensionBuildError,
    CudaExtensionError,
    CudaExtensionRegistry,
    CudaExtensionSpec,
    CudaExtensionUnavailableError,
    LoadedCudaExtension,
    load_cuda_extension,
    register_cuda_extension,
)
from voicehub.kernels.registry import (
    KERNEL_REGISTRY,
    KernelBackend,
    KernelDispatchError,
    KernelError,
    KernelRegistrationError,
    KernelRegistry,
    KernelSupport,
    RegisteredKernel,
    dispatch_kernel,
    register_kernel,
    resolve_kernel,
)

__all__ = [
    "ACTIVATION_CUDA_EXTENSION_NAME",
    "CUDA_EXTENSIONS",
    "CapabilityStatus",
    "CudaExtensionBuildError",
    "CudaExtensionError",
    "CudaExtensionRegistry",
    "CudaExtensionSpec",
    "CudaExtensionUnavailableError",
    "DIFFUSION_FUSED_BIAS_GELU",
    "KERNEL_REGISTRY",
    "KernelBackend",
    "KernelCapabilities",
    "KernelDispatchError",
    "KernelError",
    "KernelRegistry",
    "KernelRegistrationError",
    "KernelSupport",
    "LLM_GATED_SILU",
    "LoadedCudaExtension",
    "RegisteredKernel",
    "VITS_TANH_SIGMOID_GATE",
    "cuda_extension_capability",
    "cuda_runtime_capability",
    "dispatch_kernel",
    "fused_bias_gelu",
    "fused_bias_gelu_reference",
    "gated_silu",
    "gated_silu_reference",
    "get_kernel_capabilities",
    "load_cuda_extension",
    "load_tts_activation_cuda_extension",
    "register_cuda_extension",
    "register_kernel",
    "resolve_kernel",
    "tanh_sigmoid_gate",
    "tanh_sigmoid_gate_reference",
    "triton_capability",
]
