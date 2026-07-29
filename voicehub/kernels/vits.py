"""Architecture-neutral custom-kernel protocol for VITS WaveNet blocks."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module

import torch

from voicehub.kernels.activations import (
    VITS_FUSED_ADD_TANH_SIGMOID,
    fused_add_tanh_sigmoid,
    fused_add_tanh_sigmoid_reference,
)
from voicehub.kernels.registry import KernelBackend


def _lazy_triton_vits_gate(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    return import_module("voicehub.kernels.triton_activations").fused_add_tanh_sigmoid_triton(
        input_a, input_b, channels)


def _lazy_cuda_vits_gate(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.fused_add_tanh_sigmoid(
        input_a,
        input_b,
        channels,
    )


class VITSKernelOptimizable:
    """Mixin exposing one reversible selector for VITS-family blocks.

    The mixin owns no parameters, buffers, or child modules, so adopting
    it cannot change checkpoint keys. Architectures keep their original
    module topology while the universal custom-kernel pass discovers
    this protocol structurally through :meth:`set_kernel_backend`.
    """

    supported_kernel_operations = (VITS_FUSED_ADD_TANH_SIGMOID, )
    kernel_backend: KernelBackend
    _vits_gate_implementation: Callable[..., torch.Tensor]

    def _initialize_vits_kernel_backend(self) -> None:
        self.set_kernel_backend(KernelBackend.TORCH)

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        """Select the backend used by every VITS gate in this block."""
        resolved = KernelBackend.coerce(backend)
        if resolved is KernelBackend.TORCH:
            implementation = fused_add_tanh_sigmoid_reference
        elif resolved is KernelBackend.TRITON:
            loaded = sys.modules.get("voicehub.kernels.triton_activations")
            implementation = (
                _lazy_triton_vits_gate if loaded is None else loaded.fused_add_tanh_sigmoid_triton)
        elif resolved is KernelBackend.CUDA_EXTENSION:
            implementation = _lazy_cuda_vits_gate
        else:
            implementation = fused_add_tanh_sigmoid
        self.kernel_backend = resolved
        self._vits_gate_implementation = implementation

    def resolve_kernel_backend(
        self,
        backend: KernelBackend | str,
        *,
        device: str,
        dtype: str,
    ) -> KernelBackend:
        """Resolve ``auto`` once before graph capture and preload its op."""
        del dtype
        requested = KernelBackend.coerce(backend)
        selected = requested
        if requested is KernelBackend.AUTO:
            if device.partition(":")[0] != "cuda":
                selected = KernelBackend.TORCH
            else:
                from voicehub.kernels.activations import ACTIVATION_CUDA_EXTENSION_NAME
                from voicehub.kernels.cuda_extensions import CUDA_EXTENSIONS

                if CUDA_EXTENSIONS.is_loaded(ACTIVATION_CUDA_EXTENSION_NAME):
                    selected = KernelBackend.CUDA_EXTENSION
                else:
                    from voicehub.kernels.capabilities import triton_capability

                    selected = (
                        KernelBackend.TRITON if triton_capability(device).available else KernelBackend.TORCH)
        if selected is KernelBackend.TRITON:
            import_module("voicehub.kernels.triton_activations")
        elif selected is KernelBackend.CUDA_EXTENSION:
            _ = torch.ops.voicehub_kernels.fused_add_tanh_sigmoid
        return selected

    def _vits_fused_gate(
        self,
        input_a: torch.Tensor,
        input_b: torch.Tensor,
        channels: int,
    ) -> torch.Tensor:
        return self._vits_gate_implementation(
            input_a,
            input_b,
            channels,
        )


__all__ = ["VITSKernelOptimizable"]
