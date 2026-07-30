"""Architecture-neutral fused-kernel protocol for diffusion/flow DiT blocks."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module

import torch

from voicehub.kernel_operations import DIFFUSION_FUSED_MODULATE
from voicehub.kernels.activations import fused_modulate, fused_modulate_reference
from voicehub.kernels.registry import KernelBackend


def _lazy_triton_modulate(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return import_module("voicehub.kernels.triton_activations").fused_modulate_triton(
        hidden_states,
        shift,
        scale,
    )


def _lazy_cuda_modulate(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.fused_modulate(
        hidden_states,
        shift,
        scale,
    )


class DiffusionModulationKernelOptimizable:
    """Mixin selecting the exact AdaLN modulation implementation.

    The selector owns no parameters, buffers, or child modules. Adopting
    it therefore preserves checkpoint topology and lets the universal
    custom-kernel pass discover compatible DiT blocks structurally.
    """

    supported_kernel_operations = (DIFFUSION_FUSED_MODULATE, )
    kernel_backend: KernelBackend
    _diffusion_modulate_implementation: Callable[..., torch.Tensor]

    def _initialize_diffusion_kernel_backend(self) -> None:
        self.set_kernel_backend(KernelBackend.TORCH)

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        resolved = KernelBackend.coerce(backend)
        if resolved is KernelBackend.TORCH:
            implementation = fused_modulate_reference
        elif resolved is KernelBackend.TRITON:
            loaded = sys.modules.get("voicehub.kernels.triton_activations")
            implementation = (_lazy_triton_modulate if loaded is None else loaded.fused_modulate_triton)
        elif resolved is KernelBackend.CUDA_EXTENSION:
            implementation = _lazy_cuda_modulate
        else:
            implementation = fused_modulate
        self.kernel_backend = resolved
        self._diffusion_modulate_implementation = implementation

    def resolve_kernel_backend(
        self,
        backend: KernelBackend | str,
        *,
        device: str,
        dtype: str,
    ) -> KernelBackend:
        """Resolve ``auto`` before graph capture and preload its operator."""
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
            _ = torch.ops.voicehub_kernels.fused_modulate
        return selected

    def _diffusion_modulate(
        self,
        hidden_states: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return self._diffusion_modulate_implementation(
            hidden_states,
            shift,
            scale,
        )


__all__ = ["DiffusionModulationKernelOptimizable"]
