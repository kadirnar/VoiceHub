"""Structural custom-kernel protocol for native neural audio codecs."""

from __future__ import annotations

import sys
from collections.abc import Callable
from importlib import import_module

import torch

from voicehub.kernel_operations import AUDIO_CODEC_SNAKE, AUDIO_CODEC_SNAKE_BETA
from voicehub.kernels.activations import (
    codec_snake,
    codec_snake_beta,
    codec_snake_beta_reference,
    codec_snake_reference,
)
from voicehub.kernels.registry import KernelBackend


def _lazy_triton_snake(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return import_module("voicehub.kernels.triton_activations").codec_snake_triton(
        inputs,
        alpha,
    )


def _lazy_cuda_snake(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.codec_snake(inputs, alpha)


def _lazy_triton_snake_beta(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return import_module("voicehub.kernels.triton_activations").codec_snake_beta_triton(
        inputs,
        alpha,
        beta,
    )


def _lazy_cuda_snake_beta(
    inputs: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.voicehub_kernels.codec_snake_beta(
        inputs,
        alpha,
        beta,
    )


class CodecSnakeKernelOptimizable:
    """Mixin exposing an exact, reversible Snake backend selector."""

    supported_kernel_operations = (AUDIO_CODEC_SNAKE, )
    kernel_backend: KernelBackend
    _codec_snake_implementation: Callable[..., torch.Tensor]

    def _initialize_codec_kernel_backend(self) -> None:
        self.set_kernel_backend(KernelBackend.TORCH)

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        resolved = KernelBackend.coerce(backend)
        if resolved is KernelBackend.TORCH:
            implementation = codec_snake_reference
        elif resolved is KernelBackend.TRITON:
            loaded = sys.modules.get("voicehub.kernels.triton_activations")
            implementation = (_lazy_triton_snake if loaded is None else loaded.codec_snake_triton)
        elif resolved is KernelBackend.CUDA_EXTENSION:
            implementation = _lazy_cuda_snake
        else:
            implementation = codec_snake
        self.kernel_backend = resolved
        self._codec_snake_implementation = implementation

    def resolve_kernel_backend(
        self,
        backend: KernelBackend | str,
        *,
        device: str,
        dtype: str,
    ) -> KernelBackend:
        del device, dtype
        requested = KernelBackend.coerce(backend)
        selected = requested
        if requested is KernelBackend.AUTO:
            # Architecture-wide AUTO has no numerical-fidelity declaration.
            # CodecOptimizationConfig resolves relaxed AUTO to an explicit
            # accelerator before calling this selector.
            selected = KernelBackend.TORCH
        if selected is KernelBackend.TRITON:
            import_module("voicehub.kernels.triton_activations")
        elif selected is KernelBackend.CUDA_EXTENSION:
            _ = torch.ops.voicehub_kernels.codec_snake
        return selected

    def _codec_snake(
        self,
        inputs: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        return self._codec_snake_implementation(inputs, alpha)


class CodecSnakeBetaKernelOptimizable:
    """Mixin exposing an exact, reversible SnakeBeta backend selector."""

    supported_kernel_operations = (AUDIO_CODEC_SNAKE_BETA, )
    kernel_backend: KernelBackend
    _codec_snake_beta_implementation: Callable[..., torch.Tensor]

    def _initialize_codec_kernel_backend(self) -> None:
        self.set_kernel_backend(KernelBackend.TORCH)

    def set_kernel_backend(self, backend: KernelBackend | str) -> None:
        resolved = KernelBackend.coerce(backend)
        if resolved is KernelBackend.TORCH:
            implementation = codec_snake_beta_reference
        elif resolved is KernelBackend.TRITON:
            loaded = sys.modules.get("voicehub.kernels.triton_activations")
            implementation = (_lazy_triton_snake_beta if loaded is None else loaded.codec_snake_beta_triton)
        elif resolved is KernelBackend.CUDA_EXTENSION:
            implementation = _lazy_cuda_snake_beta
        else:
            implementation = codec_snake_beta
        self.kernel_backend = resolved
        self._codec_snake_beta_implementation = implementation

    def resolve_kernel_backend(
        self,
        backend: KernelBackend | str,
        *,
        device: str,
        dtype: str,
    ) -> KernelBackend:
        del device, dtype
        requested = KernelBackend.coerce(backend)
        selected = requested
        if requested is KernelBackend.AUTO:
            # Periodic accelerator transcendental math is opt-in. See the
            # Snake selector above and CodecOptimizationConfig's fidelity
            # policy.
            selected = KernelBackend.TORCH
        if selected is KernelBackend.TRITON:
            import_module("voicehub.kernels.triton_activations")
        elif selected is KernelBackend.CUDA_EXTENSION:
            _ = torch.ops.voicehub_kernels.codec_snake_beta
        return selected

    def _codec_snake_beta(
        self,
        inputs: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        return self._codec_snake_beta_implementation(
            inputs,
            alpha,
            beta,
        )


__all__ = [
    "CodecSnakeBetaKernelOptimizable",
    "CodecSnakeKernelOptimizable",
]
