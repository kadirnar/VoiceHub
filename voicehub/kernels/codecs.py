"""Structural custom-kernel protocol for native neural audio codecs."""

from __future__ import annotations

import sys
from collections.abc import Callable
from enum import Enum
from importlib import import_module

import torch

from voicehub.kernel_operations import AUDIO_CODEC_EUCLIDEAN_VQ, AUDIO_CODEC_SNAKE, AUDIO_CODEC_SNAKE_BETA
from voicehub.kernels.activations import (
    codec_snake,
    codec_snake_beta,
    codec_snake_beta_reference,
    codec_snake_reference,
)
from voicehub.kernels.registry import KernelBackend


class CodecKernelBackend(str, Enum):
    """Implementation families understood by codec-only kernel policies.

    ``cute`` is intentionally codec-scoped.  It is a recognized optional
    provider for future matrix-heavy codec operations, not a backend
    that the universal TTS selector may apply to unrelated diffusion or
    VITS blocks.
    """

    AUTO = "auto"
    NATIVE = "native"
    TORCH = "torch"
    TRITON = "triton"
    CUTE = "cute"
    CUDA_EXTENSION = "cuda_extension"

    @classmethod
    def coerce(
        cls,
        value: CodecKernelBackend | KernelBackend | str,
    ) -> CodecKernelBackend:
        if isinstance(value, cls):
            return value
        if isinstance(value, KernelBackend):
            value = value.value
        if not isinstance(value, str):
            raise TypeError(
                "Codec kernel backends must be strings, KernelBackend, or "
                "CodecKernelBackend values.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "cutlass": cls.CUTE.value,
            "cute_dsl": cls.CUTE.value,
            "disabled": cls.NATIVE.value,
        }
        try:
            return cls(aliases.get(normalized, normalized))
        except ValueError as error:
            available = ", ".join(backend.value for backend in cls)
            raise ValueError(
                f"Unknown codec kernel backend {value!r}; expected one of: "
                f"{available}.") from error

    def generic_backend(self) -> KernelBackend:
        """Return the shared backend for providers implemented today."""
        if self in {CodecKernelBackend.NATIVE, CodecKernelBackend.CUTE}:
            raise ValueError(f"Codec backend {self.value!r} has no generic kernel-backend "
                             "equivalent.")
        return KernelBackend.coerce(self.value)


class CodecKernelBackendUnavailableError(RuntimeError):
    """A recognized codec backend cannot execute the requested operation."""


class _CodecKernelSelectorMixin:
    """Codec-domain selector surface layered over the legacy generic API."""

    supported_codec_kernel_backends = (
        CodecKernelBackend.TORCH,
        CodecKernelBackend.TRITON,
        CodecKernelBackend.CUDA_EXTENSION,
    )
    kernel_backend: KernelBackend

    @property
    def codec_kernel_backend(self) -> CodecKernelBackend:
        return CodecKernelBackend.coerce(self.kernel_backend)

    def set_codec_kernel_backend(
        self,
        backend: CodecKernelBackend | KernelBackend | str,
    ) -> None:
        resolved = CodecKernelBackend.coerce(backend)
        if resolved is CodecKernelBackend.AUTO:
            self.set_kernel_backend(KernelBackend.AUTO)
            return
        if resolved is CodecKernelBackend.NATIVE:
            resolved = CodecKernelBackend.TORCH
        if resolved not in self.supported_codec_kernel_backends:
            operations = ", ".join(getattr(self, "supported_kernel_operations", ()), )
            raise CodecKernelBackendUnavailableError(
                f"Codec backend {resolved.value!r} has no implementation for "
                f"{operations or 'this operation'}.")
        self.set_kernel_backend(resolved.generic_backend())

    def resolve_codec_kernel_backend(
        self,
        backend: CodecKernelBackend | KernelBackend | str,
        *,
        device: str,
        dtype: str,
    ) -> CodecKernelBackend:
        resolved = CodecKernelBackend.coerce(backend)
        if resolved is CodecKernelBackend.NATIVE:
            return CodecKernelBackend.TORCH
        if resolved not in self.supported_codec_kernel_backends:
            # Architecture-wide codec policies are operation-specific. A
            # CuTe VQ request must leave activation-only targets on Torch
            # instead of claiming that CuTe accelerated those operations.
            return CodecKernelBackend.TORCH
        selected = self.resolve_kernel_backend(
            resolved.generic_backend(),
            device=device,
            dtype=dtype,
        )
        return CodecKernelBackend.coerce(selected)


def _validate_euclidean_vq_inputs(
    encodings: torch.Tensor,
    codebook: torch.Tensor,
) -> None:
    if not isinstance(encodings, torch.Tensor) or not isinstance(codebook, torch.Tensor):
        raise TypeError("`encodings` and `codebook` must be torch.Tensor values.")
    if encodings.ndim != 2 or codebook.ndim != 2:
        raise ValueError(
            "codec_euclidean_vq_search expects [vectors, dimension] "
            "and [codes, dimension] tensors.")
    if encodings.shape[1] != codebook.shape[1]:
        raise ValueError("codec_euclidean_vq_search expects one shared embedding dimension.")
    if codebook.shape[0] < 1:
        raise ValueError("codec_euclidean_vq_search requires a non-empty codebook.")
    if encodings.device != codebook.device:
        raise ValueError("codec_euclidean_vq_search expects tensors on the same device.")
    if encodings.dtype != codebook.dtype:
        raise ValueError("codec_euclidean_vq_search expects tensors with the same dtype.")
    if not encodings.is_floating_point():
        raise TypeError("codec_euclidean_vq_search expects floating-point tensors.")


def codec_euclidean_vq_search_reference(
    encodings: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    """Return nearest-code indices with the native PyTorch distance formula."""
    _validate_euclidean_vq_inputs(encodings, codebook)
    distances = (
        encodings.square().sum(1, keepdim=True) - 2 * encodings @ codebook.transpose(0, 1) +
        codebook.square().sum(1, keepdim=True).transpose(0, 1))
    return distances.argmin(dim=1)


def _lazy_cute_euclidean_vq_search(
    encodings: torch.Tensor,
    codebook: torch.Tensor,
) -> torch.Tensor:
    module = import_module("voicehub.kernels.cute_codecs")
    return module.codec_euclidean_vq_search_cute(encodings, codebook)


def codec_euclidean_vq_search(
    encodings: torch.Tensor,
    codebook: torch.Tensor,
    *,
    backend: CodecKernelBackend | KernelBackend | str = CodecKernelBackend.AUTO,
) -> torch.Tensor:
    """Execute codec VQ search through one explicit operation backend."""
    resolved = CodecKernelBackend.coerce(backend)
    if resolved in {
            CodecKernelBackend.AUTO,
            CodecKernelBackend.NATIVE,
            CodecKernelBackend.TORCH,
    }:
        return codec_euclidean_vq_search_reference(encodings, codebook)
    if resolved is CodecKernelBackend.CUTE:
        return _lazy_cute_euclidean_vq_search(encodings, codebook)
    raise CodecKernelBackendUnavailableError(
        f"Codec backend {resolved.value!r} has no implementation for "
        f"{AUDIO_CODEC_EUCLIDEAN_VQ}.")


class CodecEuclideanVQKernelOptimizable:
    """Mixin exposing Torch/CuTe selection for dense Euclidean VQ search."""

    supported_kernel_operations = (AUDIO_CODEC_EUCLIDEAN_VQ, )
    supported_codec_kernel_backends = (
        CodecKernelBackend.TORCH,
        CodecKernelBackend.CUTE,
    )
    _codec_euclidean_vq_backend: CodecKernelBackend
    _codec_euclidean_vq_implementation: Callable[..., torch.Tensor]

    @property
    def codec_kernel_backend(self) -> CodecKernelBackend:
        return self._codec_euclidean_vq_backend

    def _initialize_codec_kernel_backend(self) -> None:
        self._codec_euclidean_vq_backend = CodecKernelBackend.TORCH
        self._codec_euclidean_vq_implementation = (codec_euclidean_vq_search_reference)

    def set_codec_kernel_backend(
        self,
        backend: CodecKernelBackend | KernelBackend | str,
    ) -> None:
        resolved = CodecKernelBackend.coerce(backend)
        if resolved in {
                CodecKernelBackend.AUTO,
                CodecKernelBackend.NATIVE,
        }:
            resolved = CodecKernelBackend.TORCH
        if resolved not in self.supported_codec_kernel_backends:
            raise CodecKernelBackendUnavailableError(
                f"Codec backend {resolved.value!r} has no implementation for "
                f"{AUDIO_CODEC_EUCLIDEAN_VQ}.")
        if resolved is CodecKernelBackend.CUTE:
            implementation = import_module("voicehub.kernels.cute_codecs", ).codec_euclidean_vq_search_cute
        else:
            implementation = codec_euclidean_vq_search_reference
        self._codec_euclidean_vq_backend = resolved
        self._codec_euclidean_vq_implementation = implementation

    def resolve_codec_kernel_backend(
        self,
        backend: CodecKernelBackend | KernelBackend | str,
        *,
        device: str,
        dtype: str,
    ) -> CodecKernelBackend:
        del device, dtype
        resolved = CodecKernelBackend.coerce(backend)
        if resolved in {
                CodecKernelBackend.AUTO,
                CodecKernelBackend.NATIVE,
        }:
            return CodecKernelBackend.TORCH
        if resolved not in self.supported_codec_kernel_backends:
            return CodecKernelBackend.TORCH
        return resolved

    def _codec_euclidean_vq_search(
        self,
        encodings: torch.Tensor,
        codebook: torch.Tensor,
    ) -> torch.Tensor:
        return self._codec_euclidean_vq_implementation(encodings, codebook)


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


class CodecSnakeKernelOptimizable(_CodecKernelSelectorMixin):
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


class CodecSnakeBetaKernelOptimizable(_CodecKernelSelectorMixin):
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
    "CodecEuclideanVQKernelOptimizable",
    "CodecKernelBackend",
    "CodecKernelBackendUnavailableError",
    "CodecSnakeBetaKernelOptimizable",
    "CodecSnakeKernelOptimizable",
    "codec_euclidean_vq_search",
    "codec_euclidean_vq_search_reference",
]
