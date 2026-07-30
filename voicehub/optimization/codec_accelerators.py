"""Reversible accelerator selection restricted to neural audio codecs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.kernels.activations import ACTIVATION_CUDA_EXTENSION_NAME
from voicehub.kernels.capabilities import cute_operator_capability, triton_capability
from voicehub.kernels.codecs import CodecKernelBackend
from voicehub.kernels.cuda_extensions import CUDA_EXTENSIONS
from voicehub.optimization.accelerators import _SelectorPass, _SelectorTarget
from voicehub.optimization.capabilities import OptimizationContext


def _codec_cuda_extension_loaded() -> bool:
    """Check process state without compiling or loading an extension."""
    return CUDA_EXTENSIONS.is_loaded(ACTIVATION_CUDA_EXTENSION_NAME)


class CodecKernelPass(_SelectorPass):
    """Configure only modules exposing the codec-specific selector protocol.

    The legacy :class:`~voicehub.optimization.accelerators.CustomKernelPass`
    remains available for existing applications.  New codec plans use this
    pass so a mixed runtime cannot accidentally apply the codec policy to its
    diffusion, VITS, or language-model blocks.
    """

    pass_id = "codec-kernels"
    pass_version = "1"
    optimization_kind = "codec-kernels"
    setter_name = "set_codec_kernel_backend"
    state_attribute = "codec_kernel_backend"

    def __init__(
        self,
        *,
        backend: CodecKernelBackend | str = CodecKernelBackend.AUTO,
    ) -> None:
        self.backend = CodecKernelBackend.coerce(backend)

    @property
    def selection(self) -> CodecKernelBackend:
        return self.backend

    def _coerce_previous(self, value: Any) -> CodecKernelBackend:
        return CodecKernelBackend.coerce(value)

    def _selection_issues(
        self,
        context: OptimizationContext,
    ) -> tuple[str, ...]:
        issues = []
        device_family = context.device.partition(":")[0]
        accelerated = {
            CodecKernelBackend.TRITON,
            CodecKernelBackend.CUTE,
            CodecKernelBackend.CUDA_EXTENSION,
        }
        if self.backend in accelerated and device_family != "cuda":
            issues.append(f"{self.backend.value} codec kernels need a CUDA context")
        if (self.backend in accelerated and context.dtype not in {"float16", "bfloat16", "float32"}):
            issues.append(f"{self.backend.value} codec kernels do not support "
                          f"dtype {context.dtype!r}")
        if self.backend is CodecKernelBackend.CUDA_EXTENSION:
            if not _codec_cuda_extension_loaded():
                issues.append(
                    f"CUDA extension {ACTIVATION_CUDA_EXTENSION_NAME!r} is not "
                    "already loaded; load it explicitly before applying this pass")
        elif (self.backend is CodecKernelBackend.TRITON and device_family == "cuda"):
            capability = triton_capability(context.device)
            if not capability.available:
                issues.append(f"triton codec backend is unavailable: {capability.reason}")
        elif self.backend is CodecKernelBackend.CUTE and device_family == "cuda":
            capability = cute_operator_capability(context.device)
            if not capability.available:
                issues.append(f"CuTe codec backend is unavailable: {capability.reason}")
        return tuple(issues)

    def _selection_for_target(
        self,
        target: _SelectorTarget,
        context: OptimizationContext,
    ) -> CodecKernelBackend:
        resolver = getattr(
            target.module,
            "resolve_codec_kernel_backend",
            None,
        )
        if not callable(resolver):
            return self.backend
        return CodecKernelBackend.coerce(
            resolver(
                self.backend,
                device=context.device,
                dtype=context.dtype,
            ))

    def _extra_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "domain": "codec",
            "cuda_extension": ACTIVATION_CUDA_EXTENSION_NAME,
            "cuda_extension_loaded": _codec_cuda_extension_loaded(),
        }
        if self.backend is CodecKernelBackend.CUTE:
            capability = cute_operator_capability()
            metadata["cute_dsl_available"] = capability.available
            metadata["cute_dsl_reason"] = capability.reason
        return metadata

    def manifest_configuration(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend.value,
            "domain": "codec",
        }


__all__ = ["CodecKernelPass"]
