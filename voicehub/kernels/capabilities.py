"""Side-effect-free capability probes for optional accelerator backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from types import MappingProxyType
from typing import Mapping

import torch


@dataclass(frozen=True)
class CapabilityStatus:
    """Availability and diagnostic details for one optional backend."""

    available: bool
    reason: str
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("Capability `available` must be a boolean.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("Capability `reason` must be a non-empty string.")
        if not isinstance(self.details, Mapping):
            raise TypeError("Capability `details` must be a mapping.")
        normalized = {}
        for key, value in self.details.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("Capability detail keys and values must be strings.")
            normalized[key] = value
        object.__setattr__(self, "details", MappingProxyType(normalized))

    def __bool__(self) -> bool:
        return self.available


@dataclass(frozen=True)
class KernelCapabilities:
    """Snapshot of the optional kernel backends on the current host."""

    cuda_runtime: CapabilityStatus
    triton: CapabilityStatus
    cuda_extension: CapabilityStatus


def _cuda_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda")
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid CUDA probe device {device!r}.") from error
    return resolved


def cuda_runtime_capability(device: torch.device | str | None = None, ) -> CapabilityStatus:
    """Probe whether this PyTorch process can execute CUDA kernels."""
    resolved = _cuda_device(device)
    if resolved.type != "cuda":
        return CapabilityStatus(
            False,
            f"device {str(resolved)!r} is not a CUDA device",
        )
    compiled_version = torch.version.cuda
    if compiled_version is None:
        return CapabilityStatus(
            False,
            "the installed PyTorch build has no CUDA runtime",
        )
    if not torch.cuda.is_available():
        return CapabilityStatus(
            False,
            "PyTorch reports that CUDA is unavailable",
            {"torch_cuda": compiled_version},
        )

    try:
        index = resolved.index
        if index is None:
            index = torch.cuda.current_device()
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            return CapabilityStatus(
                False,
                f"CUDA device index {index} is outside the visible range",
                {
                    "device_count": str(device_count),
                    "torch_cuda": compiled_version,
                },
            )
        major, minor = torch.cuda.get_device_capability(index)
        device_name = torch.cuda.get_device_name(index)
    except (AssertionError, RuntimeError) as error:
        return CapabilityStatus(
            False,
            f"CUDA device discovery failed: {error}",
            {"torch_cuda": compiled_version},
        )
    return CapabilityStatus(
        True,
        "CUDA runtime is available",
        {
            "compute_capability": f"{major}.{minor}",
            "device": device_name,
            "device_index": str(index),
            "torch_cuda": compiled_version,
        },
    )


def triton_capability(device: torch.device | str | None = None, ) -> CapabilityStatus:
    """Probe Triton lazily after validating a usable CUDA device."""
    cuda = cuda_runtime_capability(device)
    if not cuda.available:
        return CapabilityStatus(
            False,
            f"Triton requires CUDA: {cuda.reason}",
            cuda.details,
        )
    try:
        triton = import_module("triton")
    except (ImportError, OSError, RuntimeError) as error:
        return CapabilityStatus(
            False,
            f"Triton could not be imported: {error}",
            cuda.details,
        )
    version = getattr(triton, "__version__", "unknown")
    return CapabilityStatus(
        True,
        "Triton and CUDA are available",
        {
            **cuda.details,
            "triton": str(version),
        },
    )


def cuda_extension_capability(
    *,
    require_runtime: bool = True,
) -> CapabilityStatus:
    """Probe PyTorch's JIT CUDA-extension toolchain without compiling code."""
    if not isinstance(require_runtime, bool):
        raise TypeError("`require_runtime` must be a boolean.")
    compiled_version = torch.version.cuda
    if compiled_version is None:
        return CapabilityStatus(
            False,
            "the installed PyTorch build has no CUDA extension ABI",
        )
    runtime = cuda_runtime_capability() if require_runtime else None
    if runtime is not None and not runtime.available:
        return CapabilityStatus(
            False,
            f"CUDA extension runtime is unavailable: {runtime.reason}",
            runtime.details,
        )
    try:
        cpp_extension = import_module("torch.utils.cpp_extension")
    except (ImportError, OSError, RuntimeError) as error:
        return CapabilityStatus(
            False,
            f"torch.utils.cpp_extension could not be imported: {error}",
            {"torch_cuda": compiled_version},
        )
    cuda_home = getattr(cpp_extension, "CUDA_HOME", None)
    if cuda_home is None:
        return CapabilityStatus(
            False,
            "the CUDA toolkit was not found (CUDA_HOME is unset)",
            {"torch_cuda": compiled_version},
        )
    details = {
        "cuda_home": str(cuda_home),
        "torch_cuda": compiled_version,
    }
    if runtime is not None:
        details.update(runtime.details)
    return CapabilityStatus(
        True,
        "PyTorch's CUDA extension toolchain is available",
        details,
    )


def get_kernel_capabilities() -> KernelCapabilities:
    """Return a fresh capability snapshot for diagnostics and setup UIs."""
    return KernelCapabilities(
        cuda_runtime=cuda_runtime_capability(),
        triton=triton_capability(),
        cuda_extension=cuda_extension_capability(),
    )
