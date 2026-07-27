"""Dependency-lazy helpers shared by native speech provider adapters."""

from __future__ import annotations

from importlib import import_module


def resolve_native_device(
        device: str,
        *,
        provider: str,
        supported_types: tuple[str, ...] = ("cpu", "cuda"),
        allow_device_index: bool = True,
) -> str:
    """Resolve ``auto`` without selecting an unsupported accelerator.

    PyTorch's generic device resolver prefers Apple MPS when it is
    available. Most native speech toolkits only document CPU and CUDA
    runtimes, so their adapters must not inherit that choice
    accidentally.
    """
    if not isinstance(device, str) or not device.strip():
        raise ValueError("`device` must be a non-empty string.")
    device = device.strip().lower()
    if device == "auto":
        if "cuda" in supported_types:
            try:
                torch = import_module("torch")
                cuda = getattr(torch, "cuda", None)
                is_available = getattr(cuda, "is_available", None)
                if callable(is_available) and is_available():
                    return "cuda"
            except ModuleNotFoundError:
                pass
        if "cpu" in supported_types:
            return "cpu"
        raise ValueError(
            f"{provider} cannot resolve `device='auto'` because it does not "
            "declare a CPU fallback.")

    device_type, separator, index = device.partition(":")
    if device_type in supported_types and (not separator or allow_device_index and index.isdigit()):
        return device
    if supported_types == ("cpu", "cuda"):
        formatted = "CPU and CUDA"
    else:
        formatted = ", ".join(name.upper() for name in supported_types)
    raise ValueError(f"{provider} supports {formatted} devices, but received "
                     f"`device={device!r}`.")


def resolve_cpu_cuda_device(
    device: str,
    *,
    provider: str,
    allow_cuda_index: bool = True,
) -> str:
    """Resolve a provider limited to CPU and CUDA runtimes."""
    return resolve_native_device(
        device,
        provider=provider,
        supported_types=("cpu", "cuda"),
        allow_device_index=allow_cuda_index,
    )
