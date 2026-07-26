"""
Lightweight GPU memory monitor for the MOSS-TTS pipeline.

Measurement strategy (in priority order):
  1. **Device-level used memory** via ``nvmlDeviceGetMemoryInfo().used``.
     This is the same number ``nvidia-smi`` shows in its bar chart.  We track
     deltas between snapshots — since components are loaded sequentially, the
     delta equals this process's allocation.
  2. If pynvml is unavailable, falls back to parsing ``nvidia-smi`` output.

We deliberately avoid per-process NVML queries
(``nvmlDeviceGetComputeRunningProcesses``) because they return 0 on many
H100/container/driver combinations.

For PyTorch specifically we *also* report ``torch.cuda.memory_allocated``
(true tensor footprint) vs ``memory_reserved`` (caching-allocator pool).

Usage inside pipeline::

    mon = GpuMonitor()
    mon.snapshot("after_init")
    ...
    mon.snapshot("after_prefill")
    print(mon.format_summary())
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

_nvml_inited = False
_nvml_handle = None


def _init_nvml() -> bool:
    global _nvml_inited, _nvml_handle
    if _nvml_inited:
        return _nvml_handle is not None
    _nvml_inited = True
    try:
        from pynvml import (  # noqa: F401
            nvmlInit,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetName,
            nvmlDeviceGetMemoryInfo,
        )
        nvmlInit()
        idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        _nvml_handle = nvmlDeviceGetHandleByIndex(idx)
        return True
    except Exception:
        return False


def gpu_device_used_mb() -> float:
    """Device-level GPU memory used (all processes), in MB.

    This is the reliable metric — works on all driver/container combos.
    """
    if not _init_nvml():
        return _nvidia_smi_device_used_mb()
    from pynvml import nvmlDeviceGetMemoryInfo
    mem = nvmlDeviceGetMemoryInfo(_nvml_handle)
    return mem.used / 1e6


def gpu_name_and_total() -> tuple[str, float]:
    """Return (gpu_name, total_mb)."""
    if not _init_nvml():
        return "unknown", 0.0
    from pynvml import nvmlDeviceGetName, nvmlDeviceGetMemoryInfo
    name = nvmlDeviceGetName(_nvml_handle)
    if isinstance(name, bytes):
        name = name.decode()
    mem = nvmlDeviceGetMemoryInfo(_nvml_handle)
    return name, mem.total / 1e6


def _nvidia_smi_device_used_mb() -> float:
    """Fallback: parse ``nvidia-smi`` for device-level used memory."""
    import subprocess
    try:
        idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={idx}"],
            text=True, timeout=5,
        )
        return float(out.strip())
    except Exception:
        pass
    return 0.0


def _sync():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _torch_allocated_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except ImportError:
        pass
    return 0.0


@dataclass
class _Snap:
    label: str
    gpu_used_mb: float      # device-level used (all processes)
    torch_alloc_mb: float   # torch.cuda.memory_allocated
    wall_time: float


class GpuMonitor:
    """Accumulates labelled GPU memory snapshots and prints a summary."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._snaps: list[_Snap] = []
        self._gpu_name: str = ""
        self._gpu_total_mb: float = 0.0
        if enabled:
            self._gpu_name, self._gpu_total_mb = gpu_name_and_total()

    def snapshot(self, label: str) -> float:
        """Take a snapshot.  Returns current device-level GPU used MB."""
        if not self.enabled:
            return 0.0
        _sync()
        used = gpu_device_used_mb()
        talloc = _torch_allocated_mb()
        self._snaps.append(_Snap(label=label, gpu_used_mb=used,
                                  torch_alloc_mb=talloc, wall_time=time.time()))
        return used

    @property
    def peak_gpu_mb(self) -> float:
        if not self._snaps:
            return 0.0
        return max(s.gpu_used_mb for s in self._snaps)

    @property
    def snapshots(self) -> list[tuple[str, float]]:
        return [(s.label, s.gpu_used_mb) for s in self._snaps]

    def format_summary(self) -> str:
        if not self._snaps:
            return "(no GPU snapshots)"

        base = self._snaps[0].gpu_used_mb
        lines: list[str] = []
        lines.append(f"  GPU: {self._gpu_name}  ({self._gpu_total_mb / 1024:.1f} GB total)")
        lines.append(f"  Note: values are device-level used memory (like nvidia-smi).")
        lines.append(f"        Δ columns show change from previous snapshot.")
        lines.append(f"  {'Stage':<35s} {'Used MB':>8s}  {'Δ MB':>8s}  {'This proc':>10s}")
        lines.append(f"  {'─' * 35} {'─' * 8}  {'─' * 8}  {'─' * 10}")
        prev = base
        for s in self._snaps:
            delta = s.gpu_used_mb - prev
            from_base = s.gpu_used_mb - base
            ds = f"+{delta:.0f}" if delta >= 0 else f"{delta:.0f}"
            lines.append(f"  {s.label:<35s} {s.gpu_used_mb:>8.0f}  {ds:>8s}  {from_base:>+10.0f}")
            prev = s.gpu_used_mb
        lines.append(f"  {'─' * 35} {'─' * 8}           {'─' * 10}")
        total_alloc = max(s.gpu_used_mb for s in self._snaps) - base
        lines.append(f"  {'TOTAL (peak − baseline)':<35s} {'':>8s}           {total_alloc:>+10.0f}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        base = self._snaps[0].gpu_used_mb if self._snaps else 0
        return {
            "gpu_name": self._gpu_name,
            "gpu_total_mb": self._gpu_total_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
            "total_allocated_mb": self.peak_gpu_mb - base,
            "snapshots": {s.label: s.gpu_used_mb for s in self._snaps},
        }
