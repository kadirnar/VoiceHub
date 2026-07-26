from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    rank: int
    size: int

    def __post_init__(self):
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None


def set_tp_info(rank: int, size: int) -> bool:
    """Set tensor-parallel metadata and report whether this call owns it."""
    global _TP_INFO
    requested = DistributedInfo(rank, size)
    if _TP_INFO is None:
        _TP_INFO = requested
        return True
    if _TP_INFO != requested:
        raise RuntimeError(
            "Tensor-parallel information is already configured as "
            f"rank={_TP_INFO.rank}, size={_TP_INFO.size}; cannot replace it "
            f"with rank={rank}, size={size}.")
    return False


def reset_tp_info(expected: DistributedInfo) -> None:
    """Clear tensor-parallel metadata installed by a finished engine."""
    global _TP_INFO
    if _TP_INFO is None:
        return
    if _TP_INFO != expected:
        raise RuntimeError(
            "Cannot clear tensor-parallel information owned by a different "
            "engine configuration.")
    _TP_INFO = None


def get_tp_info() -> DistributedInfo:
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    return _TP_INFO


__all__ = [
    "DistributedInfo",
    "set_tp_info",
    "reset_tp_info",
    "get_tp_info",
    "try_get_tp_info",
]
