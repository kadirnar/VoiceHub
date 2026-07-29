"""Model-neutral key/value caches for native autoregressive architectures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheEntry:
    """Key/value state shaped ``[batch, heads, time, head_dimension]``."""

    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.key, Tensor) or not isinstance(self.value, Tensor):
            raise TypeError("Cache entries require PyTorch tensors.")
        if self.key.ndim != 4 or self.value.ndim != 4:
            raise ValueError("Cache tensors must have rank four.")
        if self.key.shape != self.value.shape:
            raise ValueError("Cache key and value shapes must match.")
        if self.key.device != self.value.device or self.key.dtype != self.value.dtype:
            raise ValueError("Cache key and value device/dtype must match.")

    @property
    def sequence_length(self) -> int:
        return self.key.shape[-2]


class DynamicKVCache:
    """Per-layer cache with explicit append, reorder, and detach operations."""

    def __init__(self) -> None:
        self._entries: dict[int, CacheEntry] = {}

    @staticmethod
    def _layer_index(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Cache layer indices must be non-negative integers.")
        return value

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, layer_index: object) -> bool:
        return layer_index in self._entries

    def get(self, layer_index: int) -> CacheEntry | None:
        return self._entries.get(self._layer_index(layer_index))

    def sequence_length(self, layer_index: int = 0) -> int:
        entry = self.get(layer_index)
        return 0 if entry is None else entry.sequence_length

    def update(
        self,
        layer_index: int,
        key: Tensor,
        value: Tensor,
        *,
        append: bool = True,
    ) -> CacheEntry:
        """Append dynamic state or replace a static cross-attention entry."""
        layer_index = self._layer_index(layer_index)
        incoming = CacheEntry(key, value)
        previous = self._entries.get(layer_index)
        if previous is None or not append:
            self._entries[layer_index] = incoming
            return incoming
        if (
            previous.key.shape[:2] != key.shape[:2]
            or previous.key.shape[-1] != key.shape[-1]
            or previous.key.device != key.device
            or previous.key.dtype != key.dtype
        ):
            raise ValueError(
                f"Cache update for layer {layer_index} is incompatible with "
                "existing batch/head/device/dtype state."
            )
        entry = CacheEntry(
            torch.cat((previous.key, key), dim=-2),
            torch.cat((previous.value, value), dim=-2),
        )
        self._entries[layer_index] = entry
        return entry

    def reorder(self, batch_indices: Tensor) -> "DynamicKVCache":
        """Reorder or duplicate cache rows for beam search."""
        if (
            not isinstance(batch_indices, Tensor)
            or batch_indices.ndim != 1
            or batch_indices.dtype == torch.bool
            or batch_indices.is_floating_point()
            or batch_indices.is_complex()
        ):
            raise TypeError("Cache batch indices must be a one-dimensional integer tensor.")
        for layer_index, entry in tuple(self._entries.items()):
            indices = batch_indices.to(entry.key.device)
            self._entries[layer_index] = CacheEntry(
                entry.key.index_select(0, indices),
                entry.value.index_select(0, indices),
            )
        return self

    def detach(self) -> "DynamicKVCache":
        for layer_index, entry in tuple(self._entries.items()):
            self._entries[layer_index] = CacheEntry(
                entry.key.detach(),
                entry.value.detach(),
            )
        return self

    def clear(self) -> None:
        self._entries.clear()

    def clone(self) -> "DynamicKVCache":
        result = DynamicKVCache()
        result._entries = {
            layer_index: CacheEntry(entry.key.clone(), entry.value.clone())
            for layer_index, entry in self._entries.items()
        }
        return result


__all__ = ["CacheEntry", "DynamicKVCache"]
