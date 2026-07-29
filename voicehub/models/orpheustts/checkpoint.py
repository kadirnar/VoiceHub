"""Strict identity mapping for the official SNAC Safetensors checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, TensorPlan

REFERENCE_SNAC_CHECKPOINT = {
    "model_id": "hubertsiuzdak/snac_24khz",
    "revision": "c29a77c025506947a7ff15a678787b66b4c2ff47",
    "filename": "model.safetensors",
    "tensor_count": 269,
}
REFERENCE_SNAC_TENSOR_SHAPES = MappingProxyType({
    "decoder.model.0.bias": (768, ),
    "decoder.model.0.parametrizations.weight.original1": (768, 1, 7),
    "encoder.block.0.bias": (48, ),
    "encoder.block.0.parametrizations.weight.original1": (48, 1, 7),
    "encoder.block.5.parametrizations.weight.original1": (768, 1, 7),
    "quantizer.quantizers.0.codebook.weight": (4096, 8),
    "quantizer.quantizers.2.out_proj.parametrizations.weight.original1": (
        768,
        8,
        1,
    ),
})


class SNACCheckpointAdapter(CheckpointAdapter):
    """Load the official conversion into the pinned vendored graph."""

    architecture_id = "orpheus-snac"
    adapter_id = "official-snac-safetensors"
    adapter_version = "1"

    def __init__(self, tensor_names: tuple[str, ...]) -> None:
        if (not isinstance(tensor_names, tuple) or not tensor_names or
                any(not isinstance(name, str) or not name for name in tensor_names) or
                len(set(tensor_names)) != len(tensor_names)):
            raise ValueError("`tensor_names` must be a non-empty unique tuple.")
        self.tensor_names = tuple(sorted(tensor_names))

    @classmethod
    def for_model(cls, model: Any) -> SNACCheckpointAdapter:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("SNAC checkpoint target must expose state_dict().")
        return cls(tuple(state_dict()))

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        del config
        return any(path.suffix == ".safetensors" for path in files)

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        del config
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in self.tensor_names), )


__all__ = [
    "REFERENCE_SNAC_CHECKPOINT",
    "REFERENCE_SNAC_TENSOR_SHAPES",
    "SNACCheckpointAdapter",
]
