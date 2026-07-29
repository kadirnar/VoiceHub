"""Strict checkpoint identity mappings for native LLaSA and XCodec2."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, TensorPlan

REFERENCE_LLASSA_CHECKPOINT = MappingProxyType({
    "model_id": "HKUSTAudio/Llasa-1B-Multilingual",
    "revision": "7f094cb62b0a9779b334c60d039a61c5a6e04456",
    "filename": "model.safetensors",
    "tensor_count": 147,
    "vocab_size": 193_800,
})
REFERENCE_XCODEC2_CHECKPOINT = MappingProxyType({
    "model_id":
    "HKUSTAudio/xcodec2-hf",
    "revision":
    "64bd034d12d441299cdd535b15c33efd6ccdf252",
    "filename":
    "model.safetensors",
    "sha256": ("611a63e4dff70c19bd4718d701bb7bc522acf6293a109ab62f5db2f7ff395114"),
    "size":
    2_517_231_448,
    "tensor_count":
    811,
})
REFERENCE_XCODEC2_TENSOR_SHAPES = MappingProxyType({
    "acoustic_decoder.head.linear.weight": (1282, 1024),
    "acoustic_decoder.layers.11.self_attn.q_proj.weight": (1024, 1024),
    "acoustic_encoder.block.4.conv1.weight": (1536, 768, 10),
    "fc_encoder.weight": (2048, 2048),
    "quantizer.project_in.weight": (8, 2048),
    "semantic_encoder.encoder.layers.15.self_attn.distance_embedding.weight": (
        73,
        64,
    ),
    "semantic_encoder.feature_projection.projection.weight": (1024, 160),
})


class XCodec2CheckpointAdapter(CheckpointAdapter):
    """Load the official self-contained conversion without renaming tensors."""

    architecture_id = "llasa-xcodec2"
    adapter_id = "official-xcodec2-hf-safetensors"
    adapter_version = "1"

    def __init__(self, tensor_names: tuple[str, ...]) -> None:
        if (not isinstance(tensor_names, tuple) or not tensor_names or
                any(not isinstance(name, str) or not name for name in tensor_names) or
                len(set(tensor_names)) != len(tensor_names)):
            raise ValueError("`tensor_names` must be a non-empty unique tuple.")
        self.tensor_names = tuple(sorted(tensor_names))

    @classmethod
    def for_model(cls, model: Any) -> XCodec2CheckpointAdapter:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("XCodec2 checkpoint target must expose state_dict().")
        return cls(tuple(state_dict()))

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        return (
            str(config.get("model_type", "")).lower() == "xcodec2" and
            any(path.suffix == ".safetensors" for path in files))

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        del config
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in self.tensor_names))


__all__ = [
    "REFERENCE_LLASSA_CHECKPOINT",
    "REFERENCE_XCODEC2_CHECKPOINT",
    "REFERENCE_XCODEC2_TENSOR_SHAPES",
    "XCodec2CheckpointAdapter",
]
