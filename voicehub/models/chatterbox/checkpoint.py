"""Strict native checkpoint I/O for the Chatterbox family."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from voicehub.checkpointing import SafeTensorReader, save_safetensors

CHECKPOINT_REVISION = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
CHECKPOINT_REPOSITORY = "ResembleAI/chatterbox"
CHECKPOINT_LICENSE = "MIT"
FORMAT_NAME = "voicehub-native-chatterbox-v1"
FORMAT_VERSION = 1


def inspect_t3_text_vocabulary_size(path: str | Path) -> int:
    """Read the T3 vocabulary size from checkpoint metadata only."""
    try:
        with SafeTensorReader(path) as reader:
            embedding_shape = reader.tensor_shape("text_emb.weight")
            head_shape = reader.tensor_shape("text_head.weight")
    except KeyError as error:
        raise ValueError(
            "Chatterbox T3 checkpoint is missing its text vocabulary "
            "matrices."
        ) from error
    if len(embedding_shape) != 2 or len(head_shape) != 2:
        raise ValueError(
            "Chatterbox T3 text embedding and head must both be matrices."
        )
    if embedding_shape != head_shape:
        raise ValueError(
            "Chatterbox T3 text embedding and head shapes do not match: "
            f"{embedding_shape!r} != {head_shape!r}."
        )
    vocabulary_size = int(embedding_shape[0])
    if vocabulary_size <= 0:
        raise ValueError(
            "Chatterbox T3 checkpoint declares an invalid text vocabulary."
        )
    return vocabulary_size


def read_safetensors(path: str | Path) -> dict[str, Tensor]:
    """Read a validated Safetensors file without the safetensors package."""
    with SafeTensorReader(path) as reader:
        return reader.state_dict()


def load_module_safetensors(
    module: nn.Module,
    path: str | Path,
    *,
    allowed_missing: Iterable[str] = (),
    allowed_unexpected: Iterable[str] = (),
) -> None:
    """Load a module while enforcing a reviewed tensor inventory."""
    state = read_safetensors(path)
    allowed_missing = frozenset(allowed_missing)
    allowed_unexpected = frozenset(allowed_unexpected)
    expected = module.state_dict()
    missing = tuple(
        name for name in expected
        if name not in state
        if name not in allowed_missing
    )
    unexpected = tuple(
        name for name in state
        if name not in expected
        if name not in allowed_unexpected
    )
    shape_mismatches = tuple(
        (
            name,
            tuple(state[name].shape),
            tuple(expected[name].shape),
        )
        for name in state.keys() & expected.keys()
        if tuple(state[name].shape) != tuple(expected[name].shape)
    )
    if missing or unexpected or shape_mismatches:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing[:20]))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected[:20]))
        if shape_mismatches:
            details.append(
                "shape mismatches: "
                + ", ".join(
                    f"{name}={actual!r}, expected {wanted!r}"
                    for name, actual, wanted in shape_mismatches[:20]
                )
            )
        raise ValueError(
            f"Chatterbox checkpoint inventory mismatch for {Path(path).name}: "
            + "; ".join(details)
        )
    incompatible = module.load_state_dict(state, strict=False)
    unresolved_missing = tuple(
        name for name in incompatible.missing_keys
        if name not in allowed_missing
    )
    unresolved_unexpected = tuple(
        name for name in incompatible.unexpected_keys
        if name not in allowed_unexpected
    )
    if unresolved_missing or unresolved_unexpected:
        raise RuntimeError(
            "Chatterbox checkpoint changed during validated loading."
        )


def export_module_safetensors(
    module: nn.Module,
    path: str | Path,
    *,
    component: str,
    state_dict: Mapping[str, Tensor] | None = None,
) -> Path:
    """Write a deterministic component checkpoint with provenance metadata."""
    source_state = module.state_dict() if state_dict is None else state_dict
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in source_state.items()
    }
    return save_safetensors(
        state,
        path,
        metadata={
            "format": FORMAT_NAME,
            "format_version": str(FORMAT_VERSION),
            "component": component,
            "source_repository": CHECKPOINT_REPOSITORY,
            "source_revision": CHECKPOINT_REVISION,
            "license": CHECKPOINT_LICENSE,
        },
    )


def export_chatterbox_runtime(
    runtime: Any,
    directory: str | Path,
    *,
    t3_state_dict: Mapping[str, Tensor] | None = None,
) -> Path:
    """Export a complete inference-reloadable English Chatterbox artifact."""
    destination = Path(directory).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    export_module_safetensors(
        runtime.ve,
        destination / "ve.safetensors",
        component="voice_encoder",
    )
    export_module_safetensors(
        runtime.t3,
        destination / "t3_cfg.safetensors",
        component="t3",
        state_dict=t3_state_dict,
    )
    export_module_safetensors(
        runtime.s3gen,
        destination / "s3gen.safetensors",
        component="s3gen",
    )
    tokenizer_path = getattr(runtime.tokenizer, "asset_path", None)
    if tokenizer_path is None:
        raise ValueError("Chatterbox export requires the original tokenizer.json asset.")
    tokenizer_source = Path(tokenizer_path).expanduser().resolve()
    tokenizer_destination = (destination / "tokenizer.json").resolve()
    if tokenizer_source != tokenizer_destination:
        shutil.copyfile(tokenizer_source, tokenizer_destination)
    if runtime.conds is not None:
        runtime.conds.save(destination / "conds.pt")
    else:
        (destination / "conds.pt").unlink(missing_ok=True)
    manifest = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "architecture": "chatterbox-english-520m",
        "checkpoint": {
            "repository": CHECKPOINT_REPOSITORY,
            "revision": CHECKPOINT_REVISION,
            "license": CHECKPOINT_LICENSE,
        },
        "components": {
            "t3": "source-faithful-token-cross-entropy",
            "s3gen": "source-faithful-causal-flow-matching",
            "voice_encoder": "inference-conditioner",
            "s3tokenizer": "frozen-audio-tokenizer",
            "watermark": "native-perth-implicit",
        },
        "t3": {
            "text_vocabulary_size": int(
                runtime.t3.text_emb.num_embeddings
            ),
        },
        "training_boundary": {
            "author_recipe_published": False,
            "published_objectives": [
                "T3 text and speech token cross entropy",
                "S3Gen conditional flow matching",
            ],
            "accepts_raw_audio": True,
            "precomputed_supervision_supported": True,
        },
    }
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    (destination / "voicehub_chatterbox.json").write_text(
        encoded,
        encoding="utf-8",
    )
    return destination


__all__ = [
    "CHECKPOINT_LICENSE",
    "CHECKPOINT_REPOSITORY",
    "CHECKPOINT_REVISION",
    "FORMAT_NAME",
    "export_chatterbox_runtime",
    "export_module_safetensors",
    "inspect_t3_text_vocabulary_size",
    "load_module_safetensors",
    "read_safetensors",
]
