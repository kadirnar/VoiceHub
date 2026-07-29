"""Strict native checkpoint loading and official Encodec conversion."""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.checkpointing import SafeTensorReader, save_safetensors

from .artifacts import (
    resolve_encodec_checkpoint,
    verify_official_checkpoint,
)
from .configuration import (
    EncodecConfig,
    encodec_24khz_config,
    encodec_48khz_config,
)
from .metadata import (
    ENCODEC_NATIVE_FORMAT,
    ENCODEC_SOURCE_REVISION,
    EncodecRelease,
    encodec_release,
    normalize_encodec_model_name,
)
from .model import EncodecModel

_CONFIG_METADATA_KEY = "voicehub_config"
_FLOAT_DTYPES = {
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}
_INTEGER_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
}


def _portable_dtype(dtype: torch.dtype) -> str:
    try:
        return (_FLOAT_DTYPES | _INTEGER_DTYPES)[dtype]
    except KeyError as error:
        raise TypeError(f"Unsupported checkpoint tensor dtype {dtype!r}.") from error


def tensor_inventory_fingerprint(tensors: Mapping[str, Tensor]) -> str:
    """Hash sorted tensor names, portable dtypes, and shapes."""
    if not isinstance(tensors, Mapping) or not tensors:
        raise ValueError("Tensor inventory must be a non-empty mapping.")
    rows = []
    for name in sorted(tensors):
        tensor = tensors[name]
        if not isinstance(name, str) or not name:
            raise ValueError("Tensor inventory names must be non-empty strings.")
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Checkpoint entry {name!r} is not a tensor.")
        shape = "x".join(str(value) for value in tensor.shape)
        rows.append(f"{name}|{_portable_dtype(tensor.dtype)}|{shape}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _config_for_name(model_name: str) -> EncodecConfig:
    normalized = normalize_encodec_model_name(model_name)
    return (
        encodec_24khz_config()
        if normalized == "encodec_24khz"
        else encodec_48khz_config()
    )


@lru_cache(maxsize=2)
def _official_shape_items(
    model_name: str,
) -> tuple[tuple[str, tuple[int, ...], torch.dtype], ...]:
    config = _config_for_name(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with torch.device("meta"):
            model = EncodecModel.from_config(config)
    return tuple(
        (name, tuple(tensor.shape), tensor.dtype)
        for name, tensor in model.state_dict().items()
    )


def official_tensor_shapes(model_name: str) -> dict[str, tuple[int, ...]]:
    """Return the exact published state namespace without allocating weights."""
    return {
        name: shape
        for name, shape, _ in _official_shape_items(
            normalize_encodec_model_name(model_name),
        )
    }


def verify_native_graph_contract(model_name: str) -> None:
    """Assert that the native graph still matches the remotely audited release."""
    normalized = normalize_encodec_model_name(model_name)
    release = encodec_release(normalized)
    items = _official_shape_items(normalized)
    rows = [
        f"{name}|{_portable_dtype(dtype)}|{'x'.join(str(value) for value in shape)}"
        for name, shape, dtype in sorted(items)
    ]
    fingerprint = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    count = len(items)
    values = sum(math.prod(shape) for _, shape, _ in items)
    if (
        count != release.tensor_count
        or values != release.state_values
        or fingerprint != release.inventory_fingerprint
    ):
        raise RuntimeError(
            f"Native {normalized} graph no longer matches the audited "
            "official checkpoint contract.")


def _validate_official_state(
    state: Mapping[str, Any],
    release: EncodecRelease,
) -> dict[str, Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Official Encodec checkpoint must be a non-empty mapping.")
    if any(not isinstance(name, str) or not isinstance(value, Tensor)
           for name, value in state.items()):
        raise TypeError("Official Encodec state must map string names to tensors only.")
    tensors = dict(state)
    expected = official_tensor_shapes(release.model_name)
    names = set(tensors)
    expected_names = set(expected)
    if names != expected_names:
        raise ValueError(
            f"Official {release.model_name} tensor namespace mismatch "
            f"(missing={sorted(expected_names - names)!r}, "
            f"unexpected={sorted(names - expected_names)!r}).")
    mismatches = {
        name: (tuple(tensors[name].shape), expected[name])
        for name in sorted(expected)
        if tuple(tensors[name].shape) != expected[name]
    }
    if mismatches:
        raise ValueError(
            f"Official {release.model_name} tensor shape mismatch: {mismatches}.")
    if len(tensors) != release.tensor_count:
        raise ValueError("Official Encodec tensor count mismatch.")
    values = sum(tensor.numel() for tensor in tensors.values())
    if values != release.state_values:
        raise ValueError("Official Encodec state-value count mismatch.")
    fingerprint = tensor_inventory_fingerprint(tensors)
    if fingerprint != release.inventory_fingerprint:
        raise ValueError("Official Encodec tensor inventory fingerprint mismatch.")
    return tensors


def _restricted_official_state(
    source: str | Path,
    release: EncodecRelease,
    *,
    trust_official_pickle: bool,
) -> tuple[dict[str, Tensor], str]:
    if not trust_official_pickle:
        raise PermissionError(
            "Official Encodec `.th` artifacts use PyTorch's legacy pickle "
            "container. Pass `trust_official_pickle=True` only for the exact "
            "pinned Meta release; VoiceHub still enforces "
            "`torch.load(weights_only=True)`, size, digest, namespace, shape, "
            "and inventory checks.")
    path = Path(source).expanduser().resolve()
    digest = verify_official_checkpoint(path, release)
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not read restricted {release.model_name} tensor state.") from error
    return _validate_official_state(state, release), digest


def _validate_reader_against_model(
    reader: SafeTensorReader,
    model: EncodecModel,
    *,
    strict: bool,
) -> tuple[str, ...]:
    state = model.state_dict()
    expected = set(state)
    available = set(reader.keys())
    if strict and available != expected:
        raise ValueError(
            "Encodec Safetensors namespace mismatch "
            f"(missing={sorted(expected - available)!r}, "
            f"unexpected={sorted(available - expected)!r}).")
    selected = tuple(sorted(expected & available))
    if not selected:
        raise ValueError("Encodec Safetensors has no tensors for this graph.")
    mismatches = {
        name: (reader.tensor_shape(name), tuple(state[name].shape))
        for name in selected
        if reader.tensor_shape(name) != tuple(state[name].shape)
    }
    if mismatches:
        raise ValueError(f"Encodec Safetensors tensor shape mismatch: {mismatches}.")
    incompatible_dtypes = {
        name: reader.record(name).dtype
        for name in selected
        if (
            state[name].is_floating_point()
            and reader.record(name).dtype not in {"F16", "BF16", "F32", "F64"}
        )
    }
    if incompatible_dtypes:
        raise ValueError(
            "Encodec Safetensors contains non-floating values for floating "
            f"model state: {incompatible_dtypes}.")
    return selected


def load_encodec_safetensors(
    model: EncodecModel,
    checkpoint: str | Path,
    *,
    strict: bool = True,
) -> EncodecModel:
    """Stream a validated Safetensors checkpoint into an existing graph."""
    if not isinstance(model, EncodecModel):
        raise TypeError("`model` must be an EncodecModel.")
    path = Path(checkpoint).expanduser().resolve()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Native Encodec checkpoints must use `.safetensors`.")
    with SafeTensorReader(path) as reader:
        selected = _validate_reader_against_model(
            reader,
            model,
            strict=strict,
        )
        destination = model.state_dict()
        with torch.no_grad():
            for name in selected:
                target = destination[name]
                source = reader.get_tensor(
                    name,
                    device=target.device,
                    dtype=target.dtype,
                )
                target.copy_(source)
    return model


def save_encodec_safetensors(
    model: EncodecModel,
    checkpoint: str | Path,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Export a complete, deterministic and provider-free Encodec artifact."""
    if not isinstance(model, EncodecModel):
        raise TypeError("`model` must be an EncodecModel.")
    if model.config is None:
        raise ValueError("Encodec export requires the model's construction configuration.")
    path = Path(checkpoint).expanduser()
    if path.suffix.lower() != ".safetensors":
        raise ValueError("Encodec export path must end with `.safetensors`.")
    values = {
        "architecture": "encodec",
        "format": ENCODEC_NATIVE_FORMAT,
        "model_name": model.name,
        "model_dtype": str(next(model.parameters()).dtype),
        "source_revision": ENCODEC_SOURCE_REVISION,
        _CONFIG_METADATA_KEY: json.dumps(
            model.config.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }
    extra_metadata = dict(metadata or {})
    reserved = set(values) & set(extra_metadata)
    if reserved:
        raise ValueError(
            f"Encodec export metadata cannot override reserved keys "
            f"{sorted(reserved)!r}.")
    values.update(extra_metadata)
    output = save_safetensors(
        {
            name: tensor.detach()
            for name, tensor in model.state_dict().items()
        },
        path,
        metadata=values,
    )
    # Re-open the file so malformed or incomplete writes never escape.
    with SafeTensorReader(output) as reader:
        _validate_reader_against_model(reader, model, strict=True)
    return output


def load_encodec_model_from_safetensors(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> EncodecModel:
    """Reconstruct a fresh Encodec graph from one native artifact."""
    path = Path(checkpoint).expanduser().resolve()
    with SafeTensorReader(path) as reader:
        encoded_config = reader.metadata.get(_CONFIG_METADATA_KEY)
        if encoded_config is None:
            raise ValueError(
                "Native Encodec artifact is missing its construction configuration.")
        encoded_dtype = reader.metadata.get("model_dtype")
        try:
            raw_config = json.loads(encoded_config)
        except json.JSONDecodeError as error:
            raise ValueError("Native Encodec configuration metadata is invalid JSON.") from error
    config = EncodecConfig.from_dict(raw_config)
    model = EncodecModel.from_config(config)
    if dtype is None and encoded_dtype is not None:
        dtype = {
            "torch.float16": torch.float16,
            "torch.bfloat16": torch.bfloat16,
            "torch.float32": torch.float32,
            "torch.float64": torch.float64,
        }.get(encoded_dtype)
        if dtype is None:
            raise ValueError(
                f"Native Encodec artifact declares unsupported dtype "
                f"{encoded_dtype!r}.")
    if dtype is not None:
        if dtype not in _FLOAT_DTYPES:
            raise TypeError("Encodec model dtype must be floating-point.")
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)
    load_encodec_safetensors(model, path, strict=True)
    return model.eval()


def convert_official_encodec_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    model_name: str,
    trust_official_pickle: bool = False,
) -> Path:
    """Convert an exact hash-pinned Meta `.th` release to Safetensors."""
    normalized = normalize_encodec_model_name(model_name)
    release = encodec_release(normalized)
    verify_native_graph_contract(normalized)
    state, digest = _restricted_official_state(
        source,
        release,
        trust_official_pickle=trust_official_pickle,
    )
    model = EncodecModel.from_config(_config_for_name(normalized))
    model.load_state_dict(state, strict=True)
    target = Path(destination).expanduser()
    if target.suffix.lower() != ".safetensors":
        target = target / f"{normalized}.safetensors"
    return save_encodec_safetensors(
        model,
        target,
        metadata={
            "source_checkpoint": release.filename,
            "source_sha256": digest,
            "source_inventory_fingerprint": release.inventory_fingerprint,
        },
    )


def load_pretrained_weights(
    model: EncodecModel,
    *,
    repository: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    trust_official_pickle: bool = False,
) -> EncodecModel:
    """Resolve and strictly load weights into an official native graph."""
    release = encodec_release(model.name)
    checkpoint = resolve_encodec_checkpoint(
        release.model_name,
        repository=repository,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    if checkpoint.suffix.lower() == ".safetensors":
        return load_encodec_safetensors(model, checkpoint, strict=True)
    state, _ = _restricted_official_state(
        checkpoint,
        release,
        trust_official_pickle=trust_official_pickle,
    )
    model.load_state_dict(state, strict=True)
    return model


def load_encodec_model(
    model_name: str = "encodec_24khz",
    *,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    trust_official_pickle: bool = False,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
) -> EncodecModel:
    """Build and strictly load an official 24 kHz or 48 kHz Encodec model."""
    normalized = normalize_encodec_model_name(model_name)
    model = EncodecModel.from_config(_config_for_name(normalized))
    if dtype is not None:
        if dtype not in _FLOAT_DTYPES:
            raise TypeError("Encodec model dtype must be floating-point.")
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)
    if checkpoint is None:
        return load_pretrained_weights(
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_official_pickle=trust_official_pickle,
        ).eval()
    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        path = resolve_encodec_checkpoint(
            normalized,
            repository=path,
            local_files_only=True,
        )
    if path.suffix.lower() == ".safetensors":
        return load_encodec_safetensors(model, path, strict=True).eval()
    release = encodec_release(normalized)
    state, _ = _restricted_official_state(
        path,
        release,
        trust_official_pickle=trust_official_pickle,
    )
    model.load_state_dict(state, strict=True)
    return model.eval()


__all__ = [
    "convert_official_encodec_checkpoint",
    "load_encodec_model",
    "load_encodec_model_from_safetensors",
    "load_encodec_safetensors",
    "load_pretrained_weights",
    "official_tensor_shapes",
    "save_encodec_safetensors",
    "tensor_inventory_fingerprint",
    "verify_native_graph_contract",
]
