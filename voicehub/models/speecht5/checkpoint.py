"""Strict checkpoint adaptation and portable export for native SpeechT5."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voicehub.checkpointing import CheckpointAdapter, CopyTensor, SafeTensorReader, TensorPlan, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError
from voicehub.models.speecht5.metadata import (
    NATIVE_SPEECHT5_FORMAT,
    SPEECHT5_HIFIGAN_STATE_VALUES,
    SPEECHT5_HIFIGAN_TENSOR_COUNT,
    SPEECHT5_HIFIGAN_TENSOR_FINGERPRINT,
    SPEECHT5_STATE_VALUES,
    SPEECHT5_TENSOR_COUNT,
    SPEECHT5_TENSOR_FINGERPRINT,
)

_PORTABLE_DTYPES = {
    "torch.bool": "BOOL",
    "torch.uint8": "U8",
    "torch.int8": "I8",
    "torch.int16": "I16",
    "torch.int32": "I32",
    "torch.int64": "I64",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float32": "F32",
    "torch.float64": "F64",
}
_FLOATING_DTYPES = frozenset({
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E5M2",
})


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    """Hash sorted ``name|portable-dtype|dimxdim`` inventory rows."""
    if not isinstance(inventory, Mapping):
        raise TypeError("`inventory` must be a mapping.")
    rows: list[str] = []
    for name, record in sorted(inventory.items()):
        if (not isinstance(name, str) or not name or not isinstance(record, tuple) or len(record) != 2):
            raise ValueError("Inventory entries must be tensor-name/(dtype, shape) pairs.")
        dtype, shape = record
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"Tensor {name!r} has an invalid dtype.")
        if (not isinstance(shape, tuple) or
                any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)):
            raise ValueError(f"Tensor {name!r} has an invalid shape.")
        rows.append(f"{name}|{dtype}|{'x'.join(str(value) for value in shape)}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def state_dict_inventory(state_dict: Mapping[str, Any], ) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Build a portable tensor inventory without reading tensor values."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("SpeechT5 checkpoint must contain a non-empty state dict.")
    inventory = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not name:
            raise TypeError("SpeechT5 checkpoint tensor names must be non-empty.")
        try:
            dtype = _PORTABLE_DTYPES[str(tensor.dtype)]
            shape = tuple(tensor.shape)
        except (AttributeError, KeyError, TypeError) as error:
            raise TypeError(f"SpeechT5 checkpoint value {name!r} is not a supported tensor.") from error
        inventory[name] = dtype, shape
    return inventory


@dataclass(frozen=True, slots=True)
class SpeechT5CheckpointReport:
    """Complete tensor-header summary for one SpeechT5 checkpoint."""

    path: Path
    tensor_count: int
    state_values: int
    tensor_fingerprint: str


def _report(
    path: Path,
    inventory: Mapping[str, tuple[str, tuple[int, ...]]],
) -> SpeechT5CheckpointReport:
    state_values = 0
    for _, shape in inventory.values():
        count = 1
        for dimension in shape:
            count *= dimension
        state_values += count
    return SpeechT5CheckpointReport(
        path=path,
        tensor_count=len(inventory),
        state_values=state_values,
        tensor_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def inspect_safetensors_checkpoint(path: str | Path, ) -> SpeechT5CheckpointReport:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Expected a SpeechT5 Safetensors checkpoint.")
    with SafeTensorReader(source) as reader:
        inventory = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in reader.keys()}
    return _report(source, inventory)


def load_restricted_pytorch_state(path: str | Path, ) -> Mapping[str, Any]:
    """Load the released PyTorch archive through the restricted unpickler.

    VoiceHub deliberately has no permissive fallback for older PyTorch
    versions.  A runtime that cannot enforce ``weights_only=True`` must
    first convert the archive in a controlled environment.
    """
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - invariant
        raise RuntimeError("Native SpeechT5 checkpoint loading requires PyTorch.") from error
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".bin":
        raise ValueError("Restricted SpeechT5 source archives must use .bin.")
    parameters = inspect.signature(torch.load).parameters
    if "weights_only" not in parameters:
        raise RuntimeError(
            "This PyTorch version cannot enforce restricted checkpoint "
            "loading. Upgrade PyTorch or use a converted Safetensors bundle.")
    try:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(payload, Mapping) or not payload:
        raise TypeError("SpeechT5 .bin checkpoint must contain a non-empty tensor mapping.")
    state_dict_inventory(payload)
    return payload


class SpeechT5CheckpointAdapter(CheckpointAdapter):
    """Strict identity adapter for the released SpeechT5 namespace."""

    architecture_id = "speecht5-text-to-speech"
    adapter_id = "official-speecht5-identity"
    adapter_version = "1"

    def __init__(self, tensor_names: tuple[str, ...]) -> None:
        if (not isinstance(tensor_names, tuple) or not tensor_names or
                len(tensor_names) != len(set(tensor_names)) or
                any(not isinstance(name, str) or not name for name in tensor_names)):
            raise ValueError("`tensor_names` must be a non-empty unique tuple.")
        self.tensor_names = tuple(sorted(tensor_names))

    @classmethod
    def for_model(cls, model: Any) -> SpeechT5CheckpointAdapter:
        state_dict = getattr(model, "state_dict", None)
        if not callable(state_dict):
            raise TypeError("SpeechT5 checkpoint target must expose state_dict().")
        return cls(tuple(state_dict()))

    def probe(
        self,
        files: tuple[Path, ...],
        config: Mapping[str, Any],
    ) -> bool:
        del config
        return len(files) == 1 and files[0].suffix.lower() in {
            ".bin",
            ".safetensors",
        }

    def tensor_plan(self, config: Mapping[str, Any]) -> TensorPlan:
        del config
        return TensorPlan(rules=tuple(CopyTensor(name, name) for name in self.tensor_names))


class SpeechT5HifiGanCheckpointAdapter(SpeechT5CheckpointAdapter):
    """Strict identity adapter for the released HiFi-GAN namespace."""

    architecture_id = "speecht5-hifigan"
    adapter_id = "official-speecht5-hifigan-identity"


def _expected_inventory(model: Any, ) -> dict[str, tuple[str, tuple[int, ...]]]:
    return state_dict_inventory(model.state_dict())


def _validate_dtypes(
    expected: Mapping[str, tuple[str, tuple[int, ...]]],
    actual: Mapping[str, tuple[str, tuple[int, ...]]],
) -> None:
    incompatible = []
    for name in sorted(set(expected) & set(actual)):
        expected_dtype = expected[name][0]
        actual_dtype = actual[name][0]
        compatible = (
            expected_dtype == actual_dtype or
            (expected_dtype in _FLOATING_DTYPES and actual_dtype in _FLOATING_DTYPES))
        if not compatible:
            incompatible.append((name, actual_dtype, expected_dtype))
    if incompatible:
        raise CheckpointCompatibilityError(
            "SpeechT5 checkpoint contains incompatible tensor dtypes: "
            f"{incompatible[:12]!r}.")


def _require_official(
    report: SpeechT5CheckpointReport,
    *,
    vocoder: bool,
) -> None:
    expected = ((
        SPEECHT5_HIFIGAN_TENSOR_COUNT,
        SPEECHT5_HIFIGAN_STATE_VALUES,
        SPEECHT5_HIFIGAN_TENSOR_FINGERPRINT,
    ) if vocoder else (
        SPEECHT5_TENSOR_COUNT,
        SPEECHT5_STATE_VALUES,
        SPEECHT5_TENSOR_FINGERPRINT,
    ))
    actual = (
        report.tensor_count,
        report.state_values,
        report.tensor_fingerprint,
    )
    if actual != expected:
        owner = "SpeechT5 HiFi-GAN" if vocoder else "SpeechT5"
        raise CheckpointCompatibilityError(
            f"{owner} checkpoint matches neither the audited tensor count, "
            "state-value count, nor namespace fingerprint.")


def load_speecht5_checkpoint(
    model: Any,
    path: str | Path,
    *,
    vocoder: bool = False,
    require_official_inventory: bool = False,
) -> SpeechT5CheckpointReport:
    """Validate every name, shape, and dtype before mutating ``model``."""
    source = Path(path).expanduser().resolve()
    adapter_type = (SpeechT5HifiGanCheckpointAdapter if vocoder else SpeechT5CheckpointAdapter)
    adapter = adapter_type.for_model(model)
    expected = _expected_inventory(model)
    if source.suffix.lower() == ".safetensors":
        with SafeTensorReader(source) as reader:
            actual = {name: (reader.record(name).dtype, reader.tensor_shape(name)) for name in reader.keys()}
            report = _report(source, actual)
            if require_official_inventory:
                _require_official(report, vocoder=vocoder)
            _validate_dtypes(expected, actual)
            adapter.load_streaming(model, reader, {}, strict=True)
        return report
    if source.suffix.lower() != ".bin":
        raise ValueError("Native SpeechT5 accepts .safetensors or restricted .bin checkpoints.")
    state = load_restricted_pytorch_state(source)
    actual = state_dict_inventory(state)
    report = _report(source, actual)
    if require_official_inventory:
        _require_official(report, vocoder=vocoder)
    _validate_dtypes(expected, actual)
    adapter.load(model, state, {}, strict=True)
    return report


def save_speecht5_checkpoint(
    model: Any,
    path: str | Path,
    *,
    vocoder: bool = False,
) -> Path:
    """Write one deterministic, dependency-free portable checkpoint."""
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("SpeechT5 export target must expose state_dict().")
    return save_safetensors(
        state_dict(),
        path,
        metadata={
            "architecture": ("speecht5-hifigan" if vocoder else "speecht5-text-to-speech"),
            "format": NATIVE_SPEECHT5_FORMAT,
            "source": "voicehub-native",
        },
    )


__all__ = [
    "SpeechT5CheckpointAdapter",
    "SpeechT5CheckpointReport",
    "SpeechT5HifiGanCheckpointAdapter",
    "inspect_safetensors_checkpoint",
    "load_restricted_pytorch_state",
    "load_speecht5_checkpoint",
    "save_speecht5_checkpoint",
    "state_dict_inventory",
    "tensor_inventory_fingerprint",
]
