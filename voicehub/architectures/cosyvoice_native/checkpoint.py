"""Strict Safetensors I/O and explicit audited legacy conversion."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from voicehub.architectures.cosyvoice_native.metadata import COSYVOICE3_LEGACY_FILES, NATIVE_COSYVOICE_FORMAT
from voicehub.checkpointing import SafeTensorReader, save_safetensors
from voicehub.checkpointing.errors import CheckpointCompatibilityError, CheckpointIntegrityError

_FLOATING_DTYPES = frozenset({
    "BF16",
    "F16",
    "F32",
    "F64",
    "F8_E4M3",
    "F8_E5M2",
})
_INTEGER_DTYPES = frozenset({
    "BOOL",
    "I8",
    "I16",
    "I32",
    "I64",
    "U8",
    "U16",
    "U32",
    "U64",
})


def tensor_inventory_fingerprint(inventory: Mapping[str, tuple[str, tuple[int, ...]]], ) -> str:
    rows = [
        f"{name}\t{dtype}\t{','.join(map(str, shape))}" for name, (dtype, shape) in sorted(inventory.items())
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CosyVoiceCheckpointReport:
    component: str
    path: Path
    tensor_count: int
    parameter_count: int
    header_fingerprint: str


def inspect_cosyvoice_checkpoint(
    path: str | Path,
    *,
    component: str,
) -> CosyVoiceCheckpointReport:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".safetensors":
        raise ValueError("Native CosyVoice checkpoints must use Safetensors.")
    with SafeTensorReader(source) as reader:
        inventory = {
            name: (
                reader.record(name).dtype,
                reader.tensor_shape(name),
            )
            for name in reader.keys()
        }
        parameters = sum(reader.record(name).number_of_elements for name in reader.keys())
    return CosyVoiceCheckpointReport(
        component=component,
        path=source,
        tensor_count=len(inventory),
        parameter_count=parameters,
        header_fingerprint=tensor_inventory_fingerprint(inventory),
    )


def _validate_layout(
    model: nn.Module,
    reader: SafeTensorReader,
) -> tuple[str, ...]:
    target = model.state_dict(keep_vars=True)
    expected = set(target)
    actual = set(reader.keys())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    shape_mismatches = []
    dtype_mismatches = []
    for name in sorted(expected & actual):
        tensor = target[name]
        source_shape = reader.tensor_shape(name)
        if source_shape != tuple(tensor.shape):
            shape_mismatches.append((name, source_shape, tuple(tensor.shape)))
        dtype = reader.record(name).dtype
        allowed = _FLOATING_DTYPES if tensor.is_floating_point() else _INTEGER_DTYPES
        if dtype not in allowed:
            dtype_mismatches.append((name, dtype))
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise CheckpointCompatibilityError(
            "CosyVoice checkpoint does not match the native component: "
            f"missing={missing[:12]!r}, unexpected={unexpected[:12]!r}, "
            f"shape_mismatches={shape_mismatches[:12]!r}, "
            f"dtype_mismatches={dtype_mismatches[:12]!r}.")
    return tuple(sorted(expected))


def _materialize_runtime_buffers(
    model: nn.Module,
    *,
    device: str | torch.device,
) -> None:
    """Rebuild deterministic non-persistent buffers after a meta load."""
    target_device = torch.device(device)
    for module in model.modules():
        inverse_frequency = module._buffers.get("inverse_frequency")
        if (isinstance(inverse_frequency, Tensor) and inverse_frequency.device.type == "meta" and
                hasattr(module, "dimension") and hasattr(module, "base")):
            dimension = int(module.dimension)
            base = float(module.base)
            exponents = torch.arange(
                0,
                dimension,
                2,
                dtype=torch.float32,
                device=target_device,
            ) / dimension
            module.inverse_frequency = 1.0 / torch.pow(base, exponents)
        stft_window = module._buffers.get("stft_window")
        config = getattr(module, "config", None)
        if (isinstance(stft_window, Tensor) and stft_window.device.type == "meta" and config is not None and
                hasattr(config, "istft_n_fft")):
            module.stft_window = torch.hann_window(
                int(config.istft_n_fft),
                device=target_device,
            )


def validate_cosyvoice_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    component: str,
    require_official_inventory: bool = False,
) -> CosyVoiceCheckpointReport:
    if not isinstance(model, nn.Module):
        raise TypeError("CosyVoice checkpoint target must be a PyTorch module.")
    report = inspect_cosyvoice_checkpoint(path, component=component)
    with SafeTensorReader(report.path) as reader:
        _validate_layout(model, reader)
    if require_official_inventory:
        try:
            expected = COSYVOICE3_LEGACY_FILES[component]
        except KeyError as error:
            raise ValueError("Official inventory exists only for llm, flow, and hift.") from error
        actual = (
            report.tensor_count,
            report.parameter_count,
            report.header_fingerprint,
        )
        required = (
            expected["tensor_count"],
            expected["parameter_count"],
            expected["header_fingerprint"],
        )
        if actual != required:
            raise CheckpointCompatibilityError(
                "CosyVoice component is graph-compatible but does not match "
                f"the audited official inventory: actual={actual!r}, "
                f"expected={required!r}.")
    return report


def load_cosyvoice_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    component: str,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    require_official_inventory: bool = False,
) -> CosyVoiceCheckpointReport:
    report = validate_cosyvoice_checkpoint(
        model,
        path,
        component=component,
        require_official_inventory=require_official_inventory,
    )
    with SafeTensorReader(report.path) as reader:
        names = _validate_layout(model, reader)
        target_state = model.state_dict(keep_vars=True)
        with torch.no_grad():
            for name in names:
                value = reader.get_tensor(name)
                target = target_state[name]
                if dtype is not None and target.is_floating_point():
                    value = value.to(dtype=dtype)
                value = value.to(device=device)
                model.load_state_dict(
                    {name: value},
                    strict=False,
                    assign=True,
                )
    _materialize_runtime_buffers(model, device=device)
    if component == "llm":
        qwen = getattr(getattr(model, "llm", None), "model", None)
        tie_weights = getattr(qwen, "tie_weights", None)
        if callable(tie_weights):
            tie_weights()
    remaining = [name for name, value in model.state_dict().items() if value.device.type == "meta"]
    if remaining:
        raise CheckpointCompatibilityError("CosyVoice load left meta tensors: " + ", ".join(remaining[:12]))
    return report


def export_cosyvoice_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    component: str,
    state_override: Mapping[str, Tensor] | None = None,
) -> Path:
    if not isinstance(model, nn.Module):
        raise TypeError("CosyVoice export requires a PyTorch module.")
    destination = Path(path).expanduser()
    if destination.suffix.lower() != ".safetensors":
        raise ValueError("CosyVoice export path must end in .safetensors.")
    model_state = model.state_dict()
    state = model_state if state_override is None else dict(state_override)
    if set(state) != set(model_state):
        raise ValueError("CosyVoice export state must exactly match the component.")
    for name, value in state.items():
        if tuple(value.shape) != tuple(model_state[name].shape):
            raise ValueError(f"CosyVoice export shape differs for {name!r}.")
    return save_safetensors(
        {
            name: value.detach().cpu().contiguous()
            for name, value in state.items()
        },
        destination,
        metadata={
            "architecture": "cosyvoice",
            "component": component,
            "format": NATIVE_COSYVOICE_FORMAT,
            "producer": "voicehub",
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_audited_cosyvoice_legacy_checkpoint(
    model: nn.Module,
    source: str | Path,
    destination: str | Path,
    *,
    component: str,
) -> Path:
    """Convert one official pickle checkpoint after immutable verification.

    This is deliberately not part of normal model loading.  It accepts
    only the three published files whose exact size and SHA-256 are
    pinned in :mod:`metadata`, uses PyTorch's restricted weights-only
    unpickler, validates the complete key/shape inventory against the
    native graph, then writes a pickle-free Safetensors artifact.
    """
    try:
        expected = COSYVOICE3_LEGACY_FILES[component]
    except KeyError as error:
        raise ValueError("Legacy conversion component must be llm, flow, or hift.") from error
    source = Path(source).expanduser().resolve()
    if source.name != expected["filename"]:
        raise CheckpointIntegrityError(
            f"Expected audited file {expected['filename']!r}, found {source.name!r}.")
    stat = source.stat()
    if stat.st_size != expected["size"]:
        raise CheckpointIntegrityError(f"Legacy {component} file size differs from the audited artifact.")
    actual_hash = _sha256(source)
    if actual_hash != expected["sha256"]:
        raise CheckpointIntegrityError(f"Legacy {component} SHA-256 differs from the audited artifact.")
    try:
        state = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - unsupported old PyTorch
        raise RuntimeError(
            "Audited conversion requires a PyTorch version with "
            "`weights_only=True`; unrestricted pickle loading is forbidden.") from error
    if not isinstance(state, Mapping) or any(not isinstance(name, str) or not isinstance(value, Tensor)
                                             for name, value in state.items()):
        raise CheckpointCompatibilityError(
            "Legacy CosyVoice checkpoint is not a flat tensor state dictionary.")
    model_state = model.state_dict()
    if set(state) != set(model_state):
        raise CheckpointCompatibilityError("Legacy CosyVoice key inventory does not match the native graph.")
    for name, target in model_state.items():
        if tuple(state[name].shape) != tuple(target.shape):
            raise CheckpointCompatibilityError(f"Legacy CosyVoice shape differs for {name!r}.")
    destination = export_cosyvoice_checkpoint(
        model,
        destination,
        component=component,
        state_override=state,
    )
    report = inspect_cosyvoice_checkpoint(
        destination,
        component=component,
    )
    actual_inventory = (
        report.tensor_count,
        report.parameter_count,
        report.header_fingerprint,
    )
    expected_inventory = (
        expected["tensor_count"],
        expected["parameter_count"],
        expected["header_fingerprint"],
    )
    if actual_inventory != expected_inventory:
        destination.unlink(missing_ok=True)
        raise CheckpointCompatibilityError(
            "Converted CosyVoice inventory does not match the immutable audit.")
    return destination


__all__ = [
    "CosyVoiceCheckpointReport",
    "convert_audited_cosyvoice_legacy_checkpoint",
    "export_cosyvoice_checkpoint",
    "inspect_cosyvoice_checkpoint",
    "load_cosyvoice_checkpoint",
    "tensor_inventory_fingerprint",
    "validate_cosyvoice_checkpoint",
]
