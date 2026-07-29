"""VoiceHub-native WavMark encoder and decoder."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from voicehub.hub import resolve_pretrained_file
from voicehub.processing.waveform import normalize_waveform

from .models import my_model
from .utils import file_reader, metric_util, my_parser, path_util, wm_add_util, wm_decode_util

DEFAULT_CHECKPOINT = (
    "step59000_snr39.99_pesq4.35_BERP_none0.30_mean1.81_std1.81.model.pkl"
)


def load_model(path: str | Path = "default"):
    """Load the released WavMark graph with PyTorch's restricted reader."""
    checkpoint_path = (
        resolve_pretrained_file(
            "M4869/WavMark",
            DEFAULT_CHECKPOINT,
        )
        if path == "default"
        else Path(path).expanduser()
    )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"WavMark checkpoint was not found: {checkpoint_path}."
        )
    model = my_model.Model(
        16_000,
        num_bit=32,
        n_fft=1_000,
        hop_length=400,
        num_layers=8,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if isinstance(checkpoint, Mapping) and isinstance(
        checkpoint.get("state_dict"),
        Mapping,
    ):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise ValueError("WavMark checkpoint must contain a tensor state mapping.")
    model.load_state_dict(dict(checkpoint), strict=True)
    return model.eval()


def _binary_payload(payload: Any, *, expected_length: int) -> torch.Tensor:
    values = torch.as_tensor(payload).flatten()
    if values.numel() != expected_length:
        raise ValueError(
            f"WavMark payload must contain {expected_length} bits; found "
            f"{values.numel()}."
        )
    if values.dtype == torch.bool:
        values = values.to(dtype=torch.float32)
    elif not values.eq(values.round()).all() or not values.ge(0).all() or not values.le(1).all():
        raise ValueError("WavMark payload values must be binary zeros or ones.")
    return values.to(dtype=torch.float32)


def encode_watermark(
    model,
    signal,
    payload,
    pattern_bit_length=16,
    min_snr=20,
    max_snr=38,
    show_progress=False,
):
    """Embed a 32-bit synchronization/payload frame into a waveform."""
    if (
        not isinstance(pattern_bit_length, int)
        or isinstance(pattern_bit_length, bool)
        or not 1 <= pattern_bit_length < 32
    ):
        raise ValueError("`pattern_bit_length` must be between 1 and 31.")
    if min_snr > max_snr:
        raise ValueError("`min_snr` cannot exceed `max_snr`.")
    device = next(model.parameters()).device
    pattern = torch.tensor(
        wm_add_util.fix_pattern[:pattern_bit_length],
        dtype=torch.float32,
    )
    message = torch.cat((
        pattern,
        _binary_payload(
            payload,
            expected_length=32 - pattern_bit_length,
        ),
    ))
    waveform = normalize_waveform(signal)
    watermarked, info = wm_add_util.add_watermark(
        message,
        waveform,
        16_000,
        0.1,
        device,
        model,
        min_snr,
        max_snr,
        show_progress=show_progress,
    )
    info["snr"] = metric_util.signal_noise_ratio(waveform, watermarked)
    return watermarked, info


def decode_watermark(
    model,
    signal,
    decode_batch_size=10,
    len_start_bit=16,
    show_progress=False,
):
    """Recover the voted payload tensor, or ``None`` when sync is absent."""
    if (
        not isinstance(len_start_bit, int)
        or isinstance(len_start_bit, bool)
        or not 1 <= len_start_bit < 32
    ):
        raise ValueError("`len_start_bit` must be between 1 and 31.")
    device = next(model.parameters()).device
    start_bit = torch.tensor(
        wm_add_util.fix_pattern[:len_start_bit],
        dtype=torch.int64,
    )
    result, info = wm_decode_util.extract_watermark_v3_batch(
        signal,
        start_bit,
        0.1,
        16_000,
        model,
        device,
        decode_batch_size,
        show_progress=show_progress,
    )
    if result is None:
        return None, info
    return result[len_start_bit:], info


__all__ = [
    "DEFAULT_CHECKPOINT",
    "decode_watermark",
    "encode_watermark",
    "file_reader",
    "load_model",
    "metric_util",
    "my_model",
    "my_parser",
    "path_util",
    "wm_add_util",
    "wm_decode_util",
]
