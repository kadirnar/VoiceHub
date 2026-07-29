"""Native WavMark embedding helpers."""

from __future__ import annotations

import time
from typing import Any

import torch

from voicehub.processing.waveform import normalize_waveform

from ..utils import metric_util

# The bits are the immutable synchronization pattern released by WavMark.
fix_pattern = [
    1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
    0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
    0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
    0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
    0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
    0, 0, 1, 0,
]


def _progress(index: int, total: int, *, enabled: bool) -> None:
    if enabled and (index == 0 or index + 1 == total):
        print(f"WavMark embedding segment {index + 1}/{total}")


def add_watermark(
    bit_arr: Any,
    data: Any,
    num_point: int,
    shift_range: float,
    device: str | torch.device,
    model,
    min_snr: float,
    max_snr: float,
    show_progress: bool,
):
    """Embed one payload into every complete synchronization chunk."""
    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean.")
    if not isinstance(num_point, int) or isinstance(num_point, bool) or num_point <= 0:
        raise ValueError("`num_point` must be a positive integer.")
    waveform = normalize_waveform(data)
    started_at = time.monotonic()
    chunk_size = num_point + int(num_point * shift_range)
    if chunk_size <= num_point:
        raise ValueError("`shift_range` must reserve a positive shift area.")
    num_segments = waveform.numel() // chunk_size
    if num_segments == 0:
        raise ValueError(
            f"WavMark requires at least {chunk_size} samples for embedding."
        )
    remainder = waveform.numel() - num_segments * chunk_size
    output_chunks: list[torch.Tensor] = []
    encoded_sections = 0
    skipped_sections = 0

    for index in range(num_segments):
        _progress(index, num_segments, enabled=show_progress)
        start = index * chunk_size
        current = waveform[start:start + chunk_size].clone()
        cover = current[:num_point]
        shift = current[num_point:]
        encoded, state = encode_chunk_with_snr_check(
            index,
            cover,
            bit_arr,
            device,
            model,
            min_snr,
            max_snr,
        )
        if state == "skip":
            skipped_sections += 1
        else:
            encoded_sections += 1
        output_chunks.append(torch.cat((encoded, shift)))

    if remainder:
        output_chunks.append(waveform[-remainder:])
    reconstructed = torch.cat(output_chunks).contiguous()
    if reconstructed.shape != waveform.shape:
        raise RuntimeError("WavMark embedding changed the waveform length.")
    return reconstructed, {
        "time_cost": time.monotonic() - started_at,
        "encoded_sections": encoded_sections,
        "skip_sections": skipped_sections,
    }


def encode_chunk_with_snr_check(
    chunk_index,
    signal,
    watermark,
    device,
    model,
    min_snr,
    max_snr,
):
    source = normalize_waveform(signal)
    candidate = source
    for attempt in range(1, 12):
        encoded = encode_chunk(candidate, watermark, device, model)
        snr = metric_util.signal_noise_ratio(source, encoded)
        if attempt == 1 and snr < min_snr:
            return source, "skip"
        if snr < max_snr or attempt > 10:
            return encoded, attempt
        candidate = encoded
    raise RuntimeError(f"WavMark embedding did not terminate for chunk {chunk_index}.")


def encode_chunk(chunk, watermark, device, model):
    with torch.inference_mode():
        signal = normalize_waveform(chunk).to(device).unsqueeze(0)
        message = torch.as_tensor(
            watermark,
            dtype=signal.dtype,
            device=device,
        ).flatten().unsqueeze(0)
        return model.encode(signal, message).detach().cpu().squeeze(0)


# Preserve the misspelled upstream helper names as compatibility aliases.
encode_trunck_with_snr_check = encode_chunk_with_snr_check
encode_trunck = encode_chunk


__all__ = [
    "add_watermark",
    "encode_chunk",
    "encode_chunk_with_snr_check",
    "encode_trunck",
    "encode_trunck_with_snr_check",
    "fix_pattern",
]
