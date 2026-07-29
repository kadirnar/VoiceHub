"""Native batched WavMark extraction and synchronization."""

from __future__ import annotations

import time
from typing import Any

import torch

from voicehub.processing.waveform import normalize_waveform


def decode_chunk(chunk, model, device):
    with torch.inference_mode():
        signal = normalize_waveform(chunk).to(device).unsqueeze(0)
        return (model.decode(signal) >= 0.5).int().cpu().squeeze(0)


def extract_watermark_v3_batch(
    data: Any,
    start_bit: Any,
    shift_range: float,
    num_point: int,
    model,
    device,
    batch_size: int = 10,
    shift_range_p: float = 0.5,
    show_progress: bool = False,
):
    if not isinstance(show_progress, bool):
        raise TypeError("`show_progress` must be a boolean.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("`batch_size` must be a positive integer.")
    waveform = normalize_waveform(data)
    pattern = torch.as_tensor(start_bit, dtype=torch.int64).flatten()
    started_at = time.monotonic()
    shift_step = int(shift_range * num_point * shift_range_p)
    if shift_step <= 0:
        raise ValueError("WavMark detection requires a positive shift step.")
    total_detections = max(
        0,
        (waveform.numel() - num_point) // shift_step,
    )
    detect_points = tuple(index * shift_step for index in range(total_detections))
    results = []

    for batch_start in range(0, len(detect_points), batch_size):
        batch_points = detect_points[batch_start:batch_start + batch_size]
        if show_progress and batch_start == 0:
            print(
                "WavMark scanning "
                f"{len(detect_points)} candidate positions"
            )
        current_batch = torch.stack(
            [waveform[position:position + num_point] for position in batch_points],
        ).to(device)
        with torch.inference_mode():
            messages = (model.decode(current_batch) >= 0.5).int().cpu()
        for position, message in zip(batch_points, messages):
            decoded_pattern = message[:pattern.numel()]
            equal = pattern.eq(decoded_pattern)
            if not bool(equal.all()):
                continue
            results.append({
                "sim": 1.0,
                "num_equal_bits": int(equal.sum()),
                "msg": message,
                "start_position": position,
                "start_time_position": position / 16_000,
            })

    info = {
        "time_cost": time.monotonic() - started_at,
        "results": results,
    }
    if not results:
        return None, info
    messages = torch.stack([item["msg"] for item in results]).float()
    return messages.mean(dim=0).ge(0.5).int(), info


# Preserve the misspelled upstream helper name.
decode_trunck = decode_chunk


__all__ = [
    "decode_chunk",
    "decode_trunck",
    "extract_watermark_v3_batch",
]
