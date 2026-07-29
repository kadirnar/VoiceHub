"""Tensor-native WavMark quality metrics."""

from __future__ import annotations

import math
from typing import Any

import torch

from voicehub.processing.waveform import normalize_waveform, resample_waveform


def calc_ber(watermark_decoded_tensor, watermark_tensor, threshold=0.5):
    decoded = torch.as_tensor(watermark_decoded_tensor)
    expected = torch.as_tensor(
        watermark_tensor,
        device=decoded.device,
    )
    if decoded.shape != expected.shape:
        raise ValueError(
            "Decoded and expected watermarks must have the same shape."
        )
    decoded_binary = decoded >= threshold
    expected_binary = expected >= threshold
    return 1 - decoded_binary.eq(expected_binary).float().mean()


def to_equal_length(original, signal_watermarked):
    original_tensor = normalize_waveform(original)
    watermarked_tensor = normalize_waveform(signal_watermarked)
    length = min(
        original_tensor.numel(),
        watermarked_tensor.numel(),
    )
    if length == 0:
        raise ValueError("SNR inputs cannot be empty.")
    return original_tensor[:length], watermarked_tensor[:length]


def signal_noise_ratio(original, signal_watermarked):
    original, signal_watermarked = to_equal_length(original, signal_watermarked)
    noise_strength = (original - signal_watermarked).square().sum()
    if float(noise_strength) == 0.0:
        return math.inf
    signal_strength = original.square().sum()
    ratio = (signal_strength / noise_strength).clamp_min(1e-10)
    return float(10.0 * torch.log10(ratio))


def batch_signal_noise_ratio(original, signal_watermarked):
    original_tensor = torch.as_tensor(original)
    watermarked_tensor = torch.as_tensor(signal_watermarked)
    if original_tensor.ndim != 2 or watermarked_tensor.ndim != 2:
        raise ValueError("Batch SNR inputs must have shape [batch, samples].")
    if original_tensor.shape[0] != watermarked_tensor.shape[0]:
        raise ValueError("Batch SNR inputs must have the same batch size.")
    values = [
        signal_noise_ratio(source, encoded)
        for source, encoded in zip(original_tensor, watermarked_tensor)
    ]
    return sum(values) / len(values)


def resample_to16k(data: Any, old_sr: int):
    """Band-limit and resample audio instead of dropping arbitrary samples."""
    return resample_waveform(
        normalize_waveform(data),
        old_sr,
        16_000,
    )
