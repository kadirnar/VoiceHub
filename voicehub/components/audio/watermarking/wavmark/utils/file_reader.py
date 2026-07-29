"""Dependency-free PCM WAVE input helpers for WavMark."""

from __future__ import annotations

from pathlib import Path

import torch

from voicehub.processing.waveform import (
    load_pcm_wave,
    normalize_waveform,
    resample_waveform,
)


def is_wav_file(filename):
    return Path(filename).suffix.lower() == ".wav"


def _load_wave(path: str | Path, target_rate: int) -> tuple[torch.Tensor, int]:
    source = Path(path).expanduser()
    if source.suffix.lower() != ".wav":
        raise ValueError(
            "Native WavMark file loading supports uncompressed PCM WAVE "
            "only. Decode other containers before calling WavMark."
        )
    waveform, sample_rate = load_pcm_wave(source)
    waveform = normalize_waveform(waveform)
    if sample_rate != target_rate:
        waveform = resample_waveform(
            waveform,
            sample_rate,
            target_rate,
        )
    return waveform, sample_rate


def read_as_single_channel_16k(
    audio_file,
    def_sr=16_000,
    verbose=True,
    aim_second=None,
):
    del verbose
    waveform, original_rate = _load_wave(audio_file, def_sr)
    original_duration = waveform.numel() / def_sr
    if aim_second is not None:
        if isinstance(aim_second, bool) or not isinstance(aim_second, (int, float)):
            raise TypeError("`aim_second` must be a positive real number.")
        if aim_second <= 0:
            raise ValueError("`aim_second` must be greater than zero.")
        target_samples = round(def_sr * float(aim_second))
        if waveform.numel() < target_samples:
            repetitions = (
                target_samples + waveform.numel() - 1
            ) // waveform.numel()
            waveform = waveform.repeat(repetitions)
        waveform = waveform[:target_samples]
    del original_rate
    return waveform, def_sr, original_duration


def read_as_single_channel(file, aim_sr):
    waveform, _ = _load_wave(file, aim_sr)
    return waveform


__all__ = [
    "is_wav_file",
    "read_as_single_channel",
    "read_as_single_channel_16k",
]
