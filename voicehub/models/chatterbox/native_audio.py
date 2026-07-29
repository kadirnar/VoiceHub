"""PyTorch-native audio frontends used by the Chatterbox architecture.

The released English checkpoint was trained with three distinct feature
contracts: librosa/Slaney mel features for S3Tokenizer and the voice
encoder, HiFT's magnitude-mel frontend, and Kaldi filter banks for
CAMPPlus.  Keeping those contracts explicit avoids silently substituting
a generic frontend while removing librosa, SciPy, NumPy, and TorchAudio
from the runtime.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from voicehub.processing.audio import mel_filter_bank
from voicehub.processing.waveform import NativeAudio, load_native_audio, normalize_waveform, resample_waveform


def as_mono_waveform(value: Any, *, device: torch.device | str | None = None) -> Tensor:
    """Normalize one waveform and optionally move it to ``device``."""
    waveform = normalize_waveform(value)
    if device is not None:
        waveform = waveform.to(device)
    return waveform


def load_waveform(
    value: NativeAudio | str | Any,
    *,
    target_sample_rate: int,
    sample_rate: int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Decode/resample a native audio input and return one mono tensor."""
    audio = load_native_audio(
        value,
        sampling_rate=sample_rate,
        target_sampling_rate=target_sample_rate,
    )
    waveform = audio.waveform
    if device is not None:
        waveform = waveform.to(device)
    return waveform


def slaney_mel_filter_bank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the Slaney-normalized bank used by librosa's defaults."""
    return mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        minimum_frequency=fmin,
        maximum_frequency=fmax,
        dtype=dtype,
        device=device,
    )


def s3tokenizer_log_mel(
    audio: Tensor,
    *,
    mel_filters: Tensor,
    window: Tensor,
    n_fft: int = 400,
    hop_length: int = 160,
    padding: int = 0,
) -> Tensor:
    """Compute the exact Whisper-style 128-bin S3Tokenizer frontend."""
    waveform = torch.as_tensor(audio)
    if not waveform.is_floating_point():
        waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("S3Tokenizer audio must have shape [samples] or [batch, samples].")
    if padding > 0:
        waveform = functional.pad(waveform, (0, int(padding)))
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window.to(device=waveform.device, dtype=waveform.dtype),
        return_complex=True,
    )
    magnitudes = spectrum[..., :-1].abs().square()
    mel = mel_filters.to(
        device=waveform.device,
        dtype=magnitudes.dtype,
    ) @ magnitudes
    log_mel = mel.clamp_min(1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.amax() - 8.0)
    return (log_mel + 4.0) / 4.0


def hift_mel_spectrogram(
    audio: Tensor,
    *,
    n_fft: int = 1_920,
    n_mels: int = 80,
    sample_rate: int = 24_000,
    hop_length: int = 480,
    win_length: int = 1_920,
    fmin: float = 0.0,
    fmax: float = 8_000.0,
    center: bool = False,
) -> Tensor:
    """Compute HiFT's released Slaney magnitude-mel representation."""
    waveform = torch.as_tensor(audio)
    if not waveform.is_floating_point():
        waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("HiFT audio must have shape [samples] or [batch, samples].")
    padding = (n_fft - hop_length) // 2
    waveform = functional.pad(
        waveform.unsqueeze(1),
        (padding, padding),
        mode="reflect",
    ).squeeze(1)
    window = torch.hann_window(
        win_length,
        periodic=True,
        dtype=waveform.dtype,
        device=waveform.device,
    )
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = (spectrum.real.square() + spectrum.imag.square() + 1e-9).sqrt()
    filters = slaney_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        dtype=magnitude.dtype,
        device=magnitude.device,
    )
    return (filters @ magnitude).clamp_min(1e-5).log()


def _preemphasis(waveform: Tensor, coefficient: float) -> Tensor:
    if coefficient <= 0.0:
        return waveform
    emphasized = torch.cat((
        waveform[:1],
        waveform[1:] - coefficient * waveform[:-1],
    ), )
    return emphasized.clamp(-1.0, 1.0)


def voice_encoder_mel(
    waveform: Tensor,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    preemphasis: float,
    magnitude_minimum: float,
    mel_power: float,
    normalized: bool,
    mel_type: str = "db",
    center: bool = True,
) -> Tensor:
    """Compute the released voice-encoder mel matrix as ``[frames, mels]``."""
    values = as_mono_waveform(waveform)
    values = _preemphasis(values, preemphasis)
    window = torch.hann_window(
        win_length,
        periodic=True,
        device=values.device,
        dtype=values.dtype,
    )
    spectrum = torch.stft(
        values,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = spectrum.abs()
    if mel_power != 1.0:
        magnitude = magnitude.pow(mel_power)
    filters = slaney_mel_filter_bank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        dtype=magnitude.dtype,
        device=magnitude.device,
    )
    mel = filters @ magnitude
    if mel_type == "db":
        mel = 20.0 * mel.clamp_min(magnitude_minimum).log10()
        if normalized:
            minimum_db = 20.0 * math.log10(magnitude_minimum)
            mel = (mel - minimum_db) / (-minimum_db + 15.0)
    elif mel_type != "amp":
        raise ValueError(f"Unsupported voice-encoder mel type: {mel_type!r}")
    elif normalized:
        raise ValueError("Amplitude mel features cannot use dB normalization.")
    return mel.transpose(0, 1).contiguous().float()


def trim_silence(
    waveform: Tensor,
    *,
    top_db: float = 20.0,
    frame_length: int = 2_048,
    hop_length: int = 512,
) -> Tensor:
    """Match librosa.effects.trim's default frame-energy boundary."""
    values = as_mono_waveform(waveform)
    if values.numel() == 0:
        return values
    padded = functional.pad(
        values,
        (frame_length // 2, frame_length // 2),
    )
    if padded.numel() < frame_length:
        return values
    frames = padded.unfold(0, frame_length, hop_length)
    rms = frames.square().mean(dim=-1).sqrt()
    reference = rms.amax()
    if reference <= 0:
        return values[:0]
    decibels = 20.0 * (rms.clamp_min(torch.finfo(rms.dtype).tiny) / reference).log10()
    non_silent = decibels > -float(top_db)
    indices = non_silent.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return values[:0]
    start = max(0, int(indices[0].item()) * hop_length)
    end = min(
        values.numel(),
        (int(indices[-1].item()) + 1) * hop_length,
    )
    return values[start:end]


def resample_batch(
    waveforms: Sequence[Tensor],
    *,
    source_rate: int,
    target_rate: int,
) -> list[Tensor]:
    """Resample a sequence without importing an external audio package."""
    return [
        resample_waveform(
            as_mono_waveform(waveform),
            source_rate,
            target_rate,
        ) for waveform in waveforms
    ]


__all__ = [
    "as_mono_waveform",
    "hift_mel_spectrogram",
    "load_waveform",
    "resample_batch",
    "s3tokenizer_log_mel",
    "slaney_mel_filter_bank",
    "trim_silence",
    "voice_encoder_mel",
]
