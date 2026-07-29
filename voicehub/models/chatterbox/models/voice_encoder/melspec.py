"""PyTorch-native frontend for the released Chatterbox voice encoder."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from voicehub.models.chatterbox.native_audio import as_mono_waveform, slaney_mel_filter_bank, voice_encoder_mel


def mel_basis(hp, *, device: torch.device | str | None = None) -> Tensor:
    """Build the Slaney filter bank described by ``VoiceEncConfig``."""
    if hp.fmax > hp.sample_rate // 2:
        raise ValueError("Voice encoder fmax exceeds the Nyquist frequency.")
    return slaney_mel_filter_bank(
        sample_rate=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
        device=device,
    )


def preemphasis(wav, hp) -> Tensor:
    """Apply the source-compatible first-order pre-emphasis filter."""
    values = as_mono_waveform(wav)
    if hp.preemphasis == 0:
        return values
    return torch.cat((values[:1], values[1:] - hp.preemphasis * values[:-1]), ).clamp(-1.0, 1.0)


def melspectrogram(wav, hp, pad: bool = True) -> Tensor:
    """Return one mel spectrogram in the upstream ``[mels, frames]`` layout."""
    values = as_mono_waveform(wav)
    mel = voice_encoder_mel(
        values,
        sample_rate=hp.sample_rate,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax,
        preemphasis=hp.preemphasis,
        magnitude_minimum=hp.stft_magnitude_min,
        mel_power=hp.mel_power,
        normalized=hp.normalized_mels,
        mel_type=hp.mel_type,
        center=pad,
    ).transpose(0, 1)
    if pad and mel.shape[1] != 1 + values.numel() // hp.hop_size:
        raise RuntimeError("Voice encoder mel frontend produced an unexpected frame count.")
    return mel


def _stft(y, hp, pad: bool = True) -> Tensor:
    values = preemphasis(y, hp) if hp.preemphasis > 0 else as_mono_waveform(y)
    window = torch.hann_window(
        hp.win_size,
        periodic=True,
        device=values.device,
        dtype=values.dtype,
    )
    return torch.stft(
        values,
        n_fft=hp.n_fft,
        hop_length=hp.hop_size,
        win_length=hp.win_size,
        window=window,
        center=pad,
        pad_mode="reflect",
        return_complex=True,
    )


def _amp_to_db(values: Tensor, hp) -> Tensor:
    return 20.0 * values.clamp_min(hp.stft_magnitude_min).log10()


def _db_to_amp(values: Tensor) -> Tensor:
    return torch.pow(10.0, values * 0.05)


def _normalize(values: Tensor, hp, headroom_db: float = 15.0) -> Tensor:
    minimum_db = 20.0 * math.log10(hp.stft_magnitude_min)
    return (values - minimum_db) / (-minimum_db + headroom_db)


__all__ = ["mel_basis", "melspectrogram", "preemphasis"]
