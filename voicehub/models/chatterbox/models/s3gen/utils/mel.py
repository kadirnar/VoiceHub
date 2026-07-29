"""Native HiFT mel frontend."""

from __future__ import annotations

from torch import Tensor

from voicehub.models.chatterbox.native_audio import hift_mel_spectrogram


def dynamic_range_compression_torch(values: Tensor, C: float = 1.0, clip_val: float = 1e-5) -> Tensor:
    """Apply the log compression used by the released HiFT frontend."""
    return (values.clamp_min(clip_val) * C).log()


def spectral_normalize_torch(magnitudes: Tensor) -> Tensor:
    return dynamic_range_compression_torch(magnitudes)


def mel_spectrogram(
    y: Tensor,
    n_fft: int = 1_920,
    num_mels: int = 80,
    sampling_rate: int = 24_000,
    hop_size: int = 480,
    win_size: int = 1_920,
    fmin: float = 0.0,
    fmax: float = 8_000.0,
    center: bool = False,
) -> Tensor:
    """Compute the checkpoint-compatible Slaney magnitude-mel features."""
    return hift_mel_spectrogram(
        y,
        n_fft=n_fft,
        n_mels=num_mels,
        sample_rate=sampling_rate,
        hop_length=hop_size,
        win_length=win_size,
        fmin=fmin,
        fmax=fmax,
        center=center,
    )


__all__ = ["dynamic_range_compression_torch", "mel_spectrogram", "spectral_normalize_torch"]
