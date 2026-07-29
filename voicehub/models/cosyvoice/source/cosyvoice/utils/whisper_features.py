"""OpenAI-compatible log-mel features without the Whisper package."""

from __future__ import annotations

import torch

from voicehub.processing.audio import LogMelSpectrogram


def log_mel_spectrogram(
    audio: torch.Tensor,
    *,
    n_mels: int = 128,
) -> torch.Tensor:
    """Compute the Whisper feature convention used by CosyVoice tokenizers."""
    operation = LogMelSpectrogram(
        sample_rate=16_000,
        n_fft=400,
        hop_length=160,
        n_mels=n_mels,
        dynamic_range=8.0,
        whisper_scaling=True,
    )
    return operation.process({"waveform": audio})["input_features"]


__all__ = ["log_mel_spectrogram"]
