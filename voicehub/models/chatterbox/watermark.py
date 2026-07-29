"""VoiceHub-owned runtime for Chatterbox's bundled Perth watermark.

The network topology and released checkpoint are vendored under the
original MIT license.  This module replaces Perth's
TorchAudio/librosa/YAML execution boundary with equivalent PyTorch STFT,
iSTFT, and native resampling.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import torch
from torch import Tensor, nn

from voicehub.processing.waveform import normalize_waveform, resample_waveform


class PerthConfig(NamedTuple):
    sample_rate: int = 32_000
    n_fft: int = 2_048
    hop_size: int = 320
    window_size: int = 2_048
    magnitude_minimum: float = 1e-9
    maximum_watermark_frequency: float = 2_000.0
    hidden_size: int = 256


def _subband_size(config: PerthConfig) -> int:
    bins = config.n_fft // 2 + 1
    nyquist = config.sample_rate / 2
    return int(round(bins * config.maximum_watermark_frequency / nyquist))


def _magnitude_mask(magnitude: Tensor, threshold: float = 0.05) -> Tensor:
    energy = magnitude.sum(dim=1)
    maximum = energy.amax(dim=1)
    return (energy > maximum[:, None] * threshold).float()


class _Conv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *, activation: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.act: nn.Module | bool = nn.LeakyReLU() if activation else False

    def forward(self, values: Tensor) -> Tensor:
        values = self.conv(values)
        return self.act(values) if self.act else values


class _Encoder(nn.Module):

    def __init__(self, hidden_size: int, subband: int):
        super().__init__()
        self.subband = subband
        self.layers = nn.Sequential(
            _Conv(subband, hidden_size, 1),
            *(_Conv(hidden_size, hidden_size, 7) for _ in range(5)),
            _Conv(hidden_size, subband, 1, activation=False),
        )

    def forward(self, magnitude: Tensor) -> tuple[Tensor, Tensor]:
        output = magnitude.clone()
        mask = _magnitude_mask(magnitude)[:, None]
        output[:, :self.subband] += self.layers(magnitude[:, :self.subband]) * mask
        return output, mask


class _AudioProcessor(nn.Module):

    def __init__(self, config: PerthConfig):
        super().__init__()
        self.config = config
        self.register_buffer(
            "window",
            torch.hann_window(
                config.window_size,
                periodic=True,
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def signal_to_magphase(self, signal: Tensor) -> tuple[Tensor, Tensor]:
        waveform = normalize_waveform(signal).float()
        spectrum = torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            win_length=self.config.window_size,
            window=self.window.to(waveform),
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        magnitude = 20.0 * spectrum.abs().clamp_min(self.config.magnitude_minimum).log10()
        minimum_db = 20.0 * math.log10(self.config.magnitude_minimum)
        magnitude = (magnitude - minimum_db) / (-minimum_db + 15.0)
        return magnitude, spectrum.angle()

    def magphase_to_signal(self, magnitude: Tensor, phase: Tensor) -> Tensor:
        minimum_db = 20.0 * math.log10(self.config.magnitude_minimum)
        magnitude_db = magnitude * (-minimum_db + 15.0) + minimum_db
        linear_magnitude = torch.pow(
            10.0,
            (magnitude_db / 20.0).clamp_max(10.0),
        )
        spectrum = torch.polar(linear_magnitude, phase)
        return torch.istft(
            spectrum,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_size,
            win_length=self.config.window_size,
            window=self.window.to(magnitude),
            center=True,
            normalized=False,
            onesided=True,
            return_complex=False,
        )


class NativePerthWatermarker:
    """Apply the exact bundled Perth implicit encoder in PyTorch."""

    SOURCE_REVISION = "ce86c49d029f42272c1902eccb675556b9ed2330"

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.config = PerthConfig()
        self.device = torch.device(device)
        self.audio_processor = _AudioProcessor(self.config).to(self.device)
        self.encoder = _Encoder(
            self.config.hidden_size,
            _subband_size(self.config),
        ).to(self.device)
        source = (
            Path(checkpoint_path) if checkpoint_path is not None else Path(__file__).parent / "source" /
            "perth" / "perth_net" / "pretrained" / "implicit" / "perth_net_250000.pth.tar")
        if not source.is_file():
            raise FileNotFoundError(f"Bundled Perth checkpoint was not found: {source}.")
        checkpoint = torch.load(
            source,
            map_location="cpu",
            weights_only=True,
        )
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ValueError("Bundled Perth checkpoint has no model state dictionary.")
        prefix = "encoder."
        encoder_state = {
            name[len(prefix):]: value
            for name, value in state.items() if name.startswith(prefix)
        }
        incompatible = self.encoder.load_state_dict(encoder_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError("Bundled Perth encoder inventory does not match its topology.")
        self.encoder.eval()

    @torch.inference_mode()
    def apply_watermark(
        self,
        signal: Tensor,
        *,
        sample_rate: int,
    ) -> Tensor:
        waveform = normalize_waveform(signal).to(self.device)
        original_rate = int(sample_rate)
        if original_rate != self.config.sample_rate:
            waveform = resample_waveform(
                waveform,
                original_rate,
                self.config.sample_rate,
            )
        magnitude, phase = self.audio_processor.signal_to_magphase(waveform)
        watermarked_magnitude, _ = self.encoder(magnitude.unsqueeze(0))
        watermarked = self.audio_processor.magphase_to_signal(
            watermarked_magnitude[0],
            phase,
        )
        if original_rate != self.config.sample_rate:
            watermarked = resample_waveform(
                watermarked,
                self.config.sample_rate,
                original_rate,
            )
        return watermarked.detach().cpu()


__all__ = ["NativePerthWatermarker", "PerthConfig"]
