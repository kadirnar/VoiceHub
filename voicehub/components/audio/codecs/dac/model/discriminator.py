"""PyTorch-native discriminators for Descript Audio Codec training.

The module preserves the published DAC parameter namespace while replacing
AudioTools and einops execution with explicit VoiceHub/PyTorch operations.
STFT padding follows AudioTools' ``match_stride=True`` contract so existing
training recipes retain their frame alignment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import weight_norm

from voicehub.components.audio.codecs._compat import BaseModel
from voicehub.processing.waveform import resample_waveform

BANDS = (
    (0.0, 0.1),
    (0.1, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
)


def _weight_normalized_conv1d(*args, activation: bool = True, **kwargs):
    convolution = weight_norm(nn.Conv1d(*args, **kwargs))
    if not activation:
        return convolution
    return nn.Sequential(convolution, nn.LeakyReLU(0.1))


def _weight_normalized_conv2d(*args, activation: bool = True, **kwargs):
    convolution = weight_norm(nn.Conv2d(*args, **kwargs))
    if not activation:
        return convolution
    return nn.Sequential(convolution, nn.LeakyReLU(0.1))


def _validate_waveform(waveform: Tensor) -> None:
    if not isinstance(waveform, Tensor):
        raise TypeError("DAC discriminator input must be a PyTorch tensor.")
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError(
            "DAC discriminator input must have shape [batch, 1, time]."
        )
    if not waveform.is_floating_point():
        raise TypeError("DAC discriminator input must be floating point.")
    if waveform.shape[-1] < 2:
        raise ValueError("DAC discriminator input must contain at least two samples.")


class MPD(nn.Module):
    """One multi-period waveform discriminator."""

    def __init__(self, period: int) -> None:
        super().__init__()
        if isinstance(period, bool) or not isinstance(period, int) or period < 2:
            raise ValueError("`period` must be an integer of at least two.")
        self.period = period
        self.convs = nn.ModuleList(
            (
                _weight_normalized_conv2d(
                    1,
                    32,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                ),
                _weight_normalized_conv2d(
                    32,
                    128,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                ),
                _weight_normalized_conv2d(
                    128,
                    512,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                ),
                _weight_normalized_conv2d(
                    512,
                    1_024,
                    (5, 1),
                    (3, 1),
                    padding=(2, 0),
                ),
                _weight_normalized_conv2d(
                    1_024,
                    1_024,
                    (5, 1),
                    1,
                    padding=(2, 0),
                ),
            )
        )
        self.conv_post = _weight_normalized_conv2d(
            1_024,
            1,
            kernel_size=(3, 1),
            padding=(1, 0),
            activation=False,
        )

    def pad_to_period(self, waveform: Tensor) -> Tensor:
        sample_count = waveform.shape[-1]
        # Preserve the upstream DAC contract, which appends one complete
        # period even when the input is already divisible.
        padding = self.period - sample_count % self.period
        if padding >= sample_count:
            raise ValueError(
                "The waveform is too short for reflective period padding."
            )
        return functional.pad(waveform, (0, padding), mode="reflect")

    def forward(self, waveform: Tensor) -> list[Tensor]:
        _validate_waveform(waveform)
        hidden = self.pad_to_period(waveform)
        batch, channels, sample_count = hidden.shape
        hidden = hidden.reshape(
            batch,
            channels,
            sample_count // self.period,
            self.period,
        )
        feature_maps = []
        for layer in self.convs:
            hidden = layer(hidden)
            feature_maps.append(hidden)
        hidden = self.conv_post(hidden)
        feature_maps.append(hidden)
        return feature_maps


class MSD(nn.Module):
    """One differentiable multi-scale waveform discriminator."""

    def __init__(self, rate: int = 1, sample_rate: int = 44_100) -> None:
        super().__init__()
        if isinstance(rate, bool) or not isinstance(rate, int) or rate < 1:
            raise ValueError("`rate` must be a positive integer.")
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate < rate
        ):
            raise ValueError("`sample_rate` must be a positive integer >= `rate`.")
        self.convs = nn.ModuleList(
            (
                _weight_normalized_conv1d(1, 16, 15, 1, padding=7),
                _weight_normalized_conv1d(
                    16,
                    64,
                    41,
                    4,
                    groups=4,
                    padding=20,
                ),
                _weight_normalized_conv1d(
                    64,
                    256,
                    41,
                    4,
                    groups=16,
                    padding=20,
                ),
                _weight_normalized_conv1d(
                    256,
                    1_024,
                    41,
                    4,
                    groups=64,
                    padding=20,
                ),
                _weight_normalized_conv1d(
                    1_024,
                    1_024,
                    41,
                    4,
                    groups=256,
                    padding=20,
                ),
                _weight_normalized_conv1d(1_024, 1_024, 5, 1, padding=2),
            )
        )
        self.conv_post = _weight_normalized_conv1d(
            1_024,
            1,
            3,
            1,
            padding=1,
            activation=False,
        )
        self.sample_rate = sample_rate
        self.rate = rate

    def _resample(self, waveform: Tensor) -> Tensor:
        if self.rate == 1:
            return waveform
        target_rate = self.sample_rate // self.rate
        flattened = waveform.reshape(-1, waveform.shape[-1])
        return torch.stack(
            tuple(
                resample_waveform(
                    channel,
                    self.sample_rate,
                    target_rate,
                )
                for channel in flattened
            ),
            dim=0,
        ).reshape(*waveform.shape[:-1], -1)

    def forward(self, waveform: Tensor) -> list[Tensor]:
        _validate_waveform(waveform)
        hidden = self._resample(waveform)
        feature_maps = []
        for layer in self.convs:
            hidden = layer(hidden)
            feature_maps.append(hidden)
        hidden = self.conv_post(hidden)
        feature_maps.append(hidden)
        return feature_maps


def _match_stride_stft(
    waveform: Tensor,
    *,
    window_length: int,
    hop_length: int,
) -> Tensor:
    """Return AudioTools-compatible complex STFT frames."""
    sample_count = waveform.shape[-1]
    right_padding = math.ceil(sample_count / hop_length) * hop_length - sample_count
    side_padding = (window_length - hop_length) // 2
    if side_padding >= sample_count:
        raise ValueError(
            "The waveform is too short for reflective STFT padding."
        )
    padded = functional.pad(
        waveform,
        (side_padding, side_padding + right_padding),
        mode="reflect",
    )
    window = torch.hann_window(
        window_length,
        dtype=waveform.dtype,
        device=waveform.device,
    )
    flattened = padded.reshape(-1, padded.shape[-1])
    spectrum = torch.stft(
        flattened,
        n_fft=window_length,
        hop_length=hop_length,
        window=window,
        return_complex=True,
        center=True,
    )
    spectrum = spectrum[..., 2:-2]
    return spectrum.reshape(
        waveform.shape[0],
        waveform.shape[1],
        spectrum.shape[-2],
        spectrum.shape[-1],
    )


class MRD(nn.Module):
    """One complex multi-band spectrogram discriminator."""

    def __init__(
        self,
        window_length: int,
        hop_factor: float = 0.25,
        sample_rate: int = 44_100,
        bands: Sequence[tuple[float, float]] = BANDS,
    ) -> None:
        super().__init__()
        if (
            isinstance(window_length, bool)
            or not isinstance(window_length, int)
            or window_length < 4
        ):
            raise ValueError("`window_length` must be an integer of at least four.")
        if not math.isfinite(hop_factor) or not 0.0 < hop_factor <= 1.0:
            raise ValueError("`hop_factor` must be finite and in (0, 1].")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("`sample_rate` must be an integer.")
        self.window_length = window_length
        self.hop_factor = float(hop_factor)
        self.sample_rate = sample_rate
        self.hop_length = int(window_length * hop_factor)
        if self.hop_length != window_length // 4:
            raise ValueError(
                "DAC's match-stride spectrogram requires `hop_factor=0.25`."
            )

        frequency_bins = window_length // 2 + 1
        normalized_bands = tuple((float(start), float(stop)) for start, stop in bands)
        if not normalized_bands:
            raise ValueError("`bands` cannot be empty.")
        if any(
            not 0.0 <= start < stop <= 1.0
            for start, stop in normalized_bands
        ):
            raise ValueError("Every frequency band must satisfy 0 <= start < stop <= 1.")
        self.bands = tuple(
            (int(start * frequency_bins), int(stop * frequency_bins))
            for start, stop in normalized_bands
        )

        channels = 32

        def convolution_stack() -> nn.ModuleList:
            return nn.ModuleList(
                (
                    _weight_normalized_conv2d(
                        2,
                        channels,
                        (3, 9),
                        (1, 1),
                        padding=(1, 4),
                    ),
                    _weight_normalized_conv2d(
                        channels,
                        channels,
                        (3, 9),
                        (1, 2),
                        padding=(1, 4),
                    ),
                    _weight_normalized_conv2d(
                        channels,
                        channels,
                        (3, 9),
                        (1, 2),
                        padding=(1, 4),
                    ),
                    _weight_normalized_conv2d(
                        channels,
                        channels,
                        (3, 9),
                        (1, 2),
                        padding=(1, 4),
                    ),
                    _weight_normalized_conv2d(
                        channels,
                        channels,
                        (3, 3),
                        (1, 1),
                        padding=(1, 1),
                    ),
                )
            )

        self.band_convs = nn.ModuleList(
            convolution_stack() for _ in self.bands
        )
        self.conv_post = _weight_normalized_conv2d(
            channels,
            1,
            (3, 3),
            (1, 1),
            padding=(1, 1),
            activation=False,
        )

    def spectrogram(self, waveform: Tensor) -> list[Tensor]:
        _validate_waveform(waveform)
        spectrum = _match_stride_stft(
            waveform,
            window_length=self.window_length,
            hop_length=self.hop_length,
        )
        real = torch.view_as_real(spectrum)
        batch, channels, frequencies, frames, components = real.shape
        real = real.permute(0, 1, 4, 3, 2).reshape(
            batch * channels,
            components,
            frames,
            frequencies,
        )
        return [real[..., start:stop] for start, stop in self.bands]

    def forward(self, waveform: Tensor) -> list[Tensor]:
        band_inputs = self.spectrogram(waveform)
        feature_maps = []
        band_outputs = []
        for band, stack in zip(band_inputs, self.band_convs):
            for layer in stack:
                band = layer(band)
                feature_maps.append(band)
            band_outputs.append(band)
        hidden = self.conv_post(torch.cat(band_outputs, dim=-1))
        feature_maps.append(hidden)
        return feature_maps


class Discriminator(BaseModel):
    """Composite DAC multi-period, multi-scale, and multi-resolution critic."""

    def __init__(
        self,
        rates: Sequence[int] = (),
        periods: Sequence[int] = (2, 3, 5, 7, 11),
        fft_sizes: Sequence[int] = (2_048, 1_024, 512),
        sample_rate: int = 44_100,
        bands: Sequence[tuple[float, float]] = BANDS,
    ) -> None:
        super().__init__()
        discriminators: list[nn.Module] = []
        discriminators.extend(MPD(period) for period in periods)
        discriminators.extend(
            MSD(rate, sample_rate=sample_rate) for rate in rates
        )
        discriminators.extend(
            MRD(
                fft_size,
                sample_rate=sample_rate,
                bands=bands,
            )
            for fft_size in fft_sizes
        )
        if not discriminators:
            raise ValueError("DAC discriminator requires at least one critic.")
        self.discriminators = nn.ModuleList(discriminators)

    @staticmethod
    def preprocess(waveform: Tensor) -> Tensor:
        _validate_waveform(waveform)
        centered = waveform - waveform.mean(dim=-1, keepdim=True)
        peak = centered.abs().amax(dim=-1, keepdim=True)
        return 0.8 * centered / (peak + 1e-9)

    def forward(self, waveform: Tensor) -> list[list[Tensor]]:
        normalized = self.preprocess(waveform)
        return [
            discriminator(normalized)
            for discriminator in self.discriminators
        ]


__all__ = [
    "BANDS",
    "Discriminator",
    "MPD",
    "MRD",
    "MSD",
]
