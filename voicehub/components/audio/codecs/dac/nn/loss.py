"""Native reconstruction and adversarial objectives for DAC training."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.components.audio.codecs._compat import AudioSignal
from voicehub.processing.audio import mel_filter_bank


def _audio_tensor(value: Tensor | AudioSignal, *, name: str) -> Tensor:
    tensor = value.audio_data if isinstance(value, AudioSignal) else value
    if not isinstance(tensor, Tensor):
        raise TypeError(f"`{name}` must be a PyTorch tensor or AudioSignal.")
    if tensor.ndim != 3:
        raise ValueError(f"`{name}` must have shape [batch, channels, time].")
    if not tensor.is_floating_point():
        raise TypeError(f"`{name}` must use a floating-point dtype.")
    if tensor.shape[-1] == 0:
        raise ValueError(f"`{name}` cannot be empty.")
    return tensor


def _sample_rate(
    first: Tensor | AudioSignal,
    second: Tensor | AudioSignal,
    fallback: int | None,
) -> int:
    rates = {
        value.sample_rate
        for value in (first, second)
        if isinstance(value, AudioSignal)
    }
    if len(rates) > 1:
        raise ValueError("Compared AudioSignal values must use one sample rate.")
    resolved = next(iter(rates), fallback)
    if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved <= 0:
        raise ValueError(
            "A positive `sample_rate` is required for tensor mel losses."
        )
    return resolved


def _window(
    kind: str | None,
    length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    normalized = "hann" if kind is None else kind.strip().lower().replace("-", "_")
    if normalized == "hann":
        return torch.hann_window(length, dtype=dtype, device=device)
    if normalized == "sqrt_hann":
        return torch.hann_window(
            length,
            dtype=dtype,
            device=device,
        ).clamp_min(0.0).sqrt()
    if normalized == "hamming":
        return torch.hamming_window(length, dtype=dtype, device=device)
    if normalized == "blackman":
        return torch.blackman_window(length, dtype=dtype, device=device)
    if normalized == "average":
        return torch.full(
            (length,),
            1.0 / length,
            dtype=dtype,
            device=device,
        )
    raise ValueError(f"Unsupported native STFT window {kind!r}.")


@dataclass(frozen=True, slots=True)
class STFTParameters:
    """One validated spectral-loss resolution."""

    window_length: int
    hop_length: int
    match_stride: bool = False
    window_type: str | None = None

    def __post_init__(self) -> None:
        for name in ("window_length", "hop_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
        if self.hop_length > self.window_length:
            raise ValueError("`hop_length` cannot exceed `window_length`.")
        if not isinstance(self.match_stride, bool):
            raise TypeError("`match_stride` must be a boolean.")
        if self.match_stride and self.hop_length != self.window_length // 4:
            raise ValueError(
                "Match-stride STFT requires `hop_length=window_length//4`."
            )


def _stft_magnitude(audio: Tensor, parameters: STFTParameters) -> Tensor:
    batch, channels, sample_count = audio.shape
    materialized = audio
    if parameters.match_stride:
        right_padding = (
            math.ceil(sample_count / parameters.hop_length)
            * parameters.hop_length
            - sample_count
        )
        side_padding = (
            parameters.window_length - parameters.hop_length
        ) // 2
        if side_padding >= sample_count:
            raise ValueError(
                "Audio is too short for reflective match-stride STFT padding."
            )
        materialized = functional.pad(
            materialized,
            (side_padding, side_padding + right_padding),
            mode="reflect",
        )
    flattened = materialized.reshape(-1, materialized.shape[-1])
    spectrum = torch.stft(
        flattened,
        n_fft=parameters.window_length,
        hop_length=parameters.hop_length,
        window=_window(
            parameters.window_type,
            parameters.window_length,
            dtype=audio.dtype,
            device=audio.device,
        ),
        center=True,
        return_complex=True,
    )
    if parameters.match_stride:
        spectrum = spectrum[..., 2:-2]
    return spectrum.abs().reshape(
        batch,
        channels,
        spectrum.shape[-2],
        spectrum.shape[-1],
    )


class L1Loss(nn.L1Loss):
    """L1 distance between tensors or a selected AudioSignal attribute."""

    def __init__(
        self,
        attribute: str = "audio_data",
        weight: float = 1.0,
        **kwargs,
    ) -> None:
        if not isinstance(attribute, str) or not attribute:
            raise ValueError("`attribute` must be a non-empty string.")
        if not math.isfinite(weight):
            raise ValueError("`weight` must be finite.")
        self.attribute = attribute
        self.weight = float(weight)
        super().__init__(**kwargs)

    def forward(
        self,
        first: Tensor | AudioSignal,
        second: Tensor | AudioSignal,
    ) -> Tensor:
        if isinstance(first, AudioSignal):
            first = getattr(first, self.attribute)
        if isinstance(second, AudioSignal):
            second = getattr(second, self.attribute)
        return super().forward(first, second)


class SISDRLoss(nn.Module):
    """Scale-invariant source-to-distortion objective."""

    def __init__(
        self,
        scaling: bool = True,
        reduction: str = "mean",
        zero_mean: bool = True,
        clip_min: float | None = None,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("`reduction` must be 'mean', 'sum', or 'none'.")
        for name, value in (("scaling", scaling), ("zero_mean", zero_mean)):
            if not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")
        if clip_min is not None and not math.isfinite(clip_min):
            raise ValueError("`clip_min` must be finite or None.")
        if not math.isfinite(weight):
            raise ValueError("`weight` must be finite.")
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = float(weight)

    def forward(
        self,
        references: Tensor | AudioSignal,
        estimates: Tensor | AudioSignal,
    ) -> Tensor:
        reference_tensor = _audio_tensor(references, name="references")
        estimate_tensor = _audio_tensor(estimates, name="estimates")
        if reference_tensor.shape != estimate_tensor.shape:
            raise ValueError("SI-SDR inputs must have identical shapes.")
        batch = reference_tensor.shape[0]
        reference_tensor = reference_tensor.reshape(batch, -1, 1)
        estimate_tensor = estimate_tensor.reshape(batch, -1, 1)
        if self.zero_mean:
            reference_tensor = (
                reference_tensor
                - reference_tensor.mean(dim=1, keepdim=True)
            )
            estimate_tensor = (
                estimate_tensor
                - estimate_tensor.mean(dim=1, keepdim=True)
            )
        epsilon = 1e-8
        projection_power = reference_tensor.square().sum(dim=1) + epsilon
        correlation = (
            estimate_tensor * reference_tensor
        ).sum(dim=1) + epsilon
        scale: Tensor | float = (
            (correlation / projection_power).unsqueeze(1)
            if self.scaling
            else 1.0
        )
        target = scale * reference_tensor
        residual = estimate_tensor - target
        signal = target.square().sum(dim=1)
        noise = residual.square().sum(dim=1)
        loss = -10.0 * torch.log10(signal / noise.clamp_min(epsilon) + epsilon)
        if self.clip_min is not None:
            loss = torch.clamp(loss, min=self.clip_min)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiScaleSTFTLoss(nn.Module):
    """Multi-resolution magnitude and log-magnitude STFT objective."""

    def __init__(
        self,
        window_lengths: Sequence[int] = (2_048, 512),
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str | None = None,
    ) -> None:
        super().__init__()
        lengths = tuple(window_lengths)
        if not lengths:
            raise ValueError("`window_lengths` cannot be empty.")
        self.stft_params = tuple(
            STFTParameters(
                window_length=length,
                hop_length=length // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for length in lengths
        )
        self.loss_fn = nn.L1Loss() if loss_fn is None else loss_fn
        if not callable(self.loss_fn):
            raise TypeError("`loss_fn` must be callable.")
        for name, value in (
            ("clamp_eps", clamp_eps),
            ("mag_weight", mag_weight),
            ("log_weight", log_weight),
            ("pow", pow),
            ("weight", weight),
        ):
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
        if clamp_eps <= 0.0 or pow <= 0.0:
            raise ValueError("`clamp_eps` and `pow` must be positive.")
        self.log_weight = float(log_weight)
        self.mag_weight = float(mag_weight)
        self.clamp_eps = float(clamp_eps)
        self.weight = float(weight)
        self.pow = float(pow)

    def forward(
        self,
        estimate: Tensor | AudioSignal,
        reference: Tensor | AudioSignal,
    ) -> Tensor:
        estimate_tensor = _audio_tensor(estimate, name="estimate")
        reference_tensor = _audio_tensor(reference, name="reference")
        if estimate_tensor.shape != reference_tensor.shape:
            raise ValueError("STFT-loss inputs must have identical shapes.")
        loss = estimate_tensor.new_zeros(())
        for parameters in self.stft_params:
            estimate_magnitude = _stft_magnitude(
                estimate_tensor,
                parameters,
            )
            reference_magnitude = _stft_magnitude(
                reference_tensor,
                parameters,
            )
            loss = loss + self.log_weight * self.loss_fn(
                estimate_magnitude
                .clamp_min(self.clamp_eps)
                .pow(self.pow)
                .log10(),
                reference_magnitude
                .clamp_min(self.clamp_eps)
                .pow(self.pow)
                .log10(),
            )
            loss = loss + self.mag_weight * self.loss_fn(
                estimate_magnitude,
                reference_magnitude,
            )
        return loss


class MelSpectrogramLoss(nn.Module):
    """Multi-resolution Slaney mel and log-mel objective."""

    def __init__(
        self,
        n_mels: Sequence[int] = (150, 80),
        window_lengths: Sequence[int] = (2_048, 512),
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: Sequence[float] = (0.0, 0.0),
        mel_fmax: Sequence[float | None] = (None, None),
        window_type: str | None = None,
        sample_rate: int | None = 44_100,
    ) -> None:
        super().__init__()
        values = (
            tuple(n_mels),
            tuple(window_lengths),
            tuple(mel_fmin),
            tuple(mel_fmax),
        )
        if not values[0] or len({len(value) for value in values}) != 1:
            raise ValueError(
                "Mel resolutions, FFT windows, and frequency bounds must "
                "have one shared non-zero length."
            )
        self.stft_params = tuple(
            STFTParameters(
                window_length=length,
                hop_length=length // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for length in values[1]
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values[0]
        ):
            raise ValueError("Every mel-bin count must be a positive integer.")
        self.n_mels = values[0]
        self.mel_fmin = values[2]
        self.mel_fmax = values[3]
        self.loss_fn = nn.L1Loss() if loss_fn is None else loss_fn
        if not callable(self.loss_fn):
            raise TypeError("`loss_fn` must be callable.")
        if sample_rate is not None and (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError("`sample_rate` must be positive or None.")
        self.sample_rate = sample_rate
        for name, value in (
            ("clamp_eps", clamp_eps),
            ("mag_weight", mag_weight),
            ("log_weight", log_weight),
            ("pow", pow),
            ("weight", weight),
        ):
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
        if clamp_eps <= 0.0 or pow <= 0.0:
            raise ValueError("`clamp_eps` and `pow` must be positive.")
        self.clamp_eps = float(clamp_eps)
        self.log_weight = float(log_weight)
        self.mag_weight = float(mag_weight)
        self.weight = float(weight)
        self.pow = float(pow)

    def forward(
        self,
        estimate: Tensor | AudioSignal,
        reference: Tensor | AudioSignal,
    ) -> Tensor:
        estimate_tensor = _audio_tensor(estimate, name="estimate")
        reference_tensor = _audio_tensor(reference, name="reference")
        if estimate_tensor.shape != reference_tensor.shape:
            raise ValueError("Mel-loss inputs must have identical shapes.")
        sample_rate = _sample_rate(
            estimate,
            reference,
            self.sample_rate,
        )
        loss = estimate_tensor.new_zeros(())
        for bins, minimum, maximum, parameters in zip(
            self.n_mels,
            self.mel_fmin,
            self.mel_fmax,
            self.stft_params,
        ):
            estimate_magnitude = _stft_magnitude(
                estimate_tensor,
                parameters,
            )
            reference_magnitude = _stft_magnitude(
                reference_tensor,
                parameters,
            )
            filters = mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=parameters.window_length,
                n_mels=bins,
                minimum_frequency=minimum,
                maximum_frequency=maximum,
                dtype=estimate_magnitude.dtype,
                device=estimate_magnitude.device,
            )
            estimate_mel = torch.einsum(
                "mf,bcft->bcmt",
                filters,
                estimate_magnitude,
            )
            reference_mel = torch.einsum(
                "mf,bcft->bcmt",
                filters,
                reference_magnitude,
            )
            loss = loss + self.log_weight * self.loss_fn(
                estimate_mel
                .clamp_min(self.clamp_eps)
                .pow(self.pow)
                .log10(),
                reference_mel
                .clamp_min(self.clamp_eps)
                .pow(self.pow)
                .log10(),
            )
            loss = loss + self.mag_weight * self.loss_fn(
                estimate_mel,
                reference_mel,
            )
        return loss


class GANLoss(nn.Module):
    """Least-squares GAN and feature-matching objectives for DAC."""

    def __init__(self, discriminator: nn.Module) -> None:
        super().__init__()
        if not isinstance(discriminator, nn.Module):
            raise TypeError("`discriminator` must be a PyTorch module.")
        self.discriminator = discriminator

    def forward(
        self,
        fake: Tensor | AudioSignal,
        real: Tensor | AudioSignal,
    ):
        fake_tensor = _audio_tensor(fake, name="fake")
        real_tensor = _audio_tensor(real, name="real")
        if fake_tensor.shape != real_tensor.shape:
            raise ValueError("GAN inputs must have identical shapes.")
        return (
            self.discriminator(fake_tensor),
            self.discriminator(real_tensor),
        )

    def discriminator_loss(
        self,
        fake: Tensor | AudioSignal,
        real: Tensor | AudioSignal,
    ) -> Tensor:
        fake_tensor = _audio_tensor(fake, name="fake").detach()
        real_tensor = _audio_tensor(real, name="real")
        fake_outputs = self.discriminator(fake_tensor)
        real_outputs = self.discriminator(real_tensor)
        loss = real_tensor.new_zeros(())
        for fake_features, real_features in zip(fake_outputs, real_outputs):
            loss = loss + fake_features[-1].square().mean()
            loss = loss + (1.0 - real_features[-1]).square().mean()
        return loss

    def generator_loss(
        self,
        fake: Tensor | AudioSignal,
        real: Tensor | AudioSignal,
    ) -> tuple[Tensor, Tensor]:
        fake_outputs, real_outputs = self.forward(fake, real)
        fake_tensor = _audio_tensor(fake, name="fake")
        adversarial = fake_tensor.new_zeros(())
        feature_matching = fake_tensor.new_zeros(())
        for fake_features, real_features in zip(fake_outputs, real_outputs):
            adversarial = (
                adversarial + (1.0 - fake_features[-1]).square().mean()
            )
            for fake_feature, real_feature in zip(
                fake_features[:-1],
                real_features[:-1],
            ):
                feature_matching = feature_matching + functional.l1_loss(
                    fake_feature,
                    real_feature.detach(),
                )
        return adversarial, feature_matching


__all__ = [
    "GANLoss",
    "L1Loss",
    "MelSpectrogramLoss",
    "MultiScaleSTFTLoss",
    "SISDRLoss",
    "STFTParameters",
]
