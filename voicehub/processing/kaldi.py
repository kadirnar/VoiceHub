"""Differentiable Kaldi-compatible filter-bank features.

Several speech families publish checkpoints trained with Kaldi's exact
framing, Povey window, pre-emphasis, and mel scale.  This module keeps that
frontend inside VoiceHub so architecture implementations do not need
``torchaudio.compliance.kaldi`` at runtime.

The numerical contract follows the public Kaldi-compatible implementation in
TorchAudio 2.8.  VoiceHub adds strict validation, batched execution, explicit
waveform scaling, and serializable configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence

_FLOAT32_EPSILON = torch.finfo(torch.float32).eps
_MAX_CMVN_FILE_BYTES = 16 * 1024 * 1024
_MILLISECONDS_TO_SECONDS = 0.001
_WINDOW_TYPES = frozenset({
    "blackman",
    "hamming",
    "hanning",
    "povey",
    "rectangular",
})


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"`{name}` must be finite.")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class KaldiFbankConfig:
    """Serializable options for a Kaldi-compatible log-mel frontend."""

    sample_frequency: float = 16_000.0
    frame_length: float = 25.0
    frame_shift: float = 10.0
    num_mel_bins: int = 80
    dither: float = 0.0
    energy_floor: float = 0.0
    low_frequency: float = 20.0
    high_frequency: float = 0.0
    preemphasis_coefficient: float = 0.97
    remove_dc_offset: bool = True
    round_to_power_of_two: bool = True
    snip_edges: bool = True
    window_type: str = "povey"
    blackman_coefficient: float = 0.42
    raw_energy: bool = True
    use_energy: bool = False
    use_log_fbank: bool = True
    use_power: bool = True
    htk_compatibility: bool = False
    subtract_mean: bool = False
    minimum_duration: float = 0.0
    vtln_low: float = 100.0
    vtln_high: float = -500.0
    vtln_warp: float = 1.0

    def __post_init__(self) -> None:
        sample_frequency = _finite_real(
            self.sample_frequency,
            name="sample_frequency",
        )
        frame_length = _finite_real(self.frame_length, name="frame_length")
        frame_shift = _finite_real(self.frame_shift, name="frame_shift")
        num_mel_bins = _positive_integer(
            self.num_mel_bins,
            name="num_mel_bins",
        )
        dither = _finite_real(self.dither, name="dither")
        energy_floor = _finite_real(self.energy_floor, name="energy_floor")
        low_frequency = _finite_real(
            self.low_frequency,
            name="low_frequency",
        )
        high_frequency = _finite_real(
            self.high_frequency,
            name="high_frequency",
        )
        preemphasis = _finite_real(
            self.preemphasis_coefficient,
            name="preemphasis_coefficient",
        )
        minimum_duration = _finite_real(
            self.minimum_duration,
            name="minimum_duration",
        )
        vtln_low = _finite_real(self.vtln_low, name="vtln_low")
        vtln_high = _finite_real(self.vtln_high, name="vtln_high")
        vtln_warp = _finite_real(self.vtln_warp, name="vtln_warp")
        blackman = _finite_real(
            self.blackman_coefficient,
            name="blackman_coefficient",
        )
        if sample_frequency <= 0.0:
            raise ValueError("`sample_frequency` must be greater than zero.")
        if frame_length <= 0.0 or frame_shift <= 0.0:
            raise ValueError("Frame length and shift must be greater than zero.")
        if num_mel_bins <= 3:
            raise ValueError("`num_mel_bins` must be greater than three.")
        if dither < 0.0 or energy_floor < 0.0:
            raise ValueError("Dither and energy floor cannot be negative.")
        if low_frequency < 0.0:
            raise ValueError("`low_frequency` cannot be negative.")
        if not 0.0 <= preemphasis <= 1.0:
            raise ValueError(
                "`preemphasis_coefficient` must be between zero and one."
            )
        if minimum_duration < 0.0:
            raise ValueError("`minimum_duration` cannot be negative.")
        if vtln_warp <= 0.0:
            raise ValueError("`vtln_warp` must be greater than zero.")
        if not isinstance(self.window_type, str):
            raise TypeError("`window_type` must be a string.")
        if self.window_type not in _WINDOW_TYPES:
            choices = ", ".join(sorted(_WINDOW_TYPES))
            raise ValueError(f"`window_type` must be one of: {choices}.")
        for name in (
            "remove_dc_offset",
            "round_to_power_of_two",
            "snip_edges",
            "raw_energy",
            "use_energy",
            "use_log_fbank",
            "use_power",
            "htk_compatibility",
            "subtract_mean",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        window_size = int(
            sample_frequency * frame_length * _MILLISECONDS_TO_SECONDS
        )
        window_shift = int(
            sample_frequency * frame_shift * _MILLISECONDS_TO_SECONDS
        )
        if window_size < 2:
            raise ValueError("The configured analysis window is too short.")
        if window_shift < 1:
            raise ValueError("The configured frame shift is too short.")
        padded = (
            _next_power_of_two(window_size)
            if self.round_to_power_of_two
            else window_size
        )
        if padded % 2:
            raise ValueError(
                "The FFT window must be even; enable "
                "`round_to_power_of_two` or change `frame_length`."
            )
        nyquist = sample_frequency / 2.0
        resolved_high = (
            high_frequency + nyquist
            if high_frequency <= 0.0
            else high_frequency
        )
        if not 0.0 <= low_frequency < resolved_high <= nyquist:
            raise ValueError(
                "Mel bounds must satisfy 0 <= low < high <= Nyquist."
            )
        resolved_vtln_high = (
            vtln_high + nyquist
            if vtln_high < 0.0
            else vtln_high
        )
        if vtln_warp != 1.0 and not (
            low_frequency < vtln_low < resolved_vtln_high < resolved_high
        ):
            raise ValueError(
                "VTLN cutoffs must lie strictly inside the mel passband."
            )
        object.__setattr__(self, "sample_frequency", sample_frequency)
        object.__setattr__(self, "frame_length", frame_length)
        object.__setattr__(self, "frame_shift", frame_shift)
        object.__setattr__(self, "num_mel_bins", num_mel_bins)
        object.__setattr__(self, "dither", dither)
        object.__setattr__(self, "energy_floor", energy_floor)
        object.__setattr__(self, "low_frequency", low_frequency)
        object.__setattr__(self, "high_frequency", high_frequency)
        object.__setattr__(self, "preemphasis_coefficient", preemphasis)
        object.__setattr__(self, "minimum_duration", minimum_duration)
        object.__setattr__(self, "vtln_low", vtln_low)
        object.__setattr__(self, "vtln_high", vtln_high)
        object.__setattr__(self, "vtln_warp", vtln_warp)
        object.__setattr__(self, "blackman_coefficient", blackman)

    @classmethod
    def coerce(
        cls,
        value: "KaldiFbankConfig | dict[str, Any] | None",
    ) -> "KaldiFbankConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("Kaldi fbank config must be a mapping or config.")

    @property
    def window_size(self) -> int:
        return int(
            self.sample_frequency
            * self.frame_length
            * _MILLISECONDS_TO_SECONDS
        )

    @property
    def window_shift(self) -> int:
        return int(
            self.sample_frequency
            * self.frame_shift
            * _MILLISECONDS_TO_SECONDS
        )

    @property
    def padded_window_size(self) -> int:
        if self.round_to_power_of_two:
            return _next_power_of_two(self.window_size)
        return self.window_size

    @property
    def feature_size(self) -> int:
        return self.num_mel_bins + int(self.use_energy)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _next_power_of_two(value: int) -> int:
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def _strided_frames(
    waveform: Tensor,
    *,
    window_size: int,
    window_shift: int,
    snip_edges: bool,
) -> Tensor:
    if waveform.ndim != 1:
        raise ValueError("Internal Kaldi framing requires a rank-one tensor.")
    sample_count = waveform.shape[0]
    strides = (
        window_shift * waveform.stride(0),
        waveform.stride(0),
    )
    if snip_edges:
        if sample_count < window_size:
            return waveform.new_empty((0, window_size))
        frame_count = 1 + (sample_count - window_size) // window_shift
        return waveform.as_strided(
            (frame_count, window_size),
            strides,
        )

    frame_count = (sample_count + window_shift // 2) // window_shift
    reflected = waveform.flip(0)
    left_amount = window_size // 2 - window_shift // 2
    if left_amount > 0:
        left = reflected[-left_amount:]
        padded = torch.cat((left, waveform, reflected), dim=0)
    else:
        padded = torch.cat((waveform[-left_amount:], reflected), dim=0)
    return padded.as_strided(
        (frame_count, window_size),
        strides,
    )


def _window_function(
    config: KaldiFbankConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    size = config.window_size
    if config.window_type == "hanning":
        return torch.hann_window(
            size,
            periodic=False,
            device=device,
            dtype=dtype,
        )
    if config.window_type == "hamming":
        return torch.hamming_window(
            size,
            periodic=False,
            alpha=0.54,
            beta=0.46,
            device=device,
            dtype=dtype,
        )
    if config.window_type == "povey":
        return torch.hann_window(
            size,
            periodic=False,
            device=device,
            dtype=dtype,
        ).pow(0.85)
    if config.window_type == "rectangular":
        return torch.ones(size, device=device, dtype=dtype)
    positions = torch.arange(size, device=device, dtype=dtype)
    phase = 2.0 * math.pi / (size - 1)
    coefficient = config.blackman_coefficient
    return (
        coefficient
        - 0.5 * torch.cos(phase * positions)
        + (0.5 - coefficient) * torch.cos(2.0 * phase * positions)
    )


def _epsilon(reference: Tensor) -> Tensor:
    return torch.tensor(
        _FLOAT32_EPSILON,
        device=reference.device,
        dtype=reference.dtype,
    )


def _log_energy(
    frames: Tensor,
    *,
    energy_floor: float,
) -> Tensor:
    energy = frames.square().sum(dim=1).maximum(_epsilon(frames)).log()
    if energy_floor == 0.0:
        return energy
    floor = torch.tensor(
        math.log(energy_floor),
        device=frames.device,
        dtype=frames.dtype,
    )
    return energy.maximum(floor)


def _windowed_frames(
    waveform: Tensor,
    config: KaldiFbankConfig,
) -> tuple[Tensor, Tensor]:
    frames = _strided_frames(
        waveform,
        window_size=config.window_size,
        window_shift=config.window_shift,
        snip_edges=config.snip_edges,
    )
    if frames.shape[0] == 0:
        return (
            waveform.new_empty((0, config.padded_window_size)),
            waveform.new_empty((0,)),
        )
    if config.dither:
        frames = frames + torch.randn_like(frames) * config.dither
    if config.remove_dc_offset:
        frames = frames - frames.mean(dim=1, keepdim=True)
    energy = (
        _log_energy(frames, energy_floor=config.energy_floor)
        if config.raw_energy
        else None
    )
    if config.preemphasis_coefficient:
        previous = functional.pad(
            frames.unsqueeze(0),
            (1, 0),
            mode="replicate",
        ).squeeze(0)[:, :-1]
        frames = (
            frames
            - config.preemphasis_coefficient * previous
        )
    frames = frames * _window_function(
        config,
        device=frames.device,
        dtype=frames.dtype,
    )
    if config.padded_window_size != config.window_size:
        frames = functional.pad(
            frames,
            (0, config.padded_window_size - config.window_size),
        )
    if energy is None:
        energy = _log_energy(frames, energy_floor=config.energy_floor)
    return frames, energy


def _mel_scale_scalar(frequency: float) -> float:
    return 1_127.0 * math.log(1.0 + frequency / 700.0)


def _mel_scale(frequencies: Tensor) -> Tensor:
    return 1_127.0 * (1.0 + frequencies / 700.0).log()


def _inverse_mel_scale(frequencies: Tensor) -> Tensor:
    return 700.0 * ((frequencies / 1_127.0).exp() - 1.0)


def _vtln_warp_frequency(
    frequencies: Tensor,
    *,
    low_cutoff: float,
    high_cutoff: float,
    low_frequency: float,
    high_frequency: float,
    warp: float,
) -> Tensor:
    lower_inflection = low_cutoff * max(1.0, warp)
    upper_inflection = high_cutoff * min(1.0, warp)
    scale = 1.0 / warp
    warped_lower = scale * lower_inflection
    warped_upper = scale * upper_inflection
    left_scale = (
        (warped_lower - low_frequency)
        / (lower_inflection - low_frequency)
    )
    right_scale = (
        (high_frequency - warped_upper)
        / (high_frequency - upper_inflection)
    )
    result = torch.empty_like(frequencies)
    outside = (
        (frequencies < low_frequency)
        | (frequencies > high_frequency)
    )
    before_lower = frequencies < lower_inflection
    before_upper = frequencies < upper_inflection
    after_upper = frequencies >= upper_inflection
    result[after_upper] = (
        high_frequency
        + right_scale * (frequencies[after_upper] - high_frequency)
    )
    result[before_upper] = scale * frequencies[before_upper]
    result[before_lower] = (
        low_frequency
        + left_scale * (frequencies[before_lower] - low_frequency)
    )
    result[outside] = frequencies[outside]
    return result


def kaldi_mel_filter_bank(
    config: KaldiFbankConfig | dict[str, Any] | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build Kaldi triangular mel weights without the padded Nyquist column."""
    resolved = KaldiFbankConfig.coerce(config)
    nyquist = resolved.sample_frequency / 2.0
    high_frequency = resolved.high_frequency
    if high_frequency <= 0.0:
        high_frequency += nyquist
    vtln_high = resolved.vtln_high
    if vtln_high < 0.0:
        vtln_high += nyquist
    low_mel = _mel_scale_scalar(resolved.low_frequency)
    high_mel = _mel_scale_scalar(high_frequency)
    mel_step = (
        (high_mel - low_mel)
        / (resolved.num_mel_bins + 1)
    )
    indices = torch.arange(
        resolved.num_mel_bins,
        dtype=dtype,
        device=device,
    ).unsqueeze(1)
    left = low_mel + indices * mel_step
    center = low_mel + (indices + 1.0) * mel_step
    right = low_mel + (indices + 2.0) * mel_step
    if resolved.vtln_warp != 1.0:
        def warp(values: Tensor) -> Tensor:
            frequencies = _inverse_mel_scale(values)
            return _mel_scale(_vtln_warp_frequency(
                frequencies,
                low_cutoff=resolved.vtln_low,
                high_cutoff=vtln_high,
                low_frequency=resolved.low_frequency,
                high_frequency=high_frequency,
                warp=resolved.vtln_warp,
            ))

        left, center, right = warp(left), warp(center), warp(right)
    fft_width = (
        resolved.sample_frequency
        / resolved.padded_window_size
    )
    mel_bins = _mel_scale(
        fft_width
        * torch.arange(
            resolved.padded_window_size // 2,
            dtype=dtype,
            device=device,
        )
    ).unsqueeze(0)
    rising = (mel_bins - left) / (center - left)
    falling = (right - mel_bins) / (right - center)
    if resolved.vtln_warp == 1.0:
        return torch.minimum(rising, falling).clamp_min(0.0)
    weights = torch.zeros_like(rising)
    rising_mask = (mel_bins > left) & (mel_bins <= center)
    falling_mask = (mel_bins > center) & (mel_bins < right)
    weights[rising_mask] = rising[rising_mask]
    weights[falling_mask] = falling[falling_mask]
    return weights


def kaldi_fbank(
    waveform: Tensor,
    config: KaldiFbankConfig | dict[str, Any] | None = None,
    *,
    channel: int = -1,
) -> Tensor:
    """Compute one Kaldi-compatible feature matrix.

    ``waveform`` may have shape ``[samples]`` or ``[channels, samples]``.
    Values are not implicitly scaled; callers loading normalized PCM should
    multiply by the scale used during model training (usually ``32768``).
    """
    resolved = KaldiFbankConfig.coerce(config)
    if not isinstance(waveform, Tensor):
        raise TypeError("`waveform` must be a PyTorch tensor.")
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(
            "`waveform` must have shape [samples] or [channels, samples]."
        )
    if not waveform.is_floating_point():
        raise TypeError("`waveform` must use a floating-point dtype.")
    if waveform.is_complex():
        raise TypeError("`waveform` must contain real values.")
    if not torch.isfinite(waveform).all():
        raise ValueError("`waveform` contains NaN or infinite values.")
    selected_channel = max(int(channel), 0)
    if selected_channel >= waveform.shape[0]:
        raise ValueError(
            f"Channel {channel} is unavailable for "
            f"{waveform.shape[0]}-channel audio."
        )
    samples = waveform[selected_channel]
    if samples.shape[0] < resolved.window_size:
        return samples.new_empty((0, resolved.feature_size))
    if (
        samples.shape[0]
        < resolved.minimum_duration * resolved.sample_frequency
    ):
        return samples.new_empty((0, resolved.feature_size))
    frames, log_energy = _windowed_frames(samples, resolved)
    spectrum = torch.fft.rfft(frames).abs()
    if resolved.use_power:
        spectrum = spectrum.square()
    filters = kaldi_mel_filter_bank(
        resolved,
        device=spectrum.device,
        dtype=spectrum.dtype,
    )
    filters = functional.pad(filters, (0, 1))
    mel = spectrum @ filters.transpose(0, 1)
    if resolved.use_log_fbank:
        mel = mel.maximum(_epsilon(mel)).log()
    if resolved.use_energy:
        energy = log_energy.unsqueeze(1)
        mel = (
            torch.cat((mel, energy), dim=1)
            if resolved.htk_compatibility
            else torch.cat((energy, mel), dim=1)
        )
    if resolved.subtract_mean and mel.shape[0]:
        mel = mel - mel.mean(dim=0, keepdim=True)
    return mel


class KaldiFbank(nn.Module):
    """Batch-aware module for architecture frontends.

    Args:
        config: Kaldi feature parameters.
        waveform_scale: Scale applied before feature extraction. Use
            ``32768`` for checkpoints trained from normalized PCM multiplied
            into the signed 16-bit amplitude domain.
    """

    def __init__(
        self,
        config: KaldiFbankConfig | dict[str, Any] | None = None,
        *,
        waveform_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = KaldiFbankConfig.coerce(config)
        self.waveform_scale = _finite_real(
            waveform_scale,
            name="waveform_scale",
        )
        if self.waveform_scale <= 0.0:
            raise ValueError("`waveform_scale` must be greater than zero.")

    def frame_count(self, sample_count: int) -> int:
        sample_count = _positive_integer(
            sample_count,
            name="sample_count",
        )
        window = self.config.window_size
        shift = self.config.window_shift
        if self.config.snip_edges:
            if sample_count < window:
                return 0
            return 1 + (sample_count - window) // shift
        return (sample_count + shift // 2) // shift

    def forward(
        self,
        waveforms: Tensor,
        waveform_lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(waveforms, Tensor):
            raise TypeError("`waveforms` must be a PyTorch tensor.")
        if waveforms.ndim == 1:
            waveforms = waveforms.unsqueeze(0)
        if waveforms.ndim != 2:
            raise ValueError(
                "`waveforms` must have shape [samples] or [batch, samples]."
            )
        if not waveforms.is_floating_point():
            raise TypeError("`waveforms` must use a floating-point dtype.")
        batch_size, maximum_samples = waveforms.shape
        if waveform_lengths is None:
            lengths = torch.full(
                (batch_size,),
                maximum_samples,
                dtype=torch.long,
                device=waveforms.device,
            )
        else:
            lengths = torch.as_tensor(
                waveform_lengths,
                dtype=torch.long,
                device=waveforms.device,
            )
            if tuple(lengths.shape) != (batch_size,):
                raise ValueError(
                    "`waveform_lengths` must have shape [batch]."
                )
            if (
                (lengths <= 0).any()
                or (lengths > maximum_samples).any()
            ):
                raise ValueError(
                    "Waveform lengths must lie within the padded batch."
                )
        rows = [
            kaldi_fbank(
                waveforms[index, :int(length.item())]
                * self.waveform_scale,
                self.config,
            )
            for index, length in enumerate(lengths)
        ]
        frame_lengths = torch.tensor(
            [row.shape[0] for row in rows],
            dtype=torch.long,
            device=waveforms.device,
        )
        if not rows:
            return (
                waveforms.new_empty(
                    (0, 0, self.config.feature_size)
                ),
                frame_lengths,
            )
        return (
            pad_sequence(
                rows,
                batch_first=True,
                padding_value=0.0,
            ),
            frame_lengths,
        )

    def extra_repr(self) -> str:
        return (
            f"sample_frequency={self.config.sample_frequency:g}, "
            f"num_mel_bins={self.config.num_mel_bins}, "
            f"frame_length={self.config.frame_length:g}, "
            f"frame_shift={self.config.frame_shift:g}, "
            f"waveform_scale={self.waveform_scale:g}"
        )


class GlobalFeatureNormalization(nn.Module):
    """Apply fixed global mean and inverse-standard-deviation statistics."""

    def __init__(
        self,
        mean: Tensor,
        inverse_std: Tensor,
    ) -> None:
        super().__init__()
        for name, value in (
            ("mean", mean),
            ("inverse_std", inverse_std),
        ):
            if not isinstance(value, Tensor) or value.ndim != 1:
                raise ValueError(
                    f"`{name}` must be a rank-one PyTorch tensor."
                )
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(
                    f"`{name}` must contain finite floating-point values."
                )
        if mean.shape != inverse_std.shape:
            raise ValueError(
                "`mean` and `inverse_std` must have identical shapes."
            )
        if (inverse_std <= 0).any():
            raise ValueError("`inverse_std` must be strictly positive.")
        self.register_buffer(
            "mean",
            mean.detach().to(dtype=torch.float32).clone(),
        )
        self.register_buffer(
            "inverse_std",
            inverse_std.detach().to(dtype=torch.float32).clone(),
        )

    def forward(self, features: Tensor) -> Tensor:
        if not isinstance(features, Tensor) or features.ndim < 2:
            raise ValueError(
                "`features` must have a final feature dimension."
            )
        if features.shape[-1] != self.mean.numel():
            raise ValueError(
                f"Expected {self.mean.numel()} feature bins, found "
                f"{features.shape[-1]}."
            )
        mean = self.mean.to(
            device=features.device,
            dtype=features.dtype,
        )
        inverse_std = self.inverse_std.to(
            device=features.device,
            dtype=features.dtype,
        )
        return (features - mean) * inverse_std


def _cmvn_vector(
    value: Any,
    *,
    name: str,
) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"`{name}` must be a non-empty JSON array.")
    result: list[float] = []
    for index, item in enumerate(value):
        try:
            number = _finite_real(item, name=f"{name}[{index}]")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"`{name}` must contain only finite numbers."
            ) from error
        result.append(number)
    return result


def _json_cmvn_statistics(
    content: str,
) -> tuple[list[float], list[float], float]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("The CMVN file contains invalid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("A JSON CMVN file must contain an object.")
    missing = {
        key
        for key in ("mean_stat", "var_stat", "frame_num")
        if key not in payload
    }
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"The CMVN file is missing: {names}.")
    means = _cmvn_vector(payload["mean_stat"], name="mean_stat")
    variances = _cmvn_vector(payload["var_stat"], name="var_stat")
    count = _finite_real(payload["frame_num"], name="frame_num")
    return means, variances, count


def _kaldi_cmvn_statistics(
    content: str,
) -> tuple[list[float], list[float], float]:
    if content.startswith("\0B"):
        raise ValueError(
            "Binary Kaldi CMVN files are unsupported; export text statistics."
        )
    tokens = content.split()
    if len(tokens) < 6 or tokens[0] != "[" or tokens[-1] != "]":
        raise ValueError("The Kaldi CMVN text matrix is malformed.")
    if (len(tokens) - 4) % 2:
        raise ValueError("The Kaldi CMVN feature dimension is ambiguous.")
    dimension = (len(tokens) - 4) // 2
    if dimension < 1:
        raise ValueError("The Kaldi CMVN matrix has no feature statistics.")
    if tokens[-2] != "0":
        raise ValueError("The Kaldi CMVN matrix must end with a zero count.")
    try:
        means = [
            _finite_real(
                float(tokens[index]),
                name=f"mean_stat[{index - 1}]",
            )
            for index in range(1, dimension + 1)
        ]
        count = _finite_real(
            float(tokens[dimension + 1]),
            name="frame_num",
        )
        variances = [
            _finite_real(
                float(tokens[index]),
                name=f"var_stat[{index - dimension - 2}]",
            )
            for index in range(dimension + 2, 2 * dimension + 2)
        ]
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The Kaldi CMVN matrix must contain finite numbers."
        ) from error
    return means, variances, count


def load_global_cmvn(
    path: str | Path,
    *,
    format: str = "auto",
    expected_dimension: int | None = None,
    variance_floor: float = 1.0e-20,
    max_file_bytes: int = _MAX_CMVN_FILE_BYTES,
) -> GlobalFeatureNormalization:
    """Load JSON or text-Kaldi global CMVN statistics.

    The JSON contract uses ``mean_stat``, ``var_stat``, and ``frame_num``.
    Text matrices follow the output of Kaldi's
    ``compute-cmvn-stats --binary=false``.  Accumulated first and second
    moments are converted with the same formula used by WeNet:

    ``mean = mean_stat / frame_num``
    ``inverse_std = 1 / sqrt(max(var_stat / frame_num - mean**2, floor))``

    Args:
        path: Statistics file to load.
        format: ``"json"``, ``"kaldi"``, or ``"auto"``.
        expected_dimension: Optional exact feature dimension.
        variance_floor: Positive lower bound applied before the square root.
        max_file_bytes: Safety limit for untrusted metadata.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("`path` must be a string or pathlib.Path.")
    if format not in {"auto", "json", "kaldi"}:
        raise ValueError("`format` must be 'auto', 'json', or 'kaldi'.")
    if expected_dimension is not None:
        expected_dimension = _positive_integer(
            expected_dimension,
            name="expected_dimension",
        )
    variance_floor = _finite_real(
        variance_floor,
        name="variance_floor",
    )
    if variance_floor <= 0.0:
        raise ValueError("`variance_floor` must be greater than zero.")
    max_file_bytes = _positive_integer(
        max_file_bytes,
        name="max_file_bytes",
    )
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"CMVN statistics were not found: {resolved}")
    size = resolved.stat().st_size
    if size > max_file_bytes:
        raise ValueError(
            f"CMVN file is {size} bytes; limit is {max_file_bytes} bytes."
        )
    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            "CMVN statistics must be UTF-8 JSON or text-Kaldi data."
        ) from error
    resolved_format = format
    if resolved_format == "auto":
        resolved_format = (
            "json"
            if content.lstrip().startswith("{")
            else "kaldi"
        )
    if resolved_format == "json":
        means, variances, count = _json_cmvn_statistics(content)
    else:
        means, variances, count = _kaldi_cmvn_statistics(content)
    if count <= 0.0:
        raise ValueError("`frame_num` must be greater than zero.")
    if len(means) != len(variances):
        raise ValueError(
            "`mean_stat` and `var_stat` must have identical dimensions."
        )
    if (
        expected_dimension is not None
        and len(means) != expected_dimension
    ):
        raise ValueError(
            f"Expected {expected_dimension} CMVN bins, found {len(means)}."
        )
    mean = torch.tensor(means, dtype=torch.float64) / count
    second_moment = torch.tensor(
        variances,
        dtype=torch.float64,
    ) / count
    variance = (second_moment - mean.square()).clamp_min(variance_floor)
    inverse_std = variance.rsqrt()
    return GlobalFeatureNormalization(
        mean.to(dtype=torch.float32),
        inverse_std.to(dtype=torch.float32),
    )


__all__ = [
    "GlobalFeatureNormalization",
    "KaldiFbank",
    "KaldiFbankConfig",
    "kaldi_fbank",
    "kaldi_mel_filter_bank",
    "load_global_cmvn",
]
