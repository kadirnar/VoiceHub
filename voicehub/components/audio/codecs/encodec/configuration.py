"""Validated configuration for VoiceHub's native Encodec graphs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

_CONVOLUTION_NORMALIZATIONS = frozenset({
    "none",
    "weight_norm",
    "spectral_norm",
    "time_layer_norm",
    "layer_norm",
    "time_group_norm",
})


@dataclass(frozen=True, slots=True)
class EncodecConfig:
    """Complete, serializable construction contract for one Encodec model.

    The defaults reproduce the SEANet and residual-vector-quantizer settings
    used by the official Meta Encodec release.  All fields that affect the
    state-dict namespace or tensor shapes are explicit so exported VoiceHub
    artifacts can be reconstructed without importing a provider library.
    """

    target_bandwidths: tuple[float, ...]
    sample_rate: int = 24_000
    channels: int = 1
    normalize: bool = False
    segment: float | None = None
    overlap: float = 0.01
    name: str = "unset"
    causal: bool = True
    model_norm: str = "weight_norm"
    dimension: int = 128
    n_filters: int = 32
    n_residual_layers: int = 1
    ratios: tuple[int, ...] = (8, 5, 4, 2)
    kernel_size: int = 7
    last_kernel_size: int = 7
    residual_kernel_size: int = 3
    dilation_base: int = 2
    true_skip: bool = False
    compress: int = 2
    lstm: int = 2
    trim_right_ratio: float = 1.0
    bins: int = 1024
    n_q: int | None = None
    decay: float = 0.99
    kmeans_init: bool = True
    kmeans_iters: int = 50
    threshold_ema_dead_code: int = 2

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) for value in self.target_bandwidths):
            raise ValueError("`target_bandwidths` cannot contain booleans.")
        bandwidths = tuple(float(value) for value in self.target_bandwidths)
        object.__setattr__(self, "target_bandwidths", bandwidths)
        object.__setattr__(self, "ratios", tuple(self.ratios))

        if not bandwidths or any(not math.isfinite(value) or value <= 0 for value in bandwidths):
            raise ValueError("`target_bandwidths` must contain positive finite values.")
        if tuple(sorted(set(bandwidths))) != bandwidths:
            raise ValueError("`target_bandwidths` must be unique and strictly increasing.")
        for field_name in (
            "sample_rate",
            "channels",
            "dimension",
            "n_filters",
            "kernel_size",
            "last_kernel_size",
            "residual_kernel_size",
            "dilation_base",
            "compress",
            "bins",
            "kmeans_iters",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{field_name}` must be a positive integer.")
        if (
            isinstance(self.n_residual_layers, bool)
            or not isinstance(self.n_residual_layers, int)
            or self.n_residual_layers < 0
        ):
            raise ValueError("`n_residual_layers` must be a non-negative integer.")
        if isinstance(self.lstm, bool) or not isinstance(self.lstm, int) or self.lstm < 0:
            raise ValueError("`lstm` must be a non-negative integer.")
        if (
            isinstance(self.threshold_ema_dead_code, bool)
            or not isinstance(self.threshold_ema_dead_code, int)
            or self.threshold_ema_dead_code < 0
        ):
            raise ValueError("`threshold_ema_dead_code` must be a non-negative integer.")
        if self.channels not in {1, 2}:
            raise ValueError("Encodec supports one or two audio channels.")
        if not self.ratios or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.ratios
        ):
            raise ValueError("`ratios` must contain positive integers.")
        if self.dimension % self.compress or self.n_filters % self.compress:
            raise ValueError(
                "`dimension` and `n_filters` must be divisible by `compress`.")
        if self.model_norm not in _CONVOLUTION_NORMALIZATIONS:
            raise ValueError(f"Unsupported convolution normalization {self.model_norm!r}.")
        if self.causal and self.model_norm == "time_group_norm":
            raise ValueError("Time-wise group normalization is not causal.")
        if self.segment is not None and (
            not isinstance(self.segment, (int, float))
            or isinstance(self.segment, bool)
            or not math.isfinite(float(self.segment))
            or self.segment <= 0
        ):
            raise ValueError("`segment` must be a positive finite duration or None.")
        if (
            not isinstance(self.overlap, (int, float))
            or isinstance(self.overlap, bool)
            or not math.isfinite(float(self.overlap))
            or not 0 <= self.overlap < 1
        ):
            raise ValueError("`overlap` must be in the half-open interval [0, 1).")
        if not 0 <= self.trim_right_ratio <= 1:
            raise ValueError("`trim_right_ratio` must be in the interval [0, 1].")
        if not self.causal and self.trim_right_ratio != 1:
            raise ValueError("Right-trim ratios other than one require a causal decoder.")
        if not 0 < self.decay < 1:
            raise ValueError("`decay` must be in the open interval (0, 1).")
        if self.bins & (self.bins - 1):
            raise ValueError("`bins` must be a power of two.")
        if self.n_q is not None and (
            isinstance(self.n_q, bool) or not isinstance(self.n_q, int) or self.n_q <= 0
        ):
            raise ValueError("`n_q` must be a positive integer or None.")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("`name` must be a non-empty string.")

    @property
    def hop_length(self) -> int:
        return math.prod(self.ratios)

    @property
    def frame_rate(self) -> int:
        return math.ceil(self.sample_rate / self.hop_length)

    @property
    def bits_per_codebook(self) -> int:
        return int(math.log2(self.bins))

    @property
    def resolved_n_q(self) -> int:
        if self.n_q is not None:
            return self.n_q
        denominator = self.frame_rate * self.bits_per_codebook
        return max(1, int(1000 * self.target_bandwidths[-1] // denominator))

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["target_bandwidths"] = list(self.target_bandwidths)
        values["ratios"] = list(self.ratios)
        return values

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> EncodecConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Encodec configuration must be a mapping.")
        known = set(cls.__dataclass_fields__)
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown Encodec configuration fields: {sorted(unknown)!r}.")
        normalized = dict(values)
        if "target_bandwidths" in normalized:
            normalized["target_bandwidths"] = tuple(normalized["target_bandwidths"])
        if "ratios" in normalized:
            normalized["ratios"] = tuple(normalized["ratios"])
        return cls(**normalized)


def encodec_24khz_config() -> EncodecConfig:
    """Return the exact configuration of Meta's released 24 kHz model."""
    return EncodecConfig(
        target_bandwidths=(1.5, 3.0, 6.0, 12.0, 24.0),
        sample_rate=24_000,
        channels=1,
        causal=True,
        model_norm="weight_norm",
        normalize=False,
        segment=None,
        name="encodec_24khz",
    )


def encodec_48khz_config() -> EncodecConfig:
    """Return the exact configuration of Meta's released 48 kHz model."""
    return EncodecConfig(
        target_bandwidths=(3.0, 6.0, 12.0, 24.0),
        sample_rate=48_000,
        channels=2,
        causal=False,
        model_norm="time_group_norm",
        normalize=True,
        segment=1.0,
        name="encodec_48khz",
    )


__all__ = [
    "EncodecConfig",
    "encodec_24khz_config",
    "encodec_48khz_config",
]
