"""Validated, framework-independent SpeechT5 architecture configuration.

The public VoiceHub configuration also carries artifact-resolution
controls. The dataclasses in this module describe only tensor shapes and
mathematical behavior, which keeps the native graph reusable by
inference, training, and future optimization backends.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"`{name}` must be finite and between zero and one.")
    return value


def _from_mapping(cls, values: Mapping[str, Any]):
    if not isinstance(values, Mapping):
        raise TypeError("SpeechT5 configuration values must be a mapping.")
    names = {field.name for field in fields(cls)}
    return cls(**{name: values[name] for name in names if name in values})


@dataclass(frozen=True, slots=True)
class NativeSpeechT5Config:
    """Tensor-shape contract for ``microsoft/speecht5_tts``."""

    vocab_size: int = 81
    hidden_size: int = 768
    encoder_layers: int = 12
    encoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072
    encoder_layerdrop: float = 0.1
    decoder_layers: int = 6
    decoder_ffn_dim: int = 3072
    decoder_attention_heads: int = 12
    decoder_layerdrop: float = 0.1
    hidden_act: str = "gelu"
    positional_dropout: float = 0.1
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    scale_embedding: bool = False
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    decoder_start_token_id: int = 2
    num_mel_bins: int = 80
    speech_decoder_prenet_layers: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_dropout: float = 0.5
    speaker_embedding_dim: int = 512
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_units: int = 256
    speech_decoder_postnet_kernel: int = 5
    speech_decoder_postnet_dropout: float = 0.5
    reduction_factor: int = 2
    max_speech_positions: int = 1876
    max_text_positions: int = 600
    encoder_max_relative_position: int = 160
    use_guided_attention_loss: bool = True
    guided_attention_loss_num_heads: int = 2
    guided_attention_loss_sigma: float = 0.4
    guided_attention_loss_scale: float = 10.0
    use_cache: bool = True
    return_dict: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False

    def __post_init__(self) -> None:
        for name in (
                "vocab_size",
                "hidden_size",
                "encoder_layers",
                "encoder_attention_heads",
                "encoder_ffn_dim",
                "decoder_layers",
                "decoder_ffn_dim",
                "decoder_attention_heads",
                "num_mel_bins",
                "speech_decoder_prenet_layers",
                "speech_decoder_prenet_units",
                "speaker_embedding_dim",
                "speech_decoder_postnet_layers",
                "speech_decoder_postnet_units",
                "speech_decoder_postnet_kernel",
                "reduction_factor",
                "max_speech_positions",
                "max_text_positions",
                "encoder_max_relative_position",
        ):
            _positive_integer(getattr(self, name), name=name)
        for name in (
                "encoder_layerdrop",
                "decoder_layerdrop",
                "positional_dropout",
                "hidden_dropout",
                "attention_dropout",
                "activation_dropout",
                "speech_decoder_prenet_dropout",
                "speech_decoder_postnet_dropout",
        ):
            object.__setattr__(
                self,
                name,
                _probability(getattr(self, name), name=name),
            )
        if self.speech_decoder_prenet_dropout >= 1.0:
            raise ValueError(
                "`speech_decoder_prenet_dropout` must be less than one "
                "because SpeechT5 applies inverted prenet dropout during "
                "inference.")
        for name in (
                "initializer_range",
                "layer_norm_eps",
                "guided_attention_loss_sigma",
                "guided_attention_loss_scale",
        ):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"`{name}` must be finite and greater than zero.")
            object.__setattr__(self, name, float(value))
        if self.hidden_size % self.encoder_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `encoder_attention_heads`.")
        if self.hidden_size % self.decoder_attention_heads:
            raise ValueError("`hidden_size` must be divisible by `decoder_attention_heads`.")
        if self.guided_attention_loss_num_heads == 0:
            raise ValueError("`guided_attention_loss_num_heads` must be -1 or a positive integer.")
        if (self.guided_attention_loss_num_heads < -1 or
                self.guided_attention_loss_num_heads > self.decoder_attention_heads):
            raise ValueError("`guided_attention_loss_num_heads` is outside the decoder head range.")
        if self.hidden_act not in {"gelu", "gelu_new", "relu", "selu", "silu"}:
            raise ValueError(f"Unsupported SpeechT5 activation {self.hidden_act!r}.")
        for name in (
                "scale_embedding",
                "use_guided_attention_loss",
                "use_cache",
                "return_dict",
                "output_attentions",
                "output_hidden_states",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> NativeSpeechT5Config:
        return _from_mapping(cls, values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NativeSpeechT5HifiGanConfig:
    """Tensor-shape contract for ``microsoft/speecht5_hifigan``."""

    model_in_dim: int = 80
    sampling_rate: int = 16_000
    upsample_initial_channel: int = 512
    upsample_rates: tuple[int, ...] = (4, 4, 4, 4)
    upsample_kernel_sizes: tuple[int, ...] = (8, 8, 8, 8)
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
        (1, 3, 5),
        (1, 3, 5),
        (1, 3, 5),
    )
    initializer_range: float = 0.01
    leaky_relu_slope: float = 0.1
    normalize_before: bool = True

    def __post_init__(self) -> None:
        for name in (
                "model_in_dim",
                "sampling_rate",
                "upsample_initial_channel",
        ):
            _positive_integer(getattr(self, name), name=name)
        for name in (
                "upsample_rates",
                "upsample_kernel_sizes",
                "resblock_kernel_sizes",
        ):
            values = tuple(getattr(self, name))
            if not values:
                raise ValueError(f"`{name}` cannot be empty.")
            for index, value in enumerate(values):
                _positive_integer(value, name=f"{name}[{index}]")
            object.__setattr__(self, name, values)
        dilations = tuple(tuple(group) for group in self.resblock_dilation_sizes)
        if len(dilations) != len(self.resblock_kernel_sizes):
            raise ValueError("`resblock_dilation_sizes` must match `resblock_kernel_sizes`.")
        for group_index, group in enumerate(dilations):
            if not group:
                raise ValueError("HiFi-GAN dilation groups cannot be empty.")
            for index, value in enumerate(group):
                _positive_integer(
                    value,
                    name=f"resblock_dilation_sizes[{group_index}][{index}]",
                )
        object.__setattr__(self, "resblock_dilation_sizes", dilations)
        if len(self.upsample_rates) != len(self.upsample_kernel_sizes):
            raise ValueError("`upsample_rates` must match `upsample_kernel_sizes`.")
        for rate, kernel in zip(
                self.upsample_rates,
                self.upsample_kernel_sizes,
        ):
            if kernel < rate or (kernel - rate) % 2:
                raise ValueError("HiFi-GAN upsample kernels must produce symmetric padding.")
        for name in ("initializer_range", "leaky_relu_slope"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not isfinite(float(value)) or float(value) <= 0.0):
                raise ValueError(f"`{name}` must be finite and greater than zero.")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.normalize_before, bool):
            raise TypeError("`normalize_before` must be a boolean.")

    @property
    def upsample_factor(self) -> int:
        factor = 1
        for rate in self.upsample_rates:
            factor *= rate
        return factor

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> NativeSpeechT5HifiGanConfig:
        return _from_mapping(cls, values)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["model_type"] = "speecht5_hifigan"
        return values


__all__ = [
    "NativeSpeechT5Config",
    "NativeSpeechT5HifiGanConfig",
]
