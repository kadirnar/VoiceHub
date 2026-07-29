"""PyTorch-native streamable convolution and SEANet building blocks."""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

CONV_NORMALIZATIONS = frozenset({
    "none",
    "weight_norm",
    "spectral_norm",
    "time_layer_norm",
    "layer_norm",
    "time_group_norm",
})


class ConvLayerNorm(nn.LayerNorm):
    """Apply layer normalization to channels in convolutional layout."""

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim < 3:
            raise ValueError("Convolutional layer normalization expects rank >= 3.")
        return super().forward(value.movedim(1, -1)).movedim(-1, 1)


def _validate_norm(norm: str) -> None:
    if norm not in CONV_NORMALIZATIONS:
        raise ValueError(f"Unsupported convolution normalization {norm!r}.")


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    """Apply a parameter-level normalization without changing official names."""
    _validate_norm(norm)
    if norm == "weight_norm":
        # The released Encodec checkpoint uses the legacy ``weight_g`` and
        # ``weight_v`` namespace.  The newer parametrizations API is not
        # state-compatible with that artifact.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return weight_norm(module)
    if norm == "spectral_norm":
        return spectral_norm(module)
    return module


def get_norm_module(
    module: nn.Module,
    *,
    causal: bool = False,
    norm: str = "none",
    norm_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Construct the activation-level normalization for a convolution."""
    _validate_norm(norm)
    kwargs = dict(norm_kwargs or {})
    if norm in {"layer_norm", "time_layer_norm"}:
        if not isinstance(
            module,
            (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d),
        ):
            raise TypeError("Layer normalization requires a convolution module.")
        return ConvLayerNorm(module.out_channels, **kwargs)
    if norm == "time_group_norm":
        if causal:
            raise ValueError("Time-wise group normalization does not support causal evaluation.")
        if not isinstance(
            module,
            (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d),
        ):
            raise TypeError("Group normalization requires a convolution module.")
        return nn.GroupNorm(1, module.out_channels, **kwargs)
    return nn.Identity()


def get_extra_padding_for_conv1d(
    value: Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> int:
    """Return the right padding required to preserve all input samples."""
    if value.ndim != 3:
        raise ValueError("Streamable 1-D convolution expects [batch, channels, time].")
    if min(kernel_size, stride) <= 0 or padding_total < 0:
        raise ValueError("Convolution geometry must be positive with non-negative padding.")
    length = value.shape[-1]
    frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(frames) - 1) * stride + kernel_size - padding_total
    return int(ideal_length - length)


def pad_for_conv1d(
    value: Tensor,
    kernel_size: int,
    stride: int,
    padding_total: int = 0,
) -> Tensor:
    """Right-pad a waveform so the final convolution window is complete."""
    extra = get_extra_padding_for_conv1d(
        value,
        kernel_size,
        stride,
        padding_total,
    )
    return F.pad(value, (0, extra))


def pad1d(
    value: Tensor,
    paddings: tuple[int, int],
    *,
    mode: str = "constant",
    pad_value: float = 0.0,
) -> Tensor:
    """Pad the temporal axis, including reflect mode for very short inputs."""
    if value.ndim < 1:
        raise ValueError("Padding requires a tensor with a temporal axis.")
    left, right = paddings
    if left < 0 or right < 0:
        raise ValueError("Padding values must be non-negative.")
    normalized_mode = "constant" if mode in {"zero", "zeros"} else mode
    if normalized_mode == "reflect":
        length = value.shape[-1]
        maximum = max(left, right)
        extra = max(0, maximum - length + 1)
        if extra:
            value = F.pad(value, (0, extra))
        padded = F.pad(value, paddings, mode="reflect")
        return padded[..., :padded.shape[-1] - extra] if extra else padded
    return F.pad(value, paddings, mode=normalized_mode, value=pad_value)


def unpad1d(value: Tensor, paddings: tuple[int, int]) -> Tensor:
    """Remove fixed left/right padding from the temporal axis."""
    left, right = paddings
    if left < 0 or right < 0:
        raise ValueError("Padding values must be non-negative.")
    if left + right > value.shape[-1]:
        raise ValueError("Cannot remove more samples than the tensor contains.")
    end = value.shape[-1] - right
    return value[..., left:end] if right else value[..., left:]


class NormConv1d(nn.Module):
    """Conv1d plus the normalization selected by the Encodec graph."""

    def __init__(
        self,
        *args: Any,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(
            self.conv,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.norm_type = norm

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(self.conv(value))


class NormConvTranspose1d(nn.Module):
    """ConvTranspose1d plus the selected Encodec normalization."""

    def __init__(
        self,
        *args: Any,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.convtr = apply_parametrization_norm(
            nn.ConvTranspose1d(*args, **kwargs),
            norm,
        )
        self.norm = get_norm_module(
            self.convtr,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.norm_type = norm

    def forward(self, value: Tensor) -> Tensor:
        return self.norm(self.convtr(value))


class SConv1d(nn.Module):
    """Streamable 1-D convolution with exact Encodec padding semantics."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: Mapping[str, Any] | None = None,
        pad_mode: str = "reflect",
    ) -> None:
        super().__init__()
        if stride > 1 and dilation > 1:
            warnings.warn(
                "SConv1d was initialized with both stride and dilation "
                f"greater than one (kernel={kernel_size}, stride={stride}, "
                f"dilation={dilation}).",
                stacklevel=2,
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 3:
            raise ValueError("SConv1d expects [batch, channels, time].")
        convolution = self.conv.conv
        kernel_size = (convolution.kernel_size[0] - 1) * convolution.dilation[0] + 1
        stride = convolution.stride[0]
        padding_total = kernel_size - stride
        extra = get_extra_padding_for_conv1d(
            value,
            kernel_size,
            stride,
            padding_total,
        )
        if self.causal:
            paddings = padding_total, extra
        else:
            right = padding_total // 2
            left = padding_total - right
            paddings = left, right + extra
        return self.conv(
            pad1d(
                value,
                paddings,
                mode=self.pad_mode,
            ))


class SConvTranspose1d(nn.Module):
    """Streamable transposed convolution with deterministic trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        *,
        causal: bool = False,
        norm: str = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= trim_right_ratio <= 1:
            raise ValueError("`trim_right_ratio` must be in [0, 1].")
        if not causal and trim_right_ratio != 1:
            raise ValueError("Asymmetric right trimming requires causal convolution.")
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 3:
            raise ValueError("SConvTranspose1d expects [batch, channels, time].")
        convolution = self.convtr.convtr
        padding_total = convolution.kernel_size[0] - convolution.stride[0]
        output = self.convtr(value)
        if self.causal:
            right = math.ceil(padding_total * self.trim_right_ratio)
            left = padding_total - right
        else:
            right = padding_total // 2
            left = padding_total - right
        return unpad1d(output, (left, right))


class SLSTM(nn.Module):
    """Residual LSTM that accepts and returns convolutional layout."""

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        *,
        skip: bool = True,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(dimension, dimension, num_layers)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 3:
            raise ValueError("SLSTM expects [batch, channels, time].")
        sequence = value.permute(2, 0, 1)
        output, _ = self.lstm(sequence)
        if self.skip:
            output = output + sequence
        return output.permute(1, 2, 0)


def _activation_parameters(
    name: str,
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if parameters is not None:
        return dict(parameters)
    return {"alpha": 1.0} if name == "ELU" else {}


def _activation(name: str, parameters: Mapping[str, Any] | None) -> nn.Module:
    activation_type = getattr(nn, name, None)
    if activation_type is None or not isinstance(activation_type, type):
        raise ValueError(f"Unknown PyTorch activation {name!r}.")
    return activation_type(**_activation_parameters(name, parameters))


class SEANetResnetBlock(nn.Module):
    """Residual SEANet block used in both released Encodec graphs."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_sizes: Sequence[int] = (3, 1),
        dilations: Sequence[int] = (1, 1),
        activation: str = "ELU",
        activation_params: Mapping[str, Any] | None = None,
        norm: str = "weight_norm",
        norm_params: Mapping[str, Any] | None = None,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ) -> None:
        super().__init__()
        kernels = tuple(kernel_sizes)
        dilation_values = tuple(dilations)
        if len(kernels) != len(dilation_values) or not kernels:
            raise ValueError("Residual kernel sizes and dilations must have equal non-zero length.")
        if compress <= 0 or dim % compress:
            raise ValueError("Residual compression must divide the channel dimension.")
        hidden = dim // compress
        block: list[nn.Module] = []
        for index, (kernel_size, dilation) in enumerate(
            zip(kernels, dilation_values, strict=True),
        ):
            in_channels = dim if index == 0 else hidden
            out_channels = dim if index == len(kernels) - 1 else hidden
            block.extend((
                _activation(activation, activation_params),
                SConv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ))
        self.block = nn.Sequential(*block)
        if true_skip:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )

    def forward(self, value: Tensor) -> Tensor:
        return self.shortcut(value) + self.block(value)


class SEANetEncoder(nn.Module):
    """SEANet analysis transform with the official module namespace."""

    def __init__(
        self,
        *,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: Sequence[int] = (8, 5, 4, 2),
        activation: str = "ELU",
        activation_params: Mapping[str, Any] | None = None,
        norm: str = "weight_norm",
        norm_params: Mapping[str, Any] | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(tuple(ratios)))
        self.n_residual_layers = n_residual_layers
        self.hop_length = math.prod(self.ratios)

        activation_values = _activation_parameters(
            activation,
            activation_params,
        )
        multiplier = 1
        modules: list[nn.Module] = [
            SConv1d(
                channels,
                multiplier * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        for ratio in self.ratios:
            for residual_index in range(n_residual_layers):
                modules.append(
                    SEANetResnetBlock(
                        multiplier * n_filters,
                        kernel_sizes=(residual_kernel_size, 1),
                        dilations=(dilation_base**residual_index, 1),
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_values,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    ))
            modules.extend((
                _activation(activation, activation_values),
                SConv1d(
                    multiplier * n_filters,
                    multiplier * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ))
            multiplier *= 2
        if lstm:
            modules.append(SLSTM(multiplier * n_filters, num_layers=lstm))
        modules.extend((
            _activation(activation, activation_values),
            SConv1d(
                multiplier * n_filters,
                dimension,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ))
        self.model = nn.Sequential(*modules)

    def forward(self, value: Tensor) -> Tensor:
        return self.model(value)


class SEANetDecoder(nn.Module):
    """SEANet synthesis transform with the official module namespace."""

    def __init__(
        self,
        *,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: Sequence[int] = (8, 5, 4, 2),
        activation: str = "ELU",
        activation_params: Mapping[str, Any] | None = None,
        final_activation: str | None = None,
        final_activation_params: Mapping[str, Any] | None = None,
        norm: str = "weight_norm",
        norm_params: Mapping[str, Any] | None = None,
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = False,
        compress: int = 2,
        lstm: int = 2,
        trim_right_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = list(ratios)
        self.n_residual_layers = n_residual_layers
        self.hop_length = math.prod(self.ratios)

        activation_values = _activation_parameters(
            activation,
            activation_params,
        )
        multiplier = 2**len(self.ratios)
        modules: list[nn.Module] = [
            SConv1d(
                dimension,
                multiplier * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        if lstm:
            modules.append(SLSTM(multiplier * n_filters, num_layers=lstm))
        for ratio in self.ratios:
            modules.extend((
                _activation(activation, activation_values),
                SConvTranspose1d(
                    multiplier * n_filters,
                    multiplier * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                ),
            ))
            for residual_index in range(n_residual_layers):
                modules.append(
                    SEANetResnetBlock(
                        multiplier * n_filters // 2,
                        kernel_sizes=(residual_kernel_size, 1),
                        dilations=(dilation_base**residual_index, 1),
                        activation=activation,
                        activation_params=activation_values,
                        norm=norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    ))
            multiplier //= 2
        modules.extend((
            _activation(activation, activation_values),
            SConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ))
        if final_activation is not None:
            modules.append(_activation(final_activation, final_activation_params))
        self.model = nn.Sequential(*modules)

    def forward(self, value: Tensor) -> Tensor:
        return self.model(value)


__all__ = [
    "CONV_NORMALIZATIONS",
    "ConvLayerNorm",
    "NormConv1d",
    "NormConvTranspose1d",
    "SConv1d",
    "SConvTranspose1d",
    "SEANetDecoder",
    "SEANetEncoder",
    "SEANetResnetBlock",
    "SLSTM",
    "apply_parametrization_norm",
    "get_extra_padding_for_conv1d",
    "get_norm_module",
    "pad1d",
    "pad_for_conv1d",
    "unpad1d",
]
