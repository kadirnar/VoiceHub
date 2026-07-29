from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from functools import partial, wraps
from os import path
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, int32
from torch.amp import autocast
from torch.nn import Module
from torch.nn.utils.parametrizations import weight_norm

from voicehub.hub import resolve_pretrained_file
from voicehub.models.vui.utils import decompile_state_dict


def exists(v):
    """Return ``True`` if *v* is not ``None``."""
    return v is not None


def default(*args):
    """Return the first non-``None`` argument, or ``None`` if all are
    ``None``."""
    for arg in args:
        if exists(arg):
            return arg
    return None


def maybe(fn):
    """Wrap *fn* so that it returns ``None`` unchanged instead of calling
    through."""

    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)

    return inner


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(Module):
    """Finite Scalar Quantization: maps continuous features to a fixed grid of
    levels per dimension."""

    def __init__(
        self,
        levels: list[int],
        dim: int | None = None,
        num_codebooks: int = 1,
        keep_num_codebooks_dim: bool | None = None,
        allowed_dtypes: tuple[torch.dtype, ...] = (torch.float32, torch.float64),
        channel_first: bool = True,
        projection_has_bias: bool = True,
        return_indices=True,
        force_quantization_f32: bool = True,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=int32)
        self.register_buffer("_levels", _levels, persistent=False)

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=int32)
        self.register_buffer("_basis", _basis, persistent=False)

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        self.channel_first = channel_first

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim, bias=projection_has_bias)
            if has_projections else nn.Identity())
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim, bias=projection_has_bias)
            if has_projections else nn.Identity())

        self.has_projections = has_projections

        self.return_indices = return_indices
        if return_indices:
            self.codebook_size = self._levels.prod().item()
            implicit_codebook = self._indices_to_codes(torch.arange(self.codebook_size))
            self.register_buffer("implicit_codebook", implicit_codebook, persistent=False)

        self.allowed_dtypes = allowed_dtypes
        self.force_quantization_f32 = force_quantization_f32

    def bound(self, z, eps: float = 1e-3):
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z):
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized):
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat):
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def _indices_to_codes(self, indices):
        level_indices = self.indices_to_level_indices(indices)
        codes = self._scale_and_shift_inverse(level_indices)
        return codes

    def codes_to_indices(self, zhat):
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(dim=-1).to(int32)

    def indices_to_level_indices(self, indices):
        """Converts indices to indices at each level, perhaps needed for a
        transformer with factorized embeddings."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered

    def indices_to_codes(self, indices):
        """Inverse of `codes_to_indices`."""
        assert exists(indices)

        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))

        codes = self._indices_to_codes(indices)

        if self.keep_num_codebooks_dim:
            codes = codes.flatten(start_dim=-2)

        codes = self.project_out(codes)

        if is_img_or_video or self.channel_first:
            dimensions = tuple(range(codes.ndim))
            codes = codes.permute(0, codes.ndim - 1, *dimensions[1:-1])

        return codes

    def forward(self, z: Tensor):
        """Einstein notation.

        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        device_type = z.device.type

        with torch.autocast(device_type=device_type, enabled=False):
            spatial_shape: tuple[int, ...] | None = None
            if self.channel_first:
                spatial_shape = tuple(z.shape[2:])
                dimensions = tuple(range(z.ndim))
                z = z.permute(0, *dimensions[2:], 1)
                z = z.reshape(z.shape[0], -1, z.shape[-1])

            assert (
                z.shape[-1] == self.dim
            ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

            z = self.project_in(z)

            z = z.reshape(
                z.shape[0],
                z.shape[1],
                self.num_codebooks,
                self.codebook_dim,
            )

            # whether to force quantization step to be full precision or not

            force_f32 = self.force_quantization_f32
            quantization_context = (
                partial(autocast, device_type=device_type, enabled=False) if force_f32 else nullcontext)

            with quantization_context():
                orig_dtype = z.dtype

                if force_f32 and orig_dtype not in self.allowed_dtypes:
                    z = z.float()

                codes = self.quantize(z)

                # returning indices could be optional

                indices = None

                if self.return_indices:
                    indices = self.codes_to_indices(codes)

                codes = codes.flatten(start_dim=-2)

                codes = codes.type(orig_dtype)

            # project out

            out = self.project_out(codes)

            # reconstitute image or video dimensions

            if self.channel_first:
                if spatial_shape is None:
                    raise RuntimeError("FSQ lost its source spatial shape.")
                out = out.reshape(out.shape[0], *spatial_shape, out.shape[-1])
                dimensions = tuple(range(out.ndim))
                out = out.permute(0, out.ndim - 1, *dimensions[1:-1])
                if indices is not None:
                    indices = indices.reshape(
                        indices.shape[0],
                        *spatial_shape,
                        indices.shape[-1],
                    )

            if not self.keep_num_codebooks_dim and self.return_indices:
                indices = None if indices is None else indices.squeeze(-1)

            # return quantized output and indices

            return out, indices


def WNConv1d(*args, **kwargs):
    """Weight-normalised ``Conv1d`` factory."""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """Weight-normalised ``ConvTranspose1d`` factory."""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class _SwapTimeChannels(nn.Module):
    """Swap `[batch, channels, time]` and `[batch, time, channels]`."""

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("Expected a rank-three sequence tensor.")
        return values.transpose(1, 2)


@torch.jit.script
def snake(x, alpha):
    """Snake activation: ``x + (1/alpha) * sin^2(alpha * x)``."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Learnable Snake activation for 1-D signals with per-channel
    frequency."""

    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    """Dilated residual convolution block with Snake activations."""

    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    """Encoder down-sampling block: residual units followed by a strided
    convolution."""

    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Multi-stage convolutional encoder that progressively down-samples the
    waveform."""

    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, d_model, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in strides:
            d_model *= 2
            self.block += [EncoderBlock(d_model, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Decoder up-sampling block: transposed convolution followed by residual
    units."""

    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """Multi-stage convolutional decoder that progressively up-samples to a
    waveform."""

    def __init__(
        self,
        input_channel: int,
        channels: int,
        rates: list[int],
        d_out: int = 1,
    ):
        super().__init__()

        # Add first conv layer
        layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # Add upsampling + MRF blocks
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2**(i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        # Add final conv layer
        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    # @torch.compile(dynamic=True)
    def forward(self, z: Tensor):
        return self.model(z)


class FiniteScalarQuantize(nn.Module):
    """Single-codebook FSQ layer with optional strided down-sampling and MLP
    post-processing."""

    def __init__(self, latent_dim: int, levels: list[int], *, stride: int = 1, mlp: bool = False):
        super().__init__()

        self.stride = stride

        codebook_dim = len(levels)

        self.in_proj = WNConv1d(latent_dim, codebook_dim, kernel_size=1)
        self.quantize = FSQ(levels=levels, channel_first=True)
        self.out_proj = WNConv1d(codebook_dim, latent_dim, kernel_size=1)

        if mlp:
            self.mlp = nn.Sequential(
                _SwapTimeChannels(),
                nn.Linear(latent_dim, 4 * latent_dim),
                nn.GELU(),
                nn.Linear(4 * latent_dim, latent_dim),
                _SwapTimeChannels(),
            )
        else:
            self.mlp = None

    def from_indices(self, indices: Tensor):
        B, T = indices.size()
        z_q = self.quantize.indices_to_codes(indices)
        z_q = self.out_proj(z_q)
        return z_q

    def forward(self, z: Tensor, *args):
        if self.stride > 1:
            z = F.avg_pool1d(z, self.stride, stride=self.stride)

        z_e = self.in_proj(z)  # z_e : (B x D x T)

        # we're channels first
        # scale = scale.unsqueeze(-1)

        # z_e = z_e / scale
        z_q, indices = self.quantize(z_e)
        # z_q = z_q * scale

        z_q = self.out_proj(z_q)

        if self.stride > 1:
            z_e = z_e.repeat_interleave(self.stride, dim=-1)
            z_q = z_q.repeat_interleave(self.stride, dim=-1)
            indices = indices.repeat_interleave(self.stride, dim=-1)

        if self.mlp is not None:
            z_q = self.mlp(z_q)

        return z_q, indices, z_e


class ResidualFiniteScalarQuantize(nn.Module):
    """Residual vector quantisation using a cascade of FSQ codebooks."""

    def __init__(
        self,
        *,
        latent_dim: int,
        n_quantizers: int,
        levels: list[int],
        strides: list[int] | None = None,
        quantizer_dropout: float = 0.0,
        mlp: bool = False,
    ):
        super().__init__()

        self.n_quantizers = n_quantizers
        self.quantizer_dropout = quantizer_dropout

        strides = [1] * n_quantizers if strides is None else strides

        assert (len(strides) == n_quantizers), "Strides must be provided for each codebook"

        scales = []
        quantizers = []
        levels_tensor = torch.tensor(levels, dtype=torch.float32)

        for i in range(n_quantizers):
            scales.append((levels_tensor - 1)**-i)
            quantizers.append(
                FiniteScalarQuantize(latent_dim=latent_dim, levels=levels, stride=strides[i], mlp=mlp))

        self.quantizers = nn.ModuleList(quantizers)

        self.register_buffer("scales", torch.stack(scales), persistent=False)

        codebooks = [quantizer.quantize.implicit_codebook for quantizer in self.quantizers]
        self.codebooks = torch.stack(codebooks, dim=0)

    def from_indices(self, indices: Tensor):
        B, Q, T = indices.size()

        z_q = 0.0

        for i, quantizer in enumerate(self.quantizers):
            z_q_i = quantizer.from_indices(indices[:, i])
            z_q = z_q + z_q_i

        return z_q

    def forward(self, z: Tensor, n_quantizers: int | None = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and
        returns the corresponding codebook vectors.

        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
        """
        B = z.shape[0]
        z_q = 0
        residual = z

        indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_quantizers

        if self.training:
            n_quantizers = torch.ones((B, )) * self.n_quantizers + 1
            dropout = torch.randint(1, self.n_quantizers + 1, (B, ))
            n_dropout = int(B * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, indices_i, z_e_i = quantizer(residual)

            residual = residual - z_q_i.detach()

            mask = torch.full((B, ), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]

            indices.append(indices_i)
            latents.append(z_e_i)

        indices = torch.stack(indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return z_q, indices, latents


@dataclass(slots=True)
class FluacConfig:
    """Configuration for the Fluac neural audio codec (encoder + FSQ quantiser
    + decoder)."""

    sample_rate: int = 44100
    codebook_size: int | None = None
    encoder_dim: int = 64
    encoder_rates: list[int] = field(default_factory=lambda: [2, 4, 8, 8])
    quantizer_strides: list[int] | None = None  # SNAC style strides
    n_quantizers: int = 1
    fsq_levels: list[int] | None = field(default_factory=lambda: [8, 5, 5, 5])
    decoder_dim: int = 1536
    decoder_rates: list[int] = field(default_factory=lambda: [8, 8, 4, 2])

    def __post_init__(self) -> None:
        for name in (
                "sample_rate",
                "encoder_dim",
                "n_quantizers",
                "decoder_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"`{name}` must be an integer.")
            if value <= 0:
                raise ValueError(f"`{name}` must be positive.")
        if self.codebook_size is not None and (isinstance(self.codebook_size, bool) or
                                               not isinstance(self.codebook_size, int) or
                                               self.codebook_size <= 0):
            raise ValueError("`codebook_size` must be a positive integer or None.")
        for name in ("encoder_rates", "decoder_rates"):
            values = getattr(self, name)
            if (not isinstance(values, list) or not values or
                    any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                        for value in values)):
                raise ValueError(f"`{name}` must be a non-empty list of positive integers.")
            setattr(self, name, list(values))
        if (not isinstance(self.fsq_levels, list) or len(self.fsq_levels) < 1 or
                any(isinstance(value, bool) or not isinstance(value, int) or value < 2
                    for value in self.fsq_levels)):
            raise ValueError("`fsq_levels` must contain integers of at least two.")
        self.fsq_levels = list(self.fsq_levels)
        if self.quantizer_strides is not None:
            if (not isinstance(self.quantizer_strides, list) or
                    any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                        for value in self.quantizer_strides)):
                raise ValueError("`quantizer_strides` must contain positive integers or be None.")
            self.quantizer_strides = list(self.quantizer_strides)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> FluacConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Fluac configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def model_dump(self) -> dict[str, object]:
        return asdict(self)

    def dict(self) -> dict[str, object]:
        return self.model_dump()

    @property
    def hop_length(self) -> int:
        return math.prod(self.encoder_rates)

    @property
    def latent_dim(self) -> int:
        return self.encoder_dim * (2**len(self.encoder_rates))

    @property
    def effective_codebook_size(self) -> int:
        return math.prod(self.fsq_levels)


class Fluac(nn.Module):
    """Fluac: a lightweight neural audio codec with FSQ-based residual
    quantisation."""

    Q9_22KHZ = "fluac-22hz-22khz.pt"

    def __init__(self, config: FluacConfig):
        super().__init__()

        self.config = config

        self.encoder = Encoder(config.encoder_dim, config.encoder_rates, config.latent_dim)

        self.quantizer = ResidualFiniteScalarQuantize(
            latent_dim=config.latent_dim,
            n_quantizers=config.n_quantizers,
            levels=config.fsq_levels,
            strides=config.quantizer_strides,
        )

        self.decoder = Decoder(
            config.latent_dim,
            config.decoder_dim,
            config.decoder_rates,
        )

        self.apply(init_weights)

    @staticmethod
    def from_pretrained(
        name: str = Q9_22KHZ,
        *,
        config: FluacConfig | Mapping[str, object] | None = None,
        repo_id: str = "fluxions/vui",
        revision: str = "8dc2bd9993a8118b6e2b71f3d9d92d1deb80e5f7",
        cache_dir: str | None = None,
        token: str | bool | None = None,
        local_files_only: bool = False,
    ):
        source = Path(name).expanduser()
        is_native_directory = (source.is_dir() and (source / "codec.safetensors").is_file())
        is_native_checkpoint = (source.is_file() and source.suffix.lower() == ".safetensors")
        if is_native_directory or is_native_checkpoint:
            from voicehub.models.vui.checkpoint import load_vui_safetensors, resolve_native_vui_artifact

            artifact = resolve_native_vui_artifact(source)
            if (is_native_checkpoint and source.resolve() != artifact.codec_checkpoint):
                raise ValueError(
                    "Fluac.from_pretrained() requires the codec Safetensors "
                    "file, not the Vui model component.")
            checkpoint_path = artifact.codec_checkpoint
            if config is None:
                config = artifact.codec_config
            if isinstance(config, FluacConfig):
                model_config = config
            elif isinstance(config, Mapping):
                model_config = FluacConfig.from_dict(config)
            else:  # pragma: no cover - guarded by artifact validation
                raise TypeError("Native Fluac configuration must be a mapping.")
            generator = Fluac(model_config).eval()
            generator.load_state_dict(
                load_vui_safetensors(
                    checkpoint_path,
                    component="codec",
                ),
                strict=True,
            )
            return generator

        if path.exists(name):
            checkpoint_path = name
        else:
            checkpoint_path = resolve_pretrained_file(
                repo_id,
                name,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                local_files_only=local_files_only,
            )

        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        checkpoint_config = checkpoint["config"]
        if "model" in checkpoint_config:
            model_config = FluacConfig.from_dict(checkpoint_config["model"])
        else:
            model_config = FluacConfig.from_dict(checkpoint_config)

        generator = Fluac(model_config).eval()
        ckpt = decompile_state_dict(checkpoint["generator"])
        generator.load_state_dict(ckpt)
        return generator

    def pad(self, waveform: Tensor):
        T = waveform.size(-1)
        right_pad = math.ceil(T / self.config.hop_length) * self.config.hop_length - T
        waveform = F.pad(waveform, (0, right_pad))
        return waveform

    @torch.inference_mode()
    def from_indices(self, indices: Tensor):
        z_q = self.quantizer.from_indices(indices)
        waveform = self.decoder(z_q)
        return waveform

    @torch.inference_mode()
    def encode(self, waveforms: Tensor, n_quantizers: int | None = None):
        # Ensure that waveforms is 3 dima
        waveforms = waveforms.flatten()[None][None]
        waveforms = self.pad(waveforms)
        B, C, T = waveforms.size()
        z = self.encoder(waveforms)
        z_q, codes, latents = self.quantizer(z, n_quantizers=n_quantizers)
        return codes

    def forward(self, waveforms: Tensor, n_quantizers: int | None = None):
        B, C, T = waveforms.size()
        waveforms = self.pad(waveforms)
        z = self.encoder(waveforms)
        z_q, codes, latents = self.quantizer(z, n_quantizers=n_quantizers)

        recons = self.decoder(z_q)
        recons = recons[..., :T]
        return {
            "recons": recons,
            "codes": codes,
        }

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def hz(self):
        return self.config.sample_rate / math.prod(self.config.encoder_rates)
