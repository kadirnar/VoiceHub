"""VoiceHub-native Encodec waveform model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import torch
from torch import Tensor, nn

from voicehub.optimization.protocols import OptimizationCompileTarget

from .configuration import (
    EncodecConfig,
    encodec_24khz_config,
    encodec_48khz_config,
)
from .layers import SEANetDecoder, SEANetEncoder
from .quantization import QuantizedResult, ResidualVectorQuantizer

EncodedFrame: TypeAlias = tuple[Tensor, Tensor | None]


@dataclass(frozen=True)
class EncodecTrainingOutput:
    """Differentiable codec output and losses for fine-tuning."""

    audio_values: Tensor
    encoded_frames: tuple[EncodedFrame, ...]
    commitment_loss: Tensor
    bandwidth: Tensor


def linear_overlap_add(frames: list[Tensor], stride: int) -> Tensor:
    """Overlap-add waveform segments with the official triangular window."""
    if not frames:
        raise ValueError("Overlap-add requires at least one frame.")
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("Overlap-add stride must be a positive integer.")
    first = frames[0]
    if first.ndim < 1 or first.shape[-1] == 0:
        raise ValueError("Overlap-add frames need a non-empty temporal axis.")
    prefix = first.shape[:-1]
    for frame in frames:
        if (
            frame.ndim != first.ndim
            or frame.shape[:-1] != prefix
            or frame.shape[-1] == 0
        ):
            raise ValueError("Every overlap-add frame must share non-time dimensions.")
        if frame.device != first.device or frame.dtype != first.dtype:
            raise ValueError("Every overlap-add frame must share device and dtype.")
        if frame.shape[-1] > first.shape[-1]:
            raise ValueError("Later overlap-add frames cannot exceed the first frame.")

    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]
    frame_length = first.shape[-1]
    positions = torch.linspace(
        0,
        1,
        frame_length + 2,
        device=first.device,
        dtype=first.dtype,
    )[1:-1]
    weight = 0.5 - (positions - 0.5).abs()
    sum_weight = first.new_zeros(total_size)
    output = first.new_zeros(*prefix, total_size)
    offset = 0
    for frame in frames:
        length = frame.shape[-1]
        output[..., offset:offset + length] += weight[:length] * frame
        sum_weight[offset:offset + length] += weight[:length]
        offset += stride
    if bool((sum_weight <= 0).any()):
        raise RuntimeError("Overlap-add geometry left uncovered output samples.")
    return output / sum_weight


class EncodecModel(nn.Module):
    """SEANet audio codec with bandwidth-aware residual quantization."""

    def __init__(
        self,
        encoder: SEANetEncoder,
        decoder: SEANetDecoder,
        quantizer: ResidualVectorQuantizer,
        target_bandwidths: list[float] | tuple[float, ...],
        sample_rate: int,
        channels: int,
        normalize: bool = False,
        segment: float | None = None,
        overlap: float = 0.01,
        name: str = "unset",
        *,
        config: EncodecConfig | None = None,
    ) -> None:
        super().__init__()
        if encoder.dimension != quantizer.dimension:
            raise ValueError("Encoder and quantizer dimensions must match.")
        if decoder.dimension != quantizer.dimension:
            raise ValueError("Decoder and quantizer dimensions must match.")
        if encoder.hop_length != decoder.hop_length:
            raise ValueError("Encoder and decoder hop lengths must match.")
        if channels != encoder.channels or channels != decoder.channels:
            raise ValueError("Model, encoder, and decoder channel counts must match.")
        if not target_bandwidths:
            raise ValueError("At least one target bandwidth is required.")
        if segment is not None and segment <= 0:
            raise ValueError("Segment duration must be positive.")
        if not 0 <= overlap < 1:
            raise ValueError("Segment overlap must be in [0, 1).")
        bits_per_codebook = int(math.log2(quantizer.bins))
        if 2**bits_per_codebook != quantizer.bins:
            raise ValueError("Quantizer bins must be a power of two.")

        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.target_bandwidths = tuple(float(value) for value in target_bandwidths)
        self.sample_rate = sample_rate
        self.channels = channels
        self.normalize = normalize
        self.segment = segment
        self.overlap = overlap
        self.frame_rate = math.ceil(sample_rate / encoder.hop_length)
        self.name = name
        self.bits_per_codebook = bits_per_codebook
        self.bandwidth: float | None = None
        self.config = config

    @classmethod
    def from_config(cls, config: EncodecConfig) -> EncodecModel:
        if not isinstance(config, EncodecConfig):
            raise TypeError("`config` must be an EncodecConfig.")
        common = {
            "channels": config.channels,
            "dimension": config.dimension,
            "n_filters": config.n_filters,
            "n_residual_layers": config.n_residual_layers,
            "ratios": config.ratios,
            "norm": config.model_norm,
            "kernel_size": config.kernel_size,
            "last_kernel_size": config.last_kernel_size,
            "residual_kernel_size": config.residual_kernel_size,
            "dilation_base": config.dilation_base,
            "causal": config.causal,
            "true_skip": config.true_skip,
            "compress": config.compress,
            "lstm": config.lstm,
        }
        encoder = SEANetEncoder(**common)
        decoder = SEANetDecoder(
            **common,
            trim_right_ratio=config.trim_right_ratio,
        )
        quantizer = ResidualVectorQuantizer(
            dimension=config.dimension,
            n_q=config.resolved_n_q,
            bins=config.bins,
            decay=config.decay,
            kmeans_init=config.kmeans_init,
            kmeans_iters=config.kmeans_iters,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
        )
        return cls(
            encoder,
            decoder,
            quantizer,
            config.target_bandwidths,
            config.sample_rate,
            config.channels,
            normalize=config.normalize,
            segment=config.segment,
            overlap=config.overlap,
            name=config.name,
            config=config,
        )

    @property
    def segment_length(self) -> int | None:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)

    @property
    def segment_stride(self) -> int | None:
        length = self.segment_length
        if length is None:
            return None
        return max(1, int((1 - self.overlap) * length))

    def _validate_audio(self, value: Tensor) -> None:
        if not isinstance(value, Tensor):
            raise TypeError("Encodec audio must be a PyTorch tensor.")
        if value.ndim != 3:
            raise ValueError("Encodec audio must have shape [batch, channels, samples].")
        if value.shape[0] == 0 or value.shape[-1] == 0:
            raise ValueError("Encodec audio batches and waveforms cannot be empty.")
        if value.shape[1] != self.channels:
            raise ValueError(
                f"This Encodec model expects {self.channels} channel(s), "
                f"but received {value.shape[1]}.")
        if not value.is_floating_point():
            raise TypeError("Encodec audio tensors must use a floating-point dtype.")

    def _segments(self, value: Tensor) -> list[Tensor]:
        length = value.shape[-1]
        segment_length = self.segment_length or length
        stride = self.segment_stride or length
        return [
            value[..., offset:offset + segment_length]
            for offset in range(0, length, stride)
        ]

    def _normalize_frame(self, value: Tensor) -> tuple[Tensor, Tensor | None]:
        duration = value.shape[-1] / self.sample_rate
        if self.segment is not None and duration > self.segment + 1e-5:
            raise ValueError("An encoded frame exceeds the configured segment duration.")
        if not self.normalize:
            return value, None
        mono = value.mean(dim=1, keepdim=True)
        volume = mono.square().mean(dim=2, keepdim=True).sqrt()
        scale = volume + 1e-8
        return value / scale, scale.view(-1, 1)

    def encode(self, value: Tensor) -> list[EncodedFrame]:
        """Encode audio into `[batch, codebooks, frames]` integer codes."""
        self._validate_audio(value)
        return [self._encode_frame(frame) for frame in self._segments(value)]

    def _encode_frame(self, value: Tensor) -> EncodedFrame:
        normalized, scale = self._normalize_frame(value)
        embedding = self.encoder(normalized)
        codes = self.quantizer.encode(
            embedding,
            self.frame_rate,
            self.bandwidth,
        )
        return codes.transpose(0, 1), scale

    def _validate_encoded_frame(self, frame: EncodedFrame) -> None:
        if not isinstance(frame, tuple) or len(frame) != 2:
            raise TypeError("Each encoded frame must be a `(codes, scale)` tuple.")
        codes, scale = frame
        if not isinstance(codes, Tensor) or codes.ndim != 3:
            raise ValueError("Encodec codes must have shape [batch, codebooks, frames].")
        if codes.shape[0] == 0 or codes.shape[1] == 0 or codes.shape[2] == 0:
            raise ValueError("Encodec code dimensions cannot be empty.")
        if codes.shape[1] > self.quantizer.n_q:
            raise ValueError("Encoded frame contains more codebooks than this model.")
        if codes.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("Encodec codes must use an integer dtype.")
        if bool(((codes < 0) | (codes >= self.quantizer.bins)).any()):
            raise ValueError("Encodec code values are outside the codebook range.")
        if scale is not None:
            if (
                not isinstance(scale, Tensor)
                or scale.shape != (codes.shape[0], 1)
                or not scale.is_floating_point()
            ):
                raise ValueError("Frame scale must have shape [batch, 1].")

    def decode(self, encoded_frames: list[EncodedFrame] | tuple[EncodedFrame, ...]) -> Tensor:
        """Decode one or more Encodec frames into a waveform."""
        if not isinstance(encoded_frames, (list, tuple)) or not encoded_frames:
            raise ValueError("Decode requires at least one encoded frame.")
        for frame in encoded_frames:
            self._validate_encoded_frame(frame)
        if self.segment_length is None:
            if len(encoded_frames) != 1:
                raise ValueError("Non-segmented Encodec expects exactly one frame.")
            return self._decode_frame_unchecked(*encoded_frames[0])
        decoded = [self._decode_frame_unchecked(*frame) for frame in encoded_frames]
        stride = self.segment_stride
        if stride is None:  # guarded by segment_length above
            raise RuntimeError("Segmented Encodec has no segment stride.")
        return linear_overlap_add(decoded, stride)

    def _decode_frame_unchecked(
        self,
        codes: Tensor,
        scale: Tensor | None,
    ) -> Tensor:
        """Decode one already validated frame without tensor-to-host sync."""
        embedding = self.quantizer.decode(codes.transpose(0, 1))
        output = self.decoder(embedding)
        if scale is not None:
            output = output * scale.to(
                device=output.device,
                dtype=output.dtype,
            ).view(-1, 1, 1)
        return output

    def codec_optimization_compile_targets(
        self,
        mode: str,
    ) -> tuple[OptimizationCompileTarget, ...]:
        if mode == "inference":
            return (OptimizationCompileTarget(
                "codec.decode.encodec.decode_frame",
                self,
                "_decode_frame_unchecked",
                component="decode",
            ), )
        if mode == "training":
            return (OptimizationCompileTarget(
                "codec.forward.encodec.forward",
                self,
                "forward",
                component="forward",
            ), )
        raise ValueError(f"Unsupported optimization mode {mode!r}.")

    def forward_quantized(self, value: Tensor) -> EncodecTrainingOutput:
        """Run the differentiable straight-through fine-tuning path."""
        self._validate_audio(value)
        decoded: list[Tensor] = []
        frames: list[EncodedFrame] = []
        penalties: list[Tensor] = []
        realized_bandwidths: list[Tensor] = []
        for segment in self._segments(value):
            normalized, scale = self._normalize_frame(segment)
            embedding = self.encoder(normalized)
            result: QuantizedResult = self.quantizer(
                embedding,
                self.frame_rate,
                self.bandwidth,
            )
            frame = result.codes.transpose(0, 1), scale
            frames.append(frame)
            output = self.decoder(result.quantized)
            if scale is not None:
                output = output * scale.to(
                    device=output.device,
                    dtype=output.dtype,
                ).view(-1, 1, 1)
            decoded.append(output)
            penalties.append(
                result.penalty
                if result.penalty is not None
                else output.new_zeros(())
            )
            realized_bandwidths.append(result.bandwidth)
        if self.segment_length is None:
            audio = decoded[0]
        else:
            stride = self.segment_stride
            if stride is None:
                raise RuntimeError("Segmented Encodec has no segment stride.")
            audio = linear_overlap_add(decoded, stride)
        return EncodecTrainingOutput(
            audio_values=audio[..., :value.shape[-1]],
            encoded_frames=tuple(frames),
            commitment_loss=torch.stack(penalties).mean(),
            bandwidth=torch.stack(realized_bandwidths).mean(),
        )

    def forward(self, value: Tensor) -> Tensor:
        # Match the released encode/decode path in evaluation. Training uses
        # the straight-through result so gradients reach the encoder.
        if self.training:
            return self.forward_quantized(value).audio_values
        return self.decode(self.encode(value))[..., :value.shape[-1]]

    def set_target_bandwidth(self, bandwidth: float) -> None:
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, (int, float))
            or float(bandwidth) not in self.target_bandwidths
        ):
            raise ValueError(
                f"This model supports bandwidths {self.target_bandwidths}; "
                f"received {bandwidth!r}.")
        self.bandwidth = float(bandwidth)

    @staticmethod
    def _get_model(
        target_bandwidths: list[float] | tuple[float, ...],
        sample_rate: int = 24_000,
        channels: int = 1,
        causal: bool = True,
        model_norm: str = "weight_norm",
        audio_normalize: bool = False,
        segment: float | None = None,
        name: str = "unset",
    ) -> EncodecModel:
        """Compatibility constructor matching the official Encodec API."""
        config = EncodecConfig(
            target_bandwidths=tuple(target_bandwidths),
            sample_rate=sample_rate,
            channels=channels,
            causal=causal,
            model_norm=model_norm,
            normalize=audio_normalize,
            segment=segment,
            name=name,
        )
        return EncodecModel.from_config(config)

    @staticmethod
    def encodec_model_24khz(
        pretrained: bool = False,
        repository: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        trust_official_pickle: bool = False,
    ) -> EncodecModel:
        """Build the exact causal 24 kHz mono graph.

        Network access is opt-in through ``pretrained=True``. Official legacy
        ``.th`` loading additionally requires ``trust_official_pickle=True``;
        converted Safetensors artifacts do not.
        """
        if repository is not None and not pretrained:
            raise ValueError("`repository` is only used when `pretrained=True`.")
        model = EncodecModel.from_config(encodec_24khz_config())
        if pretrained:
            from .checkpoint import load_pretrained_weights

            load_pretrained_weights(
                model,
                repository=repository,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_official_pickle=trust_official_pickle,
            )
        return model.eval()

    @staticmethod
    def encodec_model_48khz(
        pretrained: bool = False,
        repository: str | Path | None = None,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        trust_official_pickle: bool = False,
    ) -> EncodecModel:
        """Build the exact non-causal 48 kHz stereo graph."""
        if repository is not None and not pretrained:
            raise ValueError("`repository` is only used when `pretrained=True`.")
        model = EncodecModel.from_config(encodec_48khz_config())
        if pretrained:
            from .checkpoint import load_pretrained_weights

            load_pretrained_weights(
                model,
                repository=repository,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_official_pickle=trust_official_pickle,
            )
        return model.eval()


__all__ = [
    "EncodedFrame",
    "EncodecModel",
    "EncodecTrainingOutput",
    "linear_overlap_add",
]
