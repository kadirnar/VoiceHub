"""Native residual vector quantization for Encodec inference and training."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _broadcast_tensors(tensors: tuple[Tensor, ...], source: int = 0) -> None:
    """Keep EMA codebooks aligned when a process group is active."""
    if not _distributed_available():
        return
    for tensor in tensors:
        if torch.is_floating_point(tensor) or torch.is_complex(tensor):
            torch.distributed.broadcast(tensor, src=source)


def _ema_inplace(moving_average: Tensor, update: Tensor, decay: float) -> None:
    moving_average.mul_(decay).add_(update, alpha=1 - decay)


def _laplace_smoothing(
    value: Tensor,
    categories: int,
    epsilon: float = 1e-5,
) -> Tensor:
    return (value + epsilon) / (value.sum() + categories * epsilon)


def _uniform_init(*shape: int) -> Tensor:
    value = torch.empty(shape)
    nn.init.kaiming_uniform_(value)
    return value


def _sample_vectors(samples: Tensor, count: int) -> Tensor:
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("Vector sampling requires a non-empty [items, dimension] tensor.")
    if count <= 0:
        raise ValueError("Vector sample count must be positive.")
    if samples.shape[0] >= count:
        indices = torch.randperm(samples.shape[0], device=samples.device)[:count]
    else:
        indices = torch.randint(
            0,
            samples.shape[0],
            (count,),
            device=samples.device,
        )
    return samples[indices]


def _kmeans(
    samples: Tensor,
    clusters: int,
    iterations: int = 10,
) -> tuple[Tensor, Tensor]:
    """Run the release-compatible Euclidean k-means initializer."""
    if samples.ndim != 2:
        raise ValueError("K-means samples must have shape [items, dimension].")
    if clusters <= 0 or iterations <= 0:
        raise ValueError("K-means cluster and iteration counts must be positive.")
    dimension = samples.shape[-1]
    means = _sample_vectors(samples, clusters)
    bins = torch.zeros(clusters, device=samples.device, dtype=torch.long)
    for _ in range(iterations):
        distances = -(
            samples[:, None, :] - means[None, :, :]
        ).square().sum(dim=-1)
        buckets = distances.argmax(dim=-1)
        bins = torch.bincount(buckets, minlength=clusters)
        empty = bins == 0
        denominator = bins.masked_fill(empty, 1)
        new_means = samples.new_zeros(clusters, dimension)
        new_means.scatter_add_(
            0,
            buckets[:, None].expand(-1, dimension),
            samples,
        )
        new_means = new_means / denominator[:, None]
        means = torch.where(empty[:, None], means, new_means)
    return means, bins


class EuclideanCodebook(nn.Module):
    """EMA-updated Euclidean codebook with optional k-means initialization."""

    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ) -> None:
        super().__init__()
        if min(dim, codebook_size, kmeans_iters) <= 0:
            raise ValueError("Codebook dimensions and k-means iterations must be positive.")
        if not 0 < decay < 1 or epsilon <= 0:
            raise ValueError("Codebook decay must be in (0, 1) and epsilon must be positive.")
        if threshold_ema_dead_code < 0:
            raise ValueError("Dead-code threshold must be non-negative.")
        self.decay = decay
        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        embed = (
            torch.zeros(codebook_size, dim)
            if kmeans_init
            else _uniform_init(codebook_size, dim)
        )
        # These names and dtypes intentionally match Meta's published state.
        self.register_buffer(
            "inited",
            torch.tensor([float(not kmeans_init)]),
        )
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.no_grad()
    def init_embed_(self, data: Tensor) -> None:
        if bool(self.inited.item()):
            return
        embed, cluster_size = _kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
        )
        self.embed.copy_(embed)
        self.embed_avg.copy_(embed)
        self.cluster_size.copy_(cluster_size)
        self.inited.fill_(True)
        _broadcast_tensors(tuple(self.buffers()))

    @torch.no_grad()
    def replace_(self, samples: Tensor, mask: Tensor) -> None:
        replacement = torch.where(
            mask[:, None],
            _sample_vectors(samples, self.codebook_size),
            self.embed,
        )
        self.embed.copy_(replacement)

    @torch.no_grad()
    def expire_codes_(self, batch_samples: Tensor) -> None:
        if self.threshold_ema_dead_code == 0:
            return
        expired = self.cluster_size < self.threshold_ema_dead_code
        if not bool(expired.any()):
            return
        self.replace_(
            batch_samples.reshape(-1, batch_samples.shape[-1]),
            expired,
        )
        _broadcast_tensors(tuple(self.buffers()))

    @staticmethod
    def preprocess(value: Tensor) -> Tensor:
        return value.reshape(-1, value.shape[-1])

    def quantize(self, value: Tensor) -> Tensor:
        embedding = self.embed.t()
        distances = -(
            value.square().sum(1, keepdim=True)
            - 2 * value @ embedding
            + embedding.square().sum(0, keepdim=True)
        )
        return distances.argmax(dim=-1)

    @staticmethod
    def postprocess_emb(indices: Tensor, shape: torch.Size) -> Tensor:
        return indices.view(*shape[:-1])

    def dequantize(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.embed)

    def encode(self, value: Tensor) -> Tensor:
        shape = value.shape
        flattened = self.preprocess(value)
        indices = self.quantize(flattened)
        return self.postprocess_emb(indices, shape)

    def decode(self, indices: Tensor) -> Tensor:
        return self.dequantize(indices)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        shape, dtype = value.shape, value.dtype
        flattened = self.preprocess(value)
        self.init_embed_(flattened)
        flat_indices = self.quantize(flattened)
        one_hot = F.one_hot(flat_indices, self.codebook_size).to(dtype=dtype)
        indices = self.postprocess_emb(flat_indices, shape)
        quantized = self.dequantize(indices)

        if self.training:
            with torch.no_grad():
                self.expire_codes_(flattened)
                _ema_inplace(
                    self.cluster_size,
                    one_hot.sum(0),
                    self.decay,
                )
                embedding_sum = flattened.t() @ one_hot
                _ema_inplace(
                    self.embed_avg,
                    embedding_sum.t(),
                    self.decay,
                )
                cluster_size = (
                    _laplace_smoothing(
                        self.cluster_size,
                        self.codebook_size,
                        self.epsilon,
                    )
                    * self.cluster_size.sum()
                )
                self.embed.copy_(
                    self.embed_avg / cluster_size.unsqueeze(1),
                )
        return quantized, indices


class VectorQuantization(nn.Module):
    """One straight-through vector-quantization stage."""

    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ) -> None:
        super().__init__()
        resolved_dimension = dim if codebook_dim is None else codebook_dim
        if min(dim, resolved_dimension) <= 0:
            raise ValueError("Vector-quantizer dimensions must be positive.")
        if commitment_weight < 0:
            raise ValueError("Commitment weight must be non-negative.")
        needs_projection = resolved_dimension != dim
        self.project_in: nn.Module = (
            nn.Linear(dim, resolved_dimension)
            if needs_projection
            else nn.Identity()
        )
        self.project_out: nn.Module = (
            nn.Linear(resolved_dimension, dim)
            if needs_projection
            else nn.Identity()
        )
        self.epsilon = epsilon
        self.commitment_weight = commitment_weight
        self._codebook = EuclideanCodebook(
            dim=resolved_dimension,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self) -> Tensor:
        return self._codebook.embed

    @staticmethod
    def _channels_last(value: Tensor) -> Tensor:
        if value.ndim != 3:
            raise ValueError("Vector quantization expects [batch, channels, frames].")
        return value.transpose(1, 2)

    def encode(self, value: Tensor) -> Tensor:
        projected = self.project_in(self._channels_last(value))
        return self._codebook.encode(projected)

    def decode(self, indices: Tensor) -> Tensor:
        if indices.ndim != 2:
            raise ValueError("Codebook indices must have shape [batch, frames].")
        quantized = self.project_out(self._codebook.decode(indices))
        return quantized.transpose(1, 2)

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        projected = self.project_in(self._channels_last(value))
        quantized, indices = self._codebook(projected)
        if self.training:
            quantized = projected + (quantized - projected).detach()
            commitment_loss = F.mse_loss(quantized.detach(), projected)
            loss = commitment_loss * self.commitment_weight
        else:
            loss = projected.new_zeros(())
        output = self.project_out(quantized).transpose(1, 2)
        return output, indices, loss


class ResidualVectorQuantization(nn.Module):
    """A stack of vector quantizers following Encodec's residual algorithm."""

    def __init__(self, *, num_quantizers: int, **kwargs: Any) -> None:
        super().__init__()
        if num_quantizers <= 0:
            raise ValueError("Residual vector quantization needs at least one stage.")
        self.layers = nn.ModuleList(
            VectorQuantization(**kwargs)
            for _ in range(num_quantizers)
        )

    def _count(self, n_q: int | None) -> int:
        count = len(self.layers) if n_q is None else n_q
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("The number of quantizers must be an integer.")
        if not 1 <= count <= len(self.layers):
            raise ValueError(
                f"Requested {count} quantizers, but this model provides "
                f"{len(self.layers)}.")
        return count

    def forward(
        self,
        value: Tensor,
        n_q: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = self._count(n_q)
        quantized_output = torch.zeros_like(value)
        residual = value
        losses: list[Tensor] = []
        indices: list[Tensor] = []
        for layer in self.layers[:count]:
            quantized, stage_indices, loss = layer(residual)
            # The pinned upstream revision subtracts the straight-through
            # tensor here, which gives later residual stages a zero gradient
            # (facebookresearch/encodec#25). Detaching the residual update
            # preserves identical forward values and checkpoint compatibility
            # while restoring gradients to every selected stage.
            residual = residual - quantized.detach()
            quantized_output = quantized_output + quantized
            indices.append(stage_indices)
            losses.append(loss)
        return (
            quantized_output,
            torch.stack(indices),
            torch.stack(losses),
        )

    def encode(self, value: Tensor, n_q: int | None = None) -> Tensor:
        count = self._count(n_q)
        residual = value
        indices: list[Tensor] = []
        for layer in self.layers[:count]:
            stage_indices = layer.encode(residual)
            quantized = layer.decode(stage_indices)
            residual = residual - quantized
            indices.append(stage_indices)
        return torch.stack(indices)

    def decode(self, indices: Tensor) -> Tensor:
        if indices.ndim != 3:
            raise ValueError("Residual codes must have shape [quantizers, batch, frames].")
        count = indices.shape[0]
        self._count(count)
        if indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("Residual codes must use an integer dtype.")
        output: Tensor | None = None
        for layer, stage_indices in zip(
            self.layers[:count],
            indices,
            strict=True,
        ):
            quantized = layer.decode(stage_indices)
            output = quantized if output is None else output + quantized
        if output is None:  # guarded by _count, retained for static analysis
            raise RuntimeError("No residual codebooks were decoded.")
        return output


@dataclass
class QuantizedResult:
    """Differentiable result returned by :class:`ResidualVectorQuantizer`."""

    quantized: Tensor
    codes: Tensor
    bandwidth: Tensor
    penalty: Tensor | None = None
    metrics: dict[str, Tensor] = field(default_factory=dict)


class ResidualVectorQuantizer(nn.Module):
    """Bandwidth-aware Encodec residual vector quantizer."""

    def __init__(
        self,
        *,
        dimension: int = 256,
        n_q: int = 8,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
    ) -> None:
        super().__init__()
        if bins <= 1 or bins & (bins - 1):
            raise ValueError("Codebook bins must be a power of two greater than one.")
        self.n_q = n_q
        self.dimension = dimension
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vq = ResidualVectorQuantization(
            dim=dimension,
            codebook_size=bins,
            num_quantizers=n_q,
            decay=decay,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )

    def get_bandwidth_per_quantizer(self, frame_rate: int) -> float:
        if isinstance(frame_rate, bool) or not isinstance(frame_rate, int) or frame_rate <= 0:
            raise ValueError("`frame_rate` must be a positive integer.")
        return math.log2(self.bins) * frame_rate

    def get_num_quantizers_for_bandwidth(
        self,
        frame_rate: int,
        bandwidth: float | None = None,
    ) -> int:
        bandwidth_per_quantizer = self.get_bandwidth_per_quantizer(frame_rate)
        if bandwidth is None:
            return self.n_q
        if (
            isinstance(bandwidth, bool)
            or not isinstance(bandwidth, (int, float))
            or not math.isfinite(float(bandwidth))
            or bandwidth <= 0
        ):
            raise ValueError("`bandwidth` must be a positive finite value or None.")
        count = math.floor(float(bandwidth) * 1000 / bandwidth_per_quantizer)
        return min(self.n_q, max(1, count))

    def forward(
        self,
        value: Tensor,
        frame_rate: int,
        bandwidth: float | None = None,
    ) -> QuantizedResult:
        bandwidth_per_quantizer = self.get_bandwidth_per_quantizer(frame_rate)
        count = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        quantized, codes, commitment = self.vq(value, n_q=count)
        realized_bandwidth = value.new_tensor(count * bandwidth_per_quantizer)
        return QuantizedResult(
            quantized=quantized,
            codes=codes,
            bandwidth=realized_bandwidth,
            penalty=commitment.mean(),
        )

    def encode(
        self,
        value: Tensor,
        frame_rate: int,
        bandwidth: float | None = None,
    ) -> Tensor:
        count = self.get_num_quantizers_for_bandwidth(frame_rate, bandwidth)
        return self.vq.encode(value, n_q=count)

    def decode(self, codes: Tensor) -> Tensor:
        return self.vq.decode(codes)


__all__ = [
    "EuclideanCodebook",
    "QuantizedResult",
    "ResidualVectorQuantization",
    "ResidualVectorQuantizer",
    "VectorQuantization",
]
