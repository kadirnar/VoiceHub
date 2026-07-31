from math import floor, log, pi
from typing import Any, List, Optional, Sequence, Tuple, Union

from .utils import *

import torch
import torch.nn as nn
from torch import Tensor, einsum

from voicehub.optimization.diffusion_cache import DiffusionCacheMixin


class _ChannelsFirst(nn.Module):
    def forward(self, value: Tensor) -> Tensor:
        return value.transpose(1, 2)


"""
Utils
"""

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x, s):
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)
                
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)
        
        
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)

class StyleTransformer1d(DiffusionCacheMixin, nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        context_features_multiplier: int = 1,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    style_dim=context_features,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            _ChannelsFirst(),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )
        
        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        if use_context_time or use_context_features:
            context_mapping_features = channels + context_embedding_features

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )
        
        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )
            
        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=context_embedding_features
        )
        self._initialize_diffusion_cache()
        

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = torch.stack(items).sum(dim=0)
            mapping = self.to_mapping(mapping)

        return mapping
            
    def run(self, x, time, embedding, features, *, diffusion_cache_lane="conditional"):
        
        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], axis=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)
        
        x = self._run_diffusion_blocks(
            self.blocks,
            x,
            lambda block, value: block(value + mapping, features),
            cache_lane=diffusion_cache_lane,
        )
        
        x = x.mean(axis=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)
        
        return x
        
    def forward(self, x: Tensor, 
                time: Tensor, 
                embedding_mask_proba: float = 0.0,
                embedding: Optional[Tensor] = None, 
                features: Optional[Tensor] = None,
                embedding_scale: float = 1.0,
                diffusion_cache_lane: str = "conditional") -> Tensor:
        
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(
                x,
                time,
                embedding=embedding,
                features=features,
                diffusion_cache_lane=f"{diffusion_cache_lane}:conditional",
            )
            out_masked = self.run(
                x,
                time,
                embedding=fixed_embedding,
                features=features,
                diffusion_cache_lane=f"{diffusion_cache_lane}:unconditional",
            )
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(
                x,
                time,
                embedding=embedding,
                features=features,
                diffusion_cache_lane=diffusion_cache_lane,
            )
        
        return x


class StyleTransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        style_dim: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = StyleAttention(
            features=features,
            style_dim=style_dim,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = StyleAttention(
                features=features,
                style_dim=style_dim,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, s: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x, s) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, s, context=context) + x
        x = self.feed_forward(x) + x
        return x

class StyleAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        style_dim: int,
        head_features: int,
        num_heads: int,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = AdaLayerNorm(style_dim, features)
        self.norm_context = AdaLayerNorm(style_dim, context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, s: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x, s), self.norm_context(context, s)
        
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return self.attention(q, k, v)
        
class Transformer1d(DiffusionCacheMixin, nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        context_features_multiplier: int = 1,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
        context_embedding_features: Optional[int] = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ]
        )

        self.to_out = nn.Sequential(
            _ChannelsFirst(),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )
        
        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        if use_context_time or use_context_features:
            context_mapping_features = channels + context_embedding_features

            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )
        
        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features
                ),
                nn.GELU(),
            )
            
        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=context_embedding_features
        )
        self._initialize_diffusion_cache()
        

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None
        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]
        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = torch.stack(items).sum(dim=0)
            mapping = self.to_mapping(mapping)

        return mapping
            
    def run(self, x, time, embedding, features, *, diffusion_cache_lane="conditional"):
        
        mapping = self.get_mapping(time, features)
        x = torch.cat([x.expand(-1, embedding.size(1), -1), embedding], axis=-1)
        mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)
        
        x = self._run_diffusion_blocks(
            self.blocks,
            x,
            lambda block, value: block(value + mapping),
            cache_lane=diffusion_cache_lane,
        )
        
        x = x.mean(axis=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)
        
        return x
        
    def forward(self, x: Tensor, 
                time: Tensor, 
                embedding_mask_proba: float = 0.0,
                embedding: Optional[Tensor] = None, 
                features: Optional[Tensor] = None,
                embedding_scale: float = 1.0,
                diffusion_cache_lane: str = "conditional") -> Tensor:
        
        b, device = embedding.shape[0], embedding.device
        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba, device=device
            )
            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(
                x,
                time,
                embedding=embedding,
                features=features,
                diffusion_cache_lane=f"{diffusion_cache_lane}:conditional",
            )
            out_masked = self.run(
                x,
                time,
                embedding=fixed_embedding,
                features=features,
                diffusion_cache_lane=f"{diffusion_cache_lane}:unconditional",
            )
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(
                x,
                time,
                embedding=embedding,
                features=features,
                diffusion_cache_lane=diffusion_cache_lane,
            )
        
        return x


"""
Attention Components
"""


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor, num_buckets: int, max_distance: int
    ):
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int) -> Tensor:
        i, j, device = num_queries, num_keys, self.relative_attention_bias.weight.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos.unsqueeze(0) - q_pos.unsqueeze(1)

        relative_position_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )

        bias = self.relative_attention_bias(relative_position_bucket)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias


def FeedForward(features: int, multiplier: int) -> nn.Module:
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


class AttentionBase(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        use_rel_pos: bool,
        out_features: Optional[int] = None,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads

        if use_rel_pos:
            assert exists(rel_pos_num_buckets) and exists(rel_pos_max_distance)
            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )
        if out_features is None:
            out_features = features
            
        self.to_out = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Split heads
        def split_heads(value: Tensor) -> Tensor:
            batch, steps, features = value.shape
            if features % self.num_heads:
                raise ValueError("Attention width must be divisible by heads.")
            return value.reshape(
                batch,
                steps,
                self.num_heads,
                features // self.num_heads,
            ).permute(0, 2, 1, 3)

        q, k, v = tuple(split_heads(value) for value in (q, k, v))
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = out.permute(0, 2, 1, 3).flatten(2)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return self.attention(q, k, v)


"""
Transformer Blocks
"""


class TransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = Attention(
            features=features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = Attention(
                features=features,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, context=context) + x
        x = self.feed_forward(x) + x
        return x



"""
Time Embeddings
"""


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device, half_dim = x.device, self.dim // 2
        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        freqs = x * self.weights.unsqueeze(0) * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )

class FixedEmbedding(nn.Module):
    def __init__(self, max_length: int, features: int):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, length, device = *x.shape[0:2], x.device
        assert_message = "Input sequence length must be <= max_length"
        assert length <= self.max_length, assert_message
        position = torch.arange(length, device=device)
        fixed_embedding = self.embedding(position)
        fixed_embedding = fixed_embedding.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        return fixed_embedding
