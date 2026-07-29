"""Composable native Transformer blocks for speech encoders and decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn
from torch.nn import functional

from voicehub.neural.attention import MultiHeadAttention
from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.normalization import Float32LayerNorm, RMSNorm


@dataclass(frozen=True)
class TransformerLayerConfig:
    """Configuration shared by homogeneous Transformer layers."""

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    activation: str = "swiglu"
    normalization: str = "rmsnorm"
    normalization_epsilon: float = 1e-6
    attention_bias: bool = False
    feed_forward_bias: bool = False
    causal: bool = False
    cross_attention: bool = False
    rotary_dimension: int = 0
    rotary_base: float = 10_000.0

    def __post_init__(self) -> None:
        for name in (
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("`num_attention_heads` must divide `hidden_size`.")
        if self.activation not in {"gelu", "silu", "swiglu"}:
            raise ValueError("Transformer activation must be gelu, silu, or swiglu.")
        if self.normalization not in {"layernorm", "rmsnorm"}:
            raise ValueError("Transformer normalization must be layernorm or rmsnorm.")
        for name in ("attention_dropout", "residual_dropout"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"`{name}` must be in [0, 1).")


class FeedForward(nn.Module):
    """GELU/SiLU MLP or gated SwiGLU projection."""

    def __init__(self, config: TransformerLayerConfig) -> None:
        super().__init__()
        self.activation = config.activation
        if config.activation == "swiglu":
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=config.feed_forward_bias,
            )
            self.up_proj = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=config.feed_forward_bias,
            )
        else:
            self.up_proj = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=config.feed_forward_bias,
            )
            self.gate_proj = None
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=config.feed_forward_bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.activation == "swiglu":
            hidden = functional.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        elif self.activation == "silu":
            hidden = functional.silu(self.up_proj(inputs))
        else:
            hidden = functional.gelu(self.up_proj(inputs))
        return self.down_proj(hidden)


def _normalization(config: TransformerLayerConfig) -> nn.Module:
    if config.normalization == "rmsnorm":
        return RMSNorm(
            config.hidden_size,
            epsilon=config.normalization_epsilon,
        )
    return Float32LayerNorm(
        config.hidden_size,
        eps=config.normalization_epsilon,
    )


class TransformerLayer(nn.Module):
    """Pre-normalized self-attention block with optional cross-attention."""

    def __init__(
        self,
        config: TransformerLayerConfig,
        *,
        layer_index: int,
    ) -> None:
        super().__init__()
        if isinstance(layer_index, bool) or not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError("Transformer `layer_index` must be non-negative.")
        self.layer_index = layer_index
        self.self_attention_norm = _normalization(config)
        self.self_attention = MultiHeadAttention(
            config.hidden_size,
            config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_dropout=config.attention_dropout,
            bias=config.attention_bias,
            causal=config.causal,
            rotary_dimension=config.rotary_dimension,
            rotary_base=config.rotary_base,
        )
        if config.cross_attention:
            self.cross_attention_norm = _normalization(config)
            self.cross_attention = MultiHeadAttention(
                config.hidden_size,
                config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                attention_dropout=config.attention_dropout,
                bias=config.attention_bias,
                causal=False,
            )
        else:
            self.cross_attention_norm = None
            self.cross_attention = None
        self.feed_forward_norm = _normalization(config)
        self.feed_forward = FeedForward(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        cache: DynamicKVCache | None = None,
        use_cache: bool = False,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, DynamicKVCache | None, dict[str, Tensor | None]]:
        attention = self.self_attention(
            self.self_attention_norm(hidden_states),
            attention_mask=attention_mask,
            cache=cache,
            layer_index=self.layer_index * 2,
            use_cache=use_cache,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.residual_dropout(attention.hidden_states)
        cross_weights = None
        if self.cross_attention is not None:
            if encoder_hidden_states is None:
                raise ValueError("Cross-attention layer requires encoder hidden states.")
            cross = self.cross_attention(
                self.cross_attention_norm(hidden_states),
                attention_mask=encoder_attention_mask,
                key_value_states=encoder_hidden_states,
                cache=attention.cache,
                layer_index=self.layer_index * 2 + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = hidden_states + self.residual_dropout(cross.hidden_states)
            cache = cross.cache
            cross_weights = cross.weights
        else:
            cache = attention.cache
        hidden_states = hidden_states + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(hidden_states)))
        return hidden_states, cache, {
            "self_attention": attention.weights,
            "cross_attention": cross_weights,
        }


class TransformerStack(nn.Module):
    """Homogeneous stack returning hidden states and the explicit cache."""

    def __init__(
        self,
        config: TransformerLayerConfig,
        *,
        num_layers: int,
        final_normalization: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("Transformer `num_layers` must be positive.")
        self.layers = nn.ModuleList(
            TransformerLayer(config, layer_index=index) for index in range(num_layers))
        self.final_norm = (_normalization(config) if final_normalization else nn.Identity())

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        cache: DynamicKVCache | None = None,
        use_cache: bool = False,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, DynamicKVCache | None, tuple[dict[str, Any], ...]]:
        attentions = []
        for layer in self.layers:
            hidden_states, cache, layer_attentions = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                cache=cache,
                use_cache=use_cache,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            if output_attentions:
                attentions.append(layer_attentions)
        return self.final_norm(hidden_states), cache, tuple(attentions)


__all__ = [
    "FeedForward",
    "TransformerLayer",
    "TransformerLayerConfig",
    "TransformerStack",
]
