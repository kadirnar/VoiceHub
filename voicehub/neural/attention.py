"""Native grouped-query attention with stable masks and explicit caches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.neural.cache import DynamicKVCache
from voicehub.neural.rotary import RotaryEmbedding, apply_rotary_embedding


@dataclass(frozen=True)
class AttentionOutput:
    """Attention result, optional weights, and the updated cache."""

    hidden_states: Tensor
    weights: Tensor | None = None
    cache: DynamicKVCache | None = None


def _expand_kv(tensor: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return tensor
    batch, heads, time, dimension = tensor.shape
    return (
        tensor[:, :, None, :, :]
        .expand(batch, heads, groups, time, dimension)
        .reshape(batch, heads * groups, time, dimension)
    )


def _attention_mask(
    mask: Tensor | None,
    *,
    batch_size: int,
    query_length: int,
    key_length: int,
    past_length: int,
    causal: bool,
    device: torch.device,
) -> Tensor | None:
    allowed: Tensor | None = None
    additive: Tensor | None = None
    if mask is not None:
        if not isinstance(mask, Tensor):
            raise TypeError("Attention mask must be a PyTorch tensor or None.")
        if mask.ndim == 2:
            if tuple(mask.shape) != (batch_size, key_length):
                raise ValueError(
                    f"Rank-two attention mask must have shape "
                    f"{(batch_size, key_length)!r}."
                )
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            if tuple(mask.shape) != (batch_size, query_length, key_length):
                raise ValueError(
                    "Rank-three attention mask must have shape "
                    "[batch, query, key]."
                )
            mask = mask[:, None, :, :]
        elif mask.ndim == 4:
            if (
                mask.shape[0] not in (1, batch_size)
                or mask.shape[-2] not in (1, query_length)
                or mask.shape[-1] != key_length
            ):
                raise ValueError("Rank-four attention mask is not broadcast-compatible.")
        else:
            raise ValueError("Attention mask must have rank two, three, or four.")
        mask = mask.to(device)
        if mask.dtype == torch.bool:
            allowed = mask
        elif mask.is_floating_point():
            additive = mask
        else:
            allowed = mask != 0
    if causal:
        query_positions = past_length + torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)
        causal_allowed = key_positions[None, :] <= query_positions[:, None]
        causal_allowed = causal_allowed[None, None, :, :]
        allowed = causal_allowed if allowed is None else allowed & causal_allowed
    if allowed is None:
        return additive
    boolean_bias = torch.zeros(
        (),
        dtype=torch.float32,
        device=device,
    ).expand(allowed.shape).clone()
    boolean_bias.masked_fill_(~allowed, -float("inf"))
    if additive is not None:
        boolean_bias = boolean_bias + additive.float()
    return boolean_bias


class MultiHeadAttention(nn.Module):
    """Self/cross attention supporting MHA, MQA, and grouped-query attention."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        *,
        num_key_value_heads: int | None = None,
        attention_dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
        rotary_dimension: int = 0,
        rotary_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(hidden_size, bool)
            or not isinstance(hidden_size, int)
            or hidden_size <= 0
        ):
            raise ValueError("Attention `hidden_size` must be positive.")
        if (
            isinstance(num_attention_heads, bool)
            or not isinstance(num_attention_heads, int)
            or num_attention_heads <= 0
            or hidden_size % num_attention_heads
        ):
            raise ValueError(
                "Attention heads must be positive and divide hidden size."
            )
        num_key_value_heads = (
            num_attention_heads
            if num_key_value_heads is None
            else num_key_value_heads
        )
        if (
            isinstance(num_key_value_heads, bool)
            or not isinstance(num_key_value_heads, int)
            or num_key_value_heads <= 0
            or num_attention_heads % num_key_value_heads
        ):
            raise ValueError(
                "Key/value heads must be positive and divide attention heads."
            )
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("Attention dropout must be in [0, 1).")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dimension = hidden_size // num_attention_heads
        self.causal = bool(causal)
        self.attention_dropout = float(attention_dropout)
        kv_size = num_key_value_heads * self.head_dimension
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, kv_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, kv_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        if rotary_dimension:
            if rotary_dimension > self.head_dimension:
                raise ValueError(
                    "Rotary dimension cannot exceed attention head dimension."
                )
            self.rotary = RotaryEmbedding(
                rotary_dimension,
                base=rotary_base,
            )
        else:
            self.rotary = None
        self.rotary_dimension = rotary_dimension

    def _shape(
        self,
        tensor: Tensor,
        heads: int,
    ) -> Tensor:
        batch, time, _ = tensor.shape
        return tensor.view(
            batch,
            time,
            heads,
            self.head_dimension,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        attention_mask: Tensor | None = None,
        key_value_states: Tensor | None = None,
        cache: DynamicKVCache | None = None,
        layer_index: int | None = None,
        use_cache: bool = False,
        position_ids: Tensor | None = None,
        output_attentions: bool = False,
    ) -> AttentionOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Attention hidden states must have shape "
                f"[batch, time, {self.hidden_size}]."
            )
        is_cross_attention = key_value_states is not None
        source = hidden_states if key_value_states is None else key_value_states
        if source.ndim != 3 or source.shape[0] != hidden_states.shape[0]:
            raise ValueError("Cross-attention source must share the query batch.")
        query = self._shape(
            self.q_proj(hidden_states),
            self.num_attention_heads,
        )
        key = self._shape(self.k_proj(source), self.num_key_value_heads)
        value = self._shape(self.v_proj(source), self.num_key_value_heads)

        past_length = 0
        if cache is not None:
            if layer_index is None:
                raise ValueError("Cached attention requires a `layer_index`.")
            existing = cache.get(layer_index)
            past_length = (
                0
                if existing is None or is_cross_attention
                else existing.sequence_length
            )
        if self.rotary is not None and not is_cross_attention:
            query_length = query.shape[-2]
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    past_length + query_length,
                    device=query.device,
                ).unsqueeze(0).expand(query.shape[0], -1)
            cosine, sine = self.rotary(position_ids, dtype=query.dtype)
            query, key = apply_rotary_embedding(
                query,
                key,
                cosine,
                sine,
                rotary_dimension=self.rotary_dimension,
            )

        if use_cache:
            if cache is None:
                cache = DynamicKVCache()
            if layer_index is None:
                raise ValueError("Cached attention requires a `layer_index`.")
            existing = cache.get(layer_index)
            if is_cross_attention and existing is not None:
                key, value = existing.key, existing.value
            else:
                entry = cache.update(
                    layer_index,
                    key,
                    value,
                    append=not is_cross_attention,
                )
                key, value = entry.key, entry.value
        key = _expand_kv(
            key,
            self.num_attention_heads // self.num_key_value_heads,
        )
        value = _expand_kv(
            value,
            self.num_attention_heads // self.num_key_value_heads,
        )
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        mask = _attention_mask(
            attention_mask,
            batch_size=query.shape[0],
            query_length=query_length,
            key_length=key_length,
            past_length=key_length - query_length if not is_cross_attention else 0,
            causal=self.causal and not is_cross_attention,
            device=query.device,
        )
        dropout = self.attention_dropout if self.training else 0.0
        weights = None
        if output_attentions:
            scores = torch.matmul(
                query.float(),
                key.float().transpose(-1, -2),
            ) / math.sqrt(self.head_dimension)
            if mask is not None:
                scores = scores + mask.float()
            weights = functional.softmax(scores, dim=-1)
            if dropout:
                weights = functional.dropout(
                    weights,
                    p=dropout,
                    training=True,
                )
            attended = torch.matmul(weights.to(value.dtype), value)
        else:
            attended = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask.to(query.dtype) if mask is not None else None,
                dropout_p=dropout,
                is_causal=False,
            )
        attended = attended.transpose(1, 2).contiguous().view(
            hidden_states.shape[0],
            query_length,
            self.hidden_size,
        )
        return AttentionOutput(
            hidden_states=self.o_proj(attended),
            weights=weights,
            cache=cache,
        )


__all__ = ["AttentionOutput", "MultiHeadAttention"]
