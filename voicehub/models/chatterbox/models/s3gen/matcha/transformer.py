"""Checkpoint-compatible transformer layers for Chatterbox's flow decoder.

The released model used a small subset of Diffusers' attention
components. These local implementations intentionally preserve that
subset's module names (``to_q``, ``to_k``, ``to_v``, ``to_out`` and
``ff.net.*``), so the official S3Gen Safetensors file loads strictly
without importing Diffusers.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional


class GELU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, hidden_states: Tensor) -> Tensor:
        return functional.gelu(self.proj(hidden_states), approximate=self.approximate)


class GEGLU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * functional.gelu(gate)


class ApproximateGELU(nn.Module):

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(hidden_states)
        return hidden_states * torch.sigmoid(1.702 * hidden_states)


class SnakeBeta(nn.Module):
    """Projected SnakeBeta activation used by compatible S3Gen variants."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ):
        super().__init__()
        self.in_features = [out_features]
        self.proj = nn.Linear(in_features, out_features)
        self.alpha_logscale = alpha_logscale
        initial = torch.zeros(self.in_features) if alpha_logscale else torch.ones(self.in_features)
        self.alpha = nn.Parameter(initial * alpha, requires_grad=alpha_trainable)
        self.beta = nn.Parameter(initial * alpha, requires_grad=alpha_trainable)
        self.no_div_by_zero = 1.0e-9

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(hidden_states)
        if self.alpha_logscale:
            alpha, beta = self.alpha.exp(), self.beta.exp()
        else:
            alpha, beta = self.alpha, self.beta
        periodic = torch.sin(hidden_states * alpha).square()
        return hidden_states + periodic / (beta + self.no_div_by_zero)


class FeedForward(nn.Module):
    """Diffusers-compatible feed-forward subset with stable state keys."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        activations: dict[str, nn.Module] = {
            "gelu": GELU(dim, inner_dim),
            "gelu-approximate": GELU(dim, inner_dim, approximate="tanh"),
            "geglu": GEGLU(dim, inner_dim),
            "geglu-approximate": ApproximateGELU(dim, inner_dim),
            "snakebeta": SnakeBeta(dim, inner_dim),
        }
        try:
            activation = activations[activation_fn]
        except KeyError as error:
            raise ValueError(f"Unsupported Chatterbox activation: {activation_fn!r}") from error
        layers: list[nn.Module] = [
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        ]
        if final_dropout:
            layers.append(nn.Dropout(dropout))
        self.net = nn.ModuleList(layers)

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class Attention(nn.Module):
    """Native attention with the state layout of Diffusers ``Attention``."""

    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: int | None = None,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        context_dim = query_dim if cross_attention_dim is None else cross_attention_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=True), nn.Dropout(dropout)])
        self.upcast_attention = upcast_attention

    @staticmethod
    def _attention_bias(mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            bias = torch.zeros(mask.shape, device=mask.device, dtype=dtype)
            bias = bias.masked_fill(~mask, torch.finfo(dtype).min)
        else:
            bias = mask.to(dtype=dtype)
        if bias.ndim == 2:
            bias = bias[:, None, None, :]
        elif bias.ndim == 3:
            bias = bias[:, None, :, :]
        if bias.ndim != 4:
            raise ValueError(f"attention mask must have 2-4 dimensions, got {bias.ndim}")
        return bias

    def forward(
        self,
        hidden_states: Tensor,
        *,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Tensor:
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, query_length, _ = hidden_states.shape
        key_length = context.shape[1]
        query = self.to_q(hidden_states).view(batch, query_length, self.heads, self.dim_head).transpose(1, 2)
        key = self.to_k(context).view(batch, key_length, self.heads, self.dim_head).transpose(1, 2)
        value = self.to_v(context).view(batch, key_length, self.heads, self.dim_head).transpose(1, 2)
        original_dtype = query.dtype
        if self.upcast_attention:
            query, key, value = query.float(), key.float(), value.float()
        bias = self._attention_bias(attention_mask, query.dtype)
        attended = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=0.0,
            scale=self.dim_head**-0.5,
        ).to(original_dtype)
        attended = attended.transpose(1, 2).reshape(batch, query_length, self.inner_dim)
        return self.to_out[1](self.to_out[0](attended))


class BasicTransformerBlock(nn.Module):
    """Native implementation of the exact transformer contract S3Gen uses."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: int | None = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        final_dropout: bool = False,
    ):
        super().__init__()
        if norm_type != "layer_norm" or num_embeds_ada_norm is not None:
            raise ValueError(
                "The released Chatterbox checkpoint only supports layer_norm transformer blocks.")
        self.only_cross_attention = only_cross_attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        if cross_attention_dim is not None or double_self_attention:
            self.norm2: nn.Module | None = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
            self.attn2: Attention | None = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=None if double_self_attention else cross_attention_dim,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
        )
        self._chunk_size: int | None = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: int | None, dim: int) -> None:
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        timestep: Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        class_labels: Tensor | None = None,
    ) -> Tensor:
        del timestep, class_labels
        kwargs = cross_attention_kwargs or {}
        attended = self.attn1(
            self.norm1(hidden_states),
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
            **kwargs,
        )
        hidden_states = hidden_states + attended
        if self.attn2 is not None and self.norm2 is not None:
            hidden_states = hidden_states + self.attn2(
                self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **kwargs,
            )
        normalized = self.norm3(hidden_states)
        if self._chunk_size is None:
            feed_forward = self.ff(normalized)
        else:
            if normalized.shape[self._chunk_dim] % self._chunk_size:
                raise ValueError("The feed-forward chunk size must evenly divide its dimension.")
            feed_forward = torch.cat(
                [self.ff(part) for part in normalized.split(self._chunk_size, self._chunk_dim)],
                dim=self._chunk_dim,
            )
        return hidden_states + feed_forward


__all__ = ["Attention", "BasicTransformerBlock", "FeedForward", "SnakeBeta"]
