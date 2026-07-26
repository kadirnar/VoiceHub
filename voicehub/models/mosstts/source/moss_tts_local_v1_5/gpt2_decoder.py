# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    _FLASH_ATTN_AVAILABLE = True
except Exception:
    flash_attn_func = None
    flash_attn_varlen_func = None
    pad_input = None
    unpad_input = None
    _FLASH_ATTN_AVAILABLE = False


@dataclass
class PackedSequenceMetadata:
    cu_seqlens: torch.Tensor
    max_seqlen: int
    indices: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None


def _is_static_kv_cache_layer(layer_past: object) -> bool:
    return isinstance(layer_past, dict) and bool(layer_past.get("static_kv_cache", False))


class MossTTSNanoGPT2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {dim}")
        self.dim = int(dim)
        self.base = float(base)
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        inv_freq = self._compute_inv_freq(device=device)
        freqs = torch.einsum("bs,d->bsd", position_ids.to(device=device, dtype=inv_freq.dtype), inv_freq)
        cos = freqs.cos().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
        sin = freqs.sin().repeat_interleave(2, dim=-1).unsqueeze(2).to(dtype=dtype)
        return cos, sin


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    even = hidden_states[..., ::2]
    odd = hidden_states[..., 1::2]
    return torch.stack((-odd, even), dim=-1).reshape_as(hidden_states)


def apply_rotary_pos_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    return (hidden_states * cos) + (rotate_half(hidden_states) * sin)


class MossTTSNanoGPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        inner_size = int(config.n_inner or 4 * hidden_size)
        self.fc_in = nn.Linear(hidden_size, inner_size)
        self.fc_out = nn.Linear(inner_size, hidden_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        return self.dropout(hidden_states)


class MossTTSNanoGPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int, attn_implementation: str) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        num_heads = int(config.num_attention_heads)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_attention_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.embed_dim = hidden_size
        self.layer_idx = layer_idx
        self.attn_implementation = attn_implementation
        self.attn_dropout = float(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.scale_attn_weights = bool(getattr(config, "scale_attn_weights", True))
        self.scale_attn_by_inverse_layer_idx = bool(getattr(config, "scale_attn_by_inverse_layer_idx", False))
        self.position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"Unsupported position_embedding_type={self.position_embedding_type!r}")

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
        self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.rotary_emb = None
        if self.position_embedding_type == "rope":
            self.rotary_emb = MossTTSNanoGPT2RotaryEmbedding(
                self.head_dim,
                base=float(getattr(config, "rope_base", 10000.0)),
            )

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 3:
            batch_size, seq_len, _ = tensor.shape
            return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
        if tensor.ndim == 2:
            total_tokens, _ = tensor.shape
            return tensor.view(total_tokens, self.num_heads, self.head_dim)
        raise ValueError(f"Unsupported tensor rank for attention split: {tensor.ndim}")

    def _merge_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            batch_size, seq_len, _, _ = tensor.shape
            return tensor.reshape(batch_size, seq_len, self.embed_dim)
        if tensor.ndim == 3:
            total_tokens, _, _ = tensor.shape
            return tensor.reshape(total_tokens, self.embed_dim)
        raise ValueError(f"Unsupported tensor rank for attention merge: {tensor.ndim}")

    def _causal_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        query_positions = torch.arange(query_length, device=device, dtype=torch.long)
        query_positions = query_positions + max(key_length - query_length, 0)
        key_positions = torch.arange(key_length, device=device, dtype=torch.long)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        causal = causal.unsqueeze(0).unsqueeze(0)
        if attention_mask is None:
            return causal
        key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
        return causal & key_mask

    def _eager_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scale = 1.0
        if self.scale_attn_weights:
            scale /= self.head_dim ** 0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale /= float(self.layer_idx + 1)

        scores = torch.matmul(query, key.transpose(-1, -2)) * scale
        causal_mask = self._causal_attention_mask(
            attention_mask=attention_mask,
            query_length=query.shape[-2],
            key_length=key.shape[-2],
            device=query.device,
        )
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attn_dropout > 0:
            probs = torch.dropout(probs, self.attn_dropout, train=True)
        output = torch.matmul(probs, value)
        return output.transpose(1, 2).contiguous()

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        mask = None
        if attention_mask is not None or query.shape[-2] != key.shape[-2]:
            mask = self._causal_attention_mask(
                attention_mask=attention_mask,
                query_length=query.shape[-2],
                key_length=key.shape[-2],
                device=query.device,
            )
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=mask is None,
        )
        return output.transpose(1, 2).contiguous()

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        packed_metadata: Optional[PackedSequenceMetadata],
    ) -> torch.Tensor:
        if not _FLASH_ATTN_AVAILABLE:
            raise ImportError("flash_attn is not installed, but attn_implementation='flash_attention_2' was requested.")
        if query.device.type != "cuda":
            raise ValueError("flash_attention_2 requires CUDA tensors.")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"flash_attention_2 requires fp16/bf16 tensors, but received dtype={query.dtype}."
            )

        dropout_p = self.attn_dropout if self.training else 0.0
        if packed_metadata is not None:
            if packed_metadata.indices is not None:
                query = query.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                key = key.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                value = value.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
            output = flash_attn_varlen_func(
                query,
                key,
                value,
                packed_metadata.cu_seqlens,
                packed_metadata.cu_seqlens,
                packed_metadata.max_seqlen,
                packed_metadata.max_seqlen,
                dropout_p=dropout_p,
                causal=True,
            )
            if packed_metadata.indices is None:
                return output
            return pad_input(
                output,
                packed_metadata.indices,
                packed_metadata.batch_size,
                packed_metadata.seq_len,
            )

        if attention_mask is None or bool(attention_mask.all()):
            return flash_attn_func(
                query,
                key,
                value,
                dropout_p=dropout_p,
                causal=True,
            )

        if query.shape[1] != key.shape[1]:
            query_attention_mask = attention_mask[:, -query.shape[1] :]
            unpadded_query, query_indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
                query,
                query_attention_mask,
            )
            unpadded_key, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(key, attention_mask)
            unpadded_value, _, _, _, _ = unpad_input(value, attention_mask)
            output = flash_attn_varlen_func(
                unpadded_query,
                unpadded_key,
                unpadded_value,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=dropout_p,
                causal=True,
            )
            return pad_input(output, query_indices, query.shape[0], query.shape[1])

        unpadded_query, indices, cu_seqlens, max_seqlen, _ = unpad_input(query, attention_mask)
        unpadded_key, _, _, _, _ = unpad_input(key, attention_mask)
        unpadded_value, _, _, _, _ = unpad_input(value, attention_mask)
        output = flash_attn_varlen_func(
            unpadded_query,
            unpadded_key,
            unpadded_value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=dropout_p,
            causal=True,
        )
        return pad_input(output, indices, query.shape[0], query.shape[1])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.embed_dim, dim=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if self.rotary_emb is not None:
            if position_ids is None:
                raise ValueError("position_ids must be provided when position_embedding_type='rope'.")
            cos, sin = self.rotary_emb(
                position_ids.to(device=query.device),
                device=query.device,
                dtype=query.dtype,
            )
            query = apply_rotary_pos_emb(query, cos, sin)
            key = apply_rotary_pos_emb(key, cos, sin)

        static_layer_past = layer_past is not None and _is_static_kv_cache_layer(layer_past)
        if static_layer_past:
            past_length = int(layer_past.get("length", 0))
            new_length = past_length + int(key.shape[1])
            key_cache = layer_past["key"]
            value_cache = layer_past["value"]
            if new_length > int(key_cache.shape[1]):
                raise ValueError(
                    f"Static KV cache is too short: need {new_length}, capacity={int(key_cache.shape[1])}."
                )
            key_cache[:, past_length:new_length].copy_(key)
            value_cache[:, past_length:new_length].copy_(value)
            key = key_cache[:, :new_length]
            value = value_cache[:, :new_length]
            layer_past["length"] = new_length
        elif layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat([past_key.to(device=key.device, dtype=key.dtype), key], dim=1)
            value = torch.cat([past_value.to(device=value.device, dtype=value.dtype), value], dim=1)

        present = layer_past if (use_cache and static_layer_past) else ((key, value) if use_cache else None)

        if self.attn_implementation == "flash_attention_2":
            attn_output = self._flash_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                packed_metadata=packed_metadata,
            )
        elif self.attn_implementation == "sdpa":
            attn_output = self._sdpa_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )
        else:
            attn_output = self._eager_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
            )

        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        return self.resid_dropout(attn_output), present


class MossTTSNanoGPT2Block(nn.Module):
    def __init__(self, config: GPT2Config, layer_idx: int, attn_implementation: str) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = MossTTSNanoGPT2Attention(config, layer_idx=layer_idx, attn_implementation=attn_implementation)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MossTTSNanoGPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present = self.attn(
            self.ln_1(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_metadata=packed_metadata,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present


class MossTTSNanoGPT2Model(nn.Module):
    def __init__(self, config: GPT2Config, attn_implementation: str = "eager") -> None:
        super().__init__()
        self.config = config
        self.attn_implementation = attn_implementation
        self.position_embedding_type = str(getattr(config, "position_embedding_type", "absolute")).lower()
        if self.position_embedding_type not in {"absolute", "rope"}:
            raise ValueError(f"Unsupported position_embedding_type={self.position_embedding_type!r}")
        hidden_size = int(config.hidden_size)
        self.wte = nn.Embedding(config.vocab_size, hidden_size)
        self.wpe = nn.Embedding(config.n_positions, hidden_size) if self.position_embedding_type == "absolute" else nn.Identity()
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [MossTTSNanoGPT2Block(config, layer_idx=index, attn_implementation=attn_implementation) for index in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init_std = float(self.config.initializer_range)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _normalize_num_sequences(
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        if num_sequences is None:
            diffs = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]
            return diffs.gt(0).sum(dim=-1).to(device=device, dtype=torch.long)
        if num_sequences.ndim == 0:
            num_sequences = num_sequences.unsqueeze(0)
        return num_sequences.to(device=device, dtype=torch.long)

    @staticmethod
    def _packed_segments_from_cu_seqlens(
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        cu_seqlens = cu_seqlens.to(device=device)
        batch_size, boundary_count = cu_seqlens.shape
        segment_slots = boundary_count - 1
        if segment_slots <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty

        counts = MossTTSNanoGPT2Model._normalize_num_sequences(cu_seqlens, num_sequences, device=device)
        counts = counts.clamp(min=0, max=segment_slots)
        segment_slots = int(counts.max().item()) if counts.numel() > 0 else 0
        if segment_slots <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        cu_seqlens = cu_seqlens[:, : segment_slots + 1]
        slot_ids = torch.arange(segment_slots, device=device).unsqueeze(0)
        valid_slots = slot_ids < counts.unsqueeze(1)

        starts = cu_seqlens[:, :-1].to(dtype=torch.long)
        ends = cu_seqlens[:, 1:].to(dtype=torch.long)
        lengths = (ends - starts).clamp_min(0)
        lengths = torch.where(valid_slots, lengths, torch.zeros((), dtype=torch.long, device=device))

        batch_ids = torch.arange(batch_size, device=device, dtype=torch.long).unsqueeze(1).expand(batch_size, segment_slots)
        batch_ids = batch_ids.reshape(-1)
        starts = starts.reshape(-1)
        lengths = lengths.reshape(-1)

        valid_segments = lengths.gt(0)
        valid_count = int(valid_segments.to(dtype=torch.long).sum().item())
        if valid_count <= 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty, empty
        if valid_count == lengths.numel():
            return batch_ids, starts, lengths

        valid_order = torch.argsort(valid_segments.to(dtype=torch.long), descending=True, stable=True)[:valid_count]
        return (
            batch_ids.index_select(0, valid_order),
            starts.index_select(0, valid_order),
            lengths.index_select(0, valid_order),
        )

    @staticmethod
    def _packed_token_indices(
        batch_ids: torch.Tensor,
        starts: torch.Tensor,
        lengths: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_tokens = int(lengths.sum().item())
        if total_tokens <= 0:
            empty = torch.empty(0, dtype=torch.long, device=lengths.device)
            return empty, empty

        segment_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=lengths.device, dtype=torch.long),
            lengths,
            output_size=total_tokens,
        )
        segment_starts = torch.cumsum(lengths, dim=0) - lengths
        positions = torch.arange(total_tokens, device=lengths.device, dtype=torch.long) - segment_starts[segment_ids]
        indices = batch_ids[segment_ids] * seq_len + starts[segment_ids] + positions
        return indices, positions

    @staticmethod
    def build_packed_position_ids(
        attention_mask: Optional[torch.Tensor],
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
        sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        batch_size = cu_seqlens.shape[0]
        seq_len = int(sequence_length or (cu_seqlens.shape[1] - 1))
        device = cu_seqlens.device
        position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        batch_ids, starts, lengths = MossTTSNanoGPT2Model._packed_segments_from_cu_seqlens(
            cu_seqlens,
            num_sequences,
            device,
        )
        if lengths.numel() > 0:
            indices, positions = MossTTSNanoGPT2Model._packed_token_indices(batch_ids, starts, lengths, seq_len)
            position_ids.view(-1).scatter_(0, indices, positions)
        if attention_mask is not None:
            position_ids = position_ids * attention_mask.to(dtype=position_ids.dtype)
        return position_ids

    @staticmethod
    def build_packed_metadata(
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_sequences: Optional[torch.Tensor],
    ) -> PackedSequenceMetadata:
        if cu_seqlens.ndim == 1:
            cu_seqlens = cu_seqlens.unsqueeze(0)
        device = hidden_states.device
        seq_len = hidden_states.shape[1]
        batch_ids, starts, lengths = MossTTSNanoGPT2Model._packed_segments_from_cu_seqlens(
            cu_seqlens,
            num_sequences,
            device,
        )
        if lengths.numel() == 0:
            raise ValueError("cu_seqlens did not describe any non-empty packed sequences.")

        indices, _ = MossTTSNanoGPT2Model._packed_token_indices(batch_ids, starts, lengths, seq_len)
        cumulative = torch.empty(lengths.numel() + 1, dtype=torch.int32, device=device)
        cumulative[0] = 0
        cumulative[1:] = lengths.to(dtype=torch.int32).cumsum(dim=0)
        return PackedSequenceMetadata(
            cu_seqlens=cumulative,
            max_seqlen=int(lengths.max().item()),
            indices=indices,
            batch_size=hidden_states.shape[0],
            seq_len=hidden_states.shape[1],
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cu_seqlens: Optional[torch.Tensor] = None,
        num_sequences: Optional[torch.Tensor] = None,
    ) -> BaseModelOutputWithPast:
        del input_ids, output_attentions

        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided.")

        use_cache = bool(use_cache)
        if use_cache and cu_seqlens is not None:
            raise ValueError("use_cache=True is not supported together with cu_seqlens packing.")

        hidden_states = inputs_embeds
        query_attention_mask = None
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
            query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]

        packed_metadata = None
        if position_ids is None:
            if cu_seqlens is not None:
                if attention_mask is None:
                    raise ValueError("attention_mask must be provided with cu_seqlens packing.")
                position_ids = self.build_packed_position_ids(
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                    num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
                    sequence_length=hidden_states.shape[1],
                )
            elif attention_mask is not None:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.masked_fill(~attention_mask, 0)
                position_ids = position_ids[:, -hidden_states.shape[1] :]
            else:
                past_length = 0
                if past_key_values is not None and len(past_key_values) > 0:
                    first_layer_past = past_key_values[0]
                    if _is_static_kv_cache_layer(first_layer_past):
                        past_length = int(first_layer_past.get("length", 0))
                    else:
                        past_length = first_layer_past[0].shape[1]
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
                position_ids = position_ids + past_length
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        if cu_seqlens is not None and self.attn_implementation == "flash_attention_2":
            packed_metadata = self.build_packed_metadata(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
            )

        if self.position_embedding_type == "absolute":
            hidden_states = hidden_states + self.wpe(position_ids)
        hidden_states = self.drop(hidden_states)
        if query_attention_mask is not None:
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("use_cache=True is not supported when gradient checkpointing is enabled during training.")

                def custom_forward(*inputs):
                    output, _ = block(
                        inputs[0],
                        attention_mask=inputs[1],
                        position_ids=inputs[2],
                        packed_metadata=packed_metadata,
                        layer_past=None,
                        use_cache=False,
                    )
                    return output

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    packed_metadata=packed_metadata,
                    layer_past=None if past_key_values is None else past_key_values[layer_index],
                    use_cache=use_cache,
                )
            if query_attention_mask is not None:
                hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        if query_attention_mask is not None:
            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (hidden_states, tuple(presents) if presents is not None else None, all_hidden_states, None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(presents) if presents is not None else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )
