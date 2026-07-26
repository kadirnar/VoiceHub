# coding=utf-8
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from safetensors.torch import load_file
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast

from .gpt2_decoder import PackedSequenceMetadata, MossTTSNanoGPT2Model

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


class MossQwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MossQwen3RotaryEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None:
            rope_scaling = getattr(config, "rope_scaling", None)
            if isinstance(rope_scaling, dict):
                rope_theta = rope_scaling.get("rope_theta")
        if rope_theta is None:
            rope_theta = 1000000.0
        rope_theta = float(rope_theta)
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._compute_inv_freq(device=hidden_states.device)
        freqs = torch.einsum(
            "bs,d->bsd",
            position_ids.to(device=hidden_states.device, dtype=inv_freq.dtype),
            inv_freq,
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=hidden_states.dtype), emb.sin().to(dtype=hidden_states.dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half = hidden_states[..., : hidden_states.shape[-1] // 2]
    second_half = hidden_states[..., hidden_states.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    query = (query * cos) + (rotate_half(query) * sin)
    key = (key * cos) + (rotate_half(key) * sin)
    return query, key


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seq_len, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq_len, num_key_value_heads * n_rep, head_dim)


class MossQwen3MLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MossQwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = float(getattr(config, "attention_dropout", 0.0))
        self.attn_implementation = str(getattr(config, "_attn_implementation", "eager"))

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bool(config.attention_bias))
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=bool(config.attention_bias),
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bool(config.attention_bias))
        self.q_norm = MossQwen3RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = MossQwen3RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))

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
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        mask = self._causal_attention_mask(
            attention_mask=attention_mask,
            query_length=query.shape[-2],
            key_length=key.shape[-2],
            device=query.device,
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if self.training and self.attention_dropout > 0:
            probs = torch.dropout(probs, self.attention_dropout, train=True)
        output = torch.matmul(probs, value)
        return output.transpose(1, 2).contiguous()

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
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
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=mask is None,
            scale=self.scaling,
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
            raise ValueError(f"flash_attention_2 requires fp16/bf16 tensors, got dtype={query.dtype}.")

        dropout_p = self.attention_dropout if self.training else 0.0
        if packed_metadata is not None:
            if packed_metadata.indices is not None:
                query = query.reshape(-1, self.num_heads, self.head_dim).index_select(0, packed_metadata.indices)
                key = key.reshape(-1, self.num_key_value_heads, self.head_dim).index_select(0, packed_metadata.indices)
                value = value.reshape(-1, self.num_key_value_heads, self.head_dim).index_select(0, packed_metadata.indices)
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
            return flash_attn_func(query, key, value, dropout_p=dropout_p, causal=True)

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
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_norm(
            self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim)
        )
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        )
        value_states = self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat([past_key.to(device=key_states.device, dtype=key_states.dtype), key_states], dim=1)
            value_states = torch.cat(
                [past_value.to(device=value_states.device, dtype=value_states.dtype), value_states],
                dim=1,
            )

        present = (key_states, value_states) if use_cache else None
        if self.attn_implementation == "flash_attention_2":
            attn_output = self._flash_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                packed_metadata=packed_metadata,
            )
        elif self.attn_implementation == "sdpa":
            attn_output = self._sdpa_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
            )
        else:
            attn_output = self._eager_attention(
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
            )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), present


class MossQwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = MossQwen3Attention(config=config, layer_idx=layer_idx)
        self.mlp = MossQwen3MLP(config)
        self.input_layernorm = MossQwen3RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.post_attention_layernorm = MossQwen3RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_metadata: Optional[PackedSequenceMetadata] = None,
        layer_past: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            packed_metadata=packed_metadata,
            layer_past=layer_past,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


class MossQwen3Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.attn_implementation = str(getattr(config, "_attn_implementation", "eager"))
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = int(config.vocab_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MossQwen3DecoderLayer(config, layer_idx=index) for index in range(config.num_hidden_layers)]
        )
        self.norm = MossQwen3RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.rotary_emb = MossQwen3RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_use_reentrant = bool(
            getattr(config, "gradient_checkpointing_use_reentrant", False)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def load_qwen3_pretrained_weights(self, pretrained_path: str) -> None:
        model_dir = Path(pretrained_path)
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing Qwen3 safetensors index: {index_path}")
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map", {})
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard in weight_map.items():
            if not key.startswith("model."):
                continue
            shard_to_keys.setdefault(str(shard), []).append(key)

        state_dict = self.state_dict()
        loaded_state = {}
        for shard, keys in sorted(shard_to_keys.items()):
            shard_tensors = load_file(str(model_dir / shard), device="cpu")
            for key in keys:
                target_key = key[len("model.") :]
                if target_key not in state_dict:
                    continue
                tensor = shard_tensors[key]
                if tuple(tensor.shape) != tuple(state_dict[target_key].shape):
                    raise ValueError(
                        f"Shape mismatch while loading Qwen3 weight {key}: "
                        f"checkpoint={tuple(tensor.shape)} model={tuple(state_dict[target_key].shape)}"
                    )
                loaded_state[target_key] = tensor

        missing, unexpected = self.load_state_dict(loaded_state, strict=False)
        unexpected = [key for key in unexpected if key]
        if unexpected:
            raise RuntimeError(f"Unexpected Qwen3 pretrained keys after load: {unexpected[:10]}")
        missing = [key for key in missing if key not in loaded_state]
        if missing:
            raise RuntimeError(f"Missing Qwen3 pretrained keys after load: {missing[:10]}")

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
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_states.device)
        query_attention_mask = attention_mask[:, -hidden_states.shape[1] :]

        packed_metadata = None
        if position_ids is None:
            if cu_seqlens is not None:
                position_ids = MossTTSNanoGPT2Model.build_packed_position_ids(
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
                    past_length = past_key_values[0][0].shape[1]
                position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)
                position_ids = position_ids + past_length
                position_ids = position_ids.unsqueeze(0).expand(hidden_states.shape[0], -1)

        if cu_seqlens is not None and self.attn_implementation == "flash_attention_2":
            packed_metadata = MossTTSNanoGPT2Model.build_packed_metadata(
                hidden_states=hidden_states,
                cu_seqlens=cu_seqlens.to(device=hidden_states.device),
                num_sequences=num_sequences.to(device=hidden_states.device) if num_sequences is not None else None,
            )

        hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        presents = [] if use_cache else None
        for layer_index, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError("use_cache=True is not supported when gradient checkpointing is enabled during training.")

                def custom_forward(*inputs):
                    output, _ = decoder_layer(
                        hidden_states=inputs[0],
                        attention_mask=inputs[1],
                        packed_metadata=packed_metadata,
                        layer_past=None,
                        use_cache=False,
                        position_embeddings=position_embeddings,
                    )
                    return output

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    use_reentrant=self.gradient_checkpointing_use_reentrant,
                )
                present = None
            else:
                hidden_states, present = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    packed_metadata=packed_metadata,
                    layer_past=None if past_key_values is None else past_key_values[layer_index],
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

            hidden_states = hidden_states * query_attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            if presents is not None:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
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
