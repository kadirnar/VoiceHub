# coding=utf-8
"""Modeling code for the MOSS-TTS-Local-Transformer-v1.5 HuggingFace release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.utils import ModelOutput

from .configuration_moss_tts import MossTTSLocalConfig
from .gpt2_decoder import MossTTSNanoGPT2Model
from .qwen3_decoder import MossQwen3Model


@dataclass
class MossTTSLocalOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def _find_last_equal(input_ids: torch.LongTensor, value: int) -> torch.LongTensor:
    matches = input_ids.eq(int(value))
    if not bool(matches.any(dim=1).all().item()):
        raise ValueError(f"Every sample must contain token id {int(value)}.")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.long)
    masked_positions = positions.unsqueeze(0).masked_fill(~matches, -1)
    return masked_positions.max(dim=1).values


class MossTTSLocalPreTrainedModel(PreTrainedModel):
    config_class = MossTTSLocalConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MossTTSNanoGPT2Block", "MossQwen3DecoderLayer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, MossTTSNanoGPT2Model) or isinstance(module, MossQwen3Model):
            module.gradient_checkpointing = value


class MossTTSLocalModel(MossTTSLocalPreTrainedModel):
    _tied_weights_keys = None

    def __init__(self, config: MossTTSLocalConfig) -> None:
        super().__init__(config)
        self._tied_weights_keys = self._build_tied_weights_keys(config)

        config.qwen3_config.pad_token_id = config.pad_token_id
        config.qwen3_config._attn_implementation = config.attn_implementation
        local_gpt2_config = config.gpt2_config.to_dict()
        local_gpt2_config["n_layer"] = int(getattr(config, "local_transformer_layers", config.gpt2_config.n_layer))
        local_gpt2_config["n_positions"] = int(config.n_vq) + 1
        local_gpt2_config["n_ctx"] = int(config.n_vq) + 1
        local_gpt2_config = GPT2Config(**local_gpt2_config)
        local_gpt2_config.pad_token_id = config.pad_token_id
        local_gpt2_config._attn_implementation = config.local_transformer_attn_implementation

        self.transformer = MossQwen3Model(config.qwen3_config)
        self.local_transformer = MossTTSNanoGPT2Model(
            local_gpt2_config,
            attn_implementation=config.local_transformer_attn_implementation,
        )
        self.local_transformer.wte = nn.Identity()

        hidden_size = int(config.hidden_size)
        self.audio_embeddings = nn.ModuleList(
            [
                nn.Embedding(int(config.audio_codebook_sizes[index]), hidden_size)
                for index in range(config.n_vq)
            ]
        )
        self.text_lm_head = nn.Linear(hidden_size, int(config.vocab_size), bias=False)
        self.audio_lm_heads = nn.ModuleList(
            [
                nn.Linear(hidden_size, int(config.audio_codebook_sizes[index]), bias=False)
                for index in range(config.n_vq)
            ]
        )
        self.local_text_lm_head = (
            nn.Linear(hidden_size, 2, bias=False)
            if self._use_binary_local_text_head()
            else None
        )

        self.post_init()
        self.tie_weights()
        self.initialize_local_text_lm_head_from_text_lm_head()

    def can_generate(self) -> bool:
        return True

    @staticmethod
    def _build_tied_weights_keys(config: MossTTSLocalConfig) -> dict[str, str]:
        tied_weights = {"text_lm_head.weight": "transformer.embed_tokens.weight"}
        tied_weights.update(
            {
                f"audio_lm_heads.{index}.weight": f"audio_embeddings.{index}.weight"
                for index in range(config.n_vq)
            }
        )
        return tied_weights

    def tie_weights(self, *args, **kwargs) -> None:
        del args, kwargs
        self.text_lm_head.weight = self.transformer.embed_tokens.weight
        for embedding, head in zip(self.audio_embeddings, self.audio_lm_heads):
            head.weight = embedding.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.transformer.embed_tokens = value
        self.tie_weights()
        self.initialize_local_text_lm_head_from_text_lm_head()

    def get_output_embeddings(self) -> nn.Linear:
        return self.text_lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.text_lm_head = new_embeddings
        self.tie_weights()
        self.initialize_local_text_lm_head_from_text_lm_head()

    def _use_binary_local_text_head(self) -> bool:
        return str(getattr(self.config, "local_text_head_mode", "full_vocab")).strip().lower() == "binary"

    def _local_text_candidate_ids(self, device: torch.device) -> torch.LongTensor:
        return torch.tensor(
            [
                int(self.config.audio_assistant_slot_token_id),
                int(self.config.audio_end_token_id),
            ],
            dtype=torch.long,
            device=device,
        )

    def initialize_local_text_lm_head_from_text_lm_head(self) -> None:
        if not self._use_binary_local_text_head() or self.local_text_lm_head is None:
            return
        candidate_ids = self._local_text_candidate_ids(self.text_lm_head.weight.device)
        with torch.no_grad():
            source_weight = self.text_lm_head.weight.index_select(0, candidate_ids)
            if tuple(source_weight.shape) == tuple(self.local_text_lm_head.weight.shape):
                self.local_text_lm_head.weight.copy_(
                    source_weight.to(
                        device=self.local_text_lm_head.weight.device,
                        dtype=self.local_text_lm_head.weight.dtype,
                    )
                )

    def _resolve_fixed_nq(
        self,
        n_vq_for_inference: Optional[int] = None,
        nq: Optional[int] = None,
    ) -> int:
        requested = n_vq_for_inference if n_vq_for_inference is not None else nq
        config_nq = int(self.config.n_vq)
        if requested is not None and int(requested) != config_nq:
            raise ValueError(
                "This MOSS-TTS-Local-Transformer-v1.5 release is trained with a fixed RVQ depth. "
                f"Expected n_vq={config_nq}, got {int(requested)}."
            )
        return config_nq

    def _build_inputs_embeds(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        if input_ids.ndim != 3 or input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                f"Expected input_ids shape [batch, seq, {self.config.n_vq + 1}], "
                f"got {tuple(input_ids.shape)}."
            )
        text_ids = input_ids[..., 0]
        inputs_embeds = self.transformer.embed_tokens(text_ids)
        for channel_index, embedding in enumerate(self.audio_embeddings):
            channel_ids = input_ids[..., channel_index + 1]
            valid_mask = channel_ids.ne(self.config.audio_pad_token_id)
            safe_ids = channel_ids.masked_fill(~valid_mask, 0)
            audio_embeds = embedding(safe_ids) * valid_mask.unsqueeze(-1)
            inputs_embeds = inputs_embeds + audio_embeds
        return inputs_embeds

    def _global_hidden_to_local(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return hidden_states

    @staticmethod
    def _local_past_length(past_key_values: Optional[tuple[Any, ...]]) -> int:
        if past_key_values is None or len(past_key_values) == 0:
            return 0
        first_layer_past = past_key_values[0]
        if isinstance(first_layer_past, dict) and bool(first_layer_past.get("static_kv_cache", False)):
            return int(first_layer_past.get("length", 0))
        return int(first_layer_past[0].shape[1])

    def _new_static_local_past_key_values(
        self,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[str, Any], ...]:
        layers = []
        for block in self.local_transformer.h:
            attn = block.attn
            cache_shape = (
                int(batch_size),
                int(max_length),
                int(attn.num_heads),
                int(attn.head_dim),
            )
            layers.append(
                {
                    "static_kv_cache": True,
                    "key": torch.empty(cache_shape, device=device, dtype=dtype),
                    "value": torch.empty(cache_shape, device=device, dtype=dtype),
                    "length": 0,
                }
            )
        return tuple(layers)

    def _decode_local_hidden_states_with_cache(
        self,
        local_inputs_embeds: torch.FloatTensor,
        past_key_values: Optional[tuple[Any, ...]] = None,
    ) -> tuple[torch.FloatTensor, Optional[tuple[Any, ...]]]:
        if (
            past_key_values is None
            and not self.training
            and bool(getattr(self.config, "use_static_local_kv_cache", True))
        ):
            max_length = max(int(getattr(self.config, "n_vq", 0)) + 1, int(local_inputs_embeds.shape[1]))
            past_key_values = self._new_static_local_past_key_values(
                batch_size=int(local_inputs_embeds.shape[0]),
                max_length=max_length,
                device=local_inputs_embeds.device,
                dtype=local_inputs_embeds.dtype,
            )
        past_length = self._local_past_length(past_key_values)
        local_seq_len = int(local_inputs_embeds.shape[1])
        local_position_ids = torch.arange(
            past_length,
            past_length + local_seq_len,
            device=local_inputs_embeds.device,
            dtype=torch.long,
        ).unsqueeze(0)
        if int(local_inputs_embeds.shape[0]) != 1:
            local_position_ids = local_position_ids.expand(int(local_inputs_embeds.shape[0]), -1)
        local_outputs = self.local_transformer(
            input_ids=None,
            past_key_values=past_key_values,
            attention_mask=None,
            position_ids=local_position_ids,
            inputs_embeds=local_inputs_embeds,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        return local_outputs.last_hidden_state, local_outputs.past_key_values

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[tuple, MossTTSLocalOutput]:
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            inputs_embeds = self._build_inputs_embeds(input_ids)
        outputs = self.transformer(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        if not return_dict:
            return (
                outputs.last_hidden_state,
                outputs.past_key_values,
                outputs.hidden_states,
                outputs.attentions,
            )
        return MossTTSLocalOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _decode_local_last_hidden_state(
        self,
        local_inputs_embeds: torch.FloatTensor,
    ) -> torch.FloatTensor:
        local_seq_len = int(local_inputs_embeds.shape[1])
        local_position_ids = torch.arange(
            0,
            local_seq_len,
            device=local_inputs_embeds.device,
            dtype=torch.long,
        ).unsqueeze(0)
        if int(local_inputs_embeds.shape[0]) != 1:
            local_position_ids = local_position_ids.expand(int(local_inputs_embeds.shape[0]), -1)
        local_outputs = self.local_transformer(
            input_ids=None,
            attention_mask=None,
            position_ids=local_position_ids,
            inputs_embeds=local_inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        )
        return local_outputs.last_hidden_state[:, -1, :]

    def _filter_logits(
        self,
        logits: torch.FloatTensor,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.FloatTensor:
        scores = logits
        if top_k is not None and int(top_k) > 0 and int(top_k) < scores.shape[-1]:
            kth = torch.topk(scores, int(top_k), dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < kth, -torch.inf)
        if top_p is not None and 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative_probs = sorted_probs.cumsum(dim=-1)
            sorted_mask = cumulative_probs > float(top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            remove_mask = torch.zeros_like(scores, dtype=torch.bool)
            remove_mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
            scores = scores.masked_fill(remove_mask, -torch.inf)
        return scores

    def _apply_repetition_penalty(
        self,
        scores: torch.FloatTensor,
        previous_token_ids: Optional[torch.LongTensor],
        penalty: float,
    ) -> torch.FloatTensor:
        if previous_token_ids is None or float(penalty) == 1.0:
            return scores
        if previous_token_ids.ndim == 1:
            previous_token_ids = previous_token_ids.unsqueeze(0)
        updated = scores.clone()
        for batch_index in range(updated.shape[0]):
            unique_token_ids = torch.unique(previous_token_ids[batch_index])
            unique_token_ids = unique_token_ids[
                (unique_token_ids >= 0) & (unique_token_ids < updated.shape[-1])
            ]
            if unique_token_ids.numel() == 0:
                continue
            token_scores = updated[batch_index].index_select(0, unique_token_ids)
            token_scores = torch.where(
                token_scores < 0,
                token_scores * float(penalty),
                token_scores / float(penalty),
            )
            updated[batch_index].scatter_(0, unique_token_ids, token_scores)
        return updated

    def _sample_next_token(
        self,
        logits: torch.FloatTensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        previous_token_ids: Optional[torch.LongTensor] = None,
        repetition_penalty: float = 1.0,
    ) -> torch.LongTensor:
        scores = logits.float()
        scores = self._apply_repetition_penalty(scores, previous_token_ids, repetition_penalty)
        if not do_sample:
            return torch.argmax(scores, dim=-1)
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive when do_sample=True.")
        scores = scores / float(temperature)
        scores = self._filter_logits(scores, top_k=top_k, top_p=top_p)
        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _sample_next_assistant_text_token(
        self,
        local_hidden_states: torch.FloatTensor,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.LongTensor:
        if self._use_binary_local_text_head() and self.local_text_lm_head is not None:
            logits = self.local_text_lm_head(local_hidden_states)
            sampled_indices = self._sample_next_token(
                logits=logits,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            candidate_ids = self._local_text_candidate_ids(logits.device)
            return candidate_ids[sampled_indices]

        candidate_ids = self._local_text_candidate_ids(local_hidden_states.device)
        logits = self.text_lm_head(local_hidden_states).index_select(dim=-1, index=candidate_ids)
        sampled_indices = self._sample_next_token(
            logits=logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        return candidate_ids[sampled_indices]

    def _build_generation_row(
        self,
        batch_size: int,
        device: torch.device,
        audio_token_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        row = torch.full(
            (batch_size, 1, self.config.n_vq + 1),
            int(self.config.audio_pad_token_id),
            dtype=torch.long,
            device=device,
        )
        row[:, :, 0] = int(self.config.audio_assistant_slot_token_id)
        row[:, :, 1:] = audio_token_ids.unsqueeze(1)
        return row

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        max_new_frames: Optional[int] = None,
        do_sample: bool = True,
        text_temperature: float = 1.0,
        text_top_p: float = 1.0,
        text_top_k: int = 50,
        audio_temperature: Optional[float] = None,
        audio_top_p: Optional[float] = None,
        audio_top_k: Optional[int] = None,
        audio_repetition_penalty: Optional[float] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        use_kv_cache: bool = True,
        n_vq_for_inference: Optional[int] = None,
        nq: Optional[int] = None,
        **kwargs,
    ) -> list[tuple[int, torch.LongTensor]]:
        del kwargs
        self._resolve_fixed_nq(n_vq_for_inference=n_vq_for_inference, nq=nq)

        if input_ids.ndim == 2:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 3:
            raise ValueError(f"Expected input_ids with 3 dims, got {tuple(input_ids.shape)}.")
        if input_ids.shape[-1] != self.config.n_vq + 1:
            raise ValueError(
                f"Expected {self.config.n_vq + 1} channels from config.n_vq, got {input_ids.shape[-1]}."
            )
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        frame_budget = max_new_frames if max_new_frames is not None else max_new_tokens
        if frame_budget is None:
            frame_budget = 4096
        frame_budget = int(frame_budget)

        audio_temperature = float(temperature if audio_temperature is None else audio_temperature)
        audio_top_p = float(top_p if audio_top_p is None else audio_top_p)
        audio_top_k = int(top_k if audio_top_k is None else audio_top_k)
        audio_repetition_penalty = float(
            repetition_penalty if audio_repetition_penalty is None else audio_repetition_penalty
        )

        batch_size = input_ids.shape[0]
        input_ids_length = input_ids.shape[1]
        current_input_ids = input_ids
        current_attention_mask = attention_mask
        current_model_input_ids = current_input_ids
        generated_frames: list[torch.LongTensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        past_key_values = None
        local_dtype = self.local_transformer.ln_f.weight.dtype

        for _ in range(frame_budget):
            generated_audio_history = torch.stack(generated_frames, dim=1) if generated_frames else None
            global_inputs_embeds = self._build_inputs_embeds(current_model_input_ids)
            global_outputs = self.transformer(
                input_ids=None,
                past_key_values=past_key_values,
                attention_mask=current_attention_mask,
                position_ids=None,
                inputs_embeds=global_inputs_embeds,
                use_cache=use_kv_cache,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                cu_seqlens=None,
                num_sequences=None,
            )
            global_hidden_states = global_outputs.last_hidden_state[:, -1, :]
            local_global_hidden_states = self._global_hidden_to_local(global_hidden_states).to(dtype=local_dtype)

            local_prefix_hidden_states, local_prefix_past_key_values = self._decode_local_hidden_states_with_cache(
                local_global_hidden_states.unsqueeze(1)
            )
            local_hidden_states = local_prefix_hidden_states[:, -1, :]
            next_text_tokens = self._sample_next_assistant_text_token(
                local_hidden_states=local_hidden_states,
                do_sample=do_sample,
                temperature=text_temperature,
                top_k=text_top_k,
                top_p=text_top_p,
            )
            should_continue = next_text_tokens.eq(int(self.config.audio_assistant_slot_token_id)) & ~finished
            finished = finished | next_text_tokens.eq(int(self.config.audio_end_token_id))
            if not bool(should_continue.any().item()):
                break

            next_frame_tokens = []
            for channel_index in range(int(self.config.n_vq)):
                channel_logits = self.audio_lm_heads[channel_index](local_hidden_states)
                channel_token = self._sample_next_token(
                    logits=channel_logits,
                    do_sample=do_sample,
                    temperature=audio_temperature,
                    top_k=audio_top_k,
                    top_p=audio_top_p,
                    previous_token_ids=(
                        None
                        if generated_audio_history is None
                        else generated_audio_history[:, :, channel_index]
                    ),
                    repetition_penalty=audio_repetition_penalty,
                )
                next_frame_tokens.append(channel_token)
                if channel_index + 1 < int(self.config.n_vq):
                    current_local_input = self.audio_embeddings[channel_index](channel_token).to(dtype=local_dtype)
                    local_token_hidden_states, local_prefix_past_key_values = (
                        self._decode_local_hidden_states_with_cache(
                            current_local_input.unsqueeze(1),
                            past_key_values=local_prefix_past_key_values,
                        )
                    )
                    local_hidden_states = local_token_hidden_states[:, -1, :]

            next_frame = torch.stack(next_frame_tokens, dim=-1)
            next_frame = next_frame.masked_fill(
                ~should_continue.unsqueeze(-1),
                int(self.config.audio_pad_token_id),
            )
            generated_frames.append(next_frame)

            next_row = self._build_generation_row(
                batch_size=batch_size,
                device=input_ids.device,
                audio_token_ids=next_frame,
            )
            if bool((~should_continue).any().item()):
                next_row[~should_continue, 0, 0] = int(self.config.pad_token_id)
                next_row[~should_continue, 0, 1:] = int(self.config.audio_pad_token_id)

            current_input_ids = torch.cat([current_input_ids, next_row], dim=1)
            current_attention_mask = torch.cat(
                [current_attention_mask, should_continue.unsqueeze(1)],
                dim=1,
            )
            if use_kv_cache:
                current_model_input_ids = next_row
                past_key_values = global_outputs.past_key_values
            else:
                current_model_input_ids = current_input_ids

        start_indices = _find_last_equal(input_ids[..., 0], int(self.config.audio_start_token_id))
        start_lengths = input_ids_length - start_indices - 1
        outputs: list[tuple[int, torch.LongTensor]] = []
        for start_index, start_length, generation_ids in zip(
            start_indices.tolist(),
            start_lengths.tolist(),
            current_input_ids,
        ):
            outputs.append((int(start_length), generation_ids[int(start_index):].detach().cpu()))
        return outputs
