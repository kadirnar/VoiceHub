# coding=utf-8
"""Configuration for the MOSS-TTS-Local-Transformer-v1.5 release."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


SUPPORTED_ATTENTION_IMPLEMENTATIONS = {"flash_attention_2", "sdpa", "eager"}


def _normalize_attention_implementation(value: Optional[str], default: str = "flash_attention_2") -> str:
    normalized = str(value or default).strip().lower()
    if normalized in {"flash", "flash_attn", "flash-attn", "flash_attention"}:
        normalized = "flash_attention_2"
    if normalized not in SUPPORTED_ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            "attn_implementation must be one of "
            f"{sorted(SUPPORTED_ATTENTION_IMPLEMENTATIONS)}, got {value!r}."
        )
    return normalized


class MossTTSLocalConfig(PretrainedConfig):
    model_type = "moss_tts_local"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        qwen3_config: Optional[Union[Qwen3Config, Dict[str, Any]]] = None,
        gpt2_config: Optional[Union[GPT2Config, Dict[str, Any]]] = None,
        language_config: Optional[Union[Qwen3Config, Dict[str, Any]]] = None,
        n_vq: int = 12,
        audio_vocab_size: int = 1024,
        audio_codebook_sizes: Optional[list[int]] = None,
        audio_pad_token_id: int = 1024,
        audio_pad_code: Optional[int] = None,
        pad_token_id: int = 151643,
        im_start_token_id: int = 151644,
        im_end_token_id: int = 151645,
        audio_start_token_id: int = 151669,
        audio_end_token_id: int = 151670,
        audio_user_slot_token_id: int = 151654,
        audio_assistant_slot_token_id: int = 151656,
        audio_assistant_gen_slot_token_id: Optional[int] = None,
        sampling_rate: int = 48000,
        audio_tokenizer_name_or_path: Optional[str] = None,
        attn_implementation: str = "flash_attention_2",
        local_transformer_attn_implementation: Optional[str] = None,
        local_text_head_mode: str = "binary",
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        if qwen3_config is None and language_config is not None:
            qwen3_config = language_config
        if isinstance(qwen3_config, dict):
            self.qwen3_config = Qwen3Config(**qwen3_config)
        elif qwen3_config is None:
            self.qwen3_config = Qwen3Config()
        else:
            self.qwen3_config = qwen3_config

        if isinstance(gpt2_config, dict):
            self.gpt2_config = GPT2Config(**gpt2_config)
        elif gpt2_config is None:
            self.gpt2_config = GPT2Config(
                vocab_size=int(self.qwen3_config.vocab_size),
                n_embd=int(self.qwen3_config.hidden_size),
                n_layer=1,
                n_head=max(1, int(self.qwen3_config.hidden_size) // 80),
                n_positions=int(n_vq) + 1,
                n_ctx=int(n_vq) + 1,
                activation_function="silu",
                layer_norm_epsilon=1e-6,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        else:
            self.gpt2_config = gpt2_config

        self.n_vq = int(n_vq)
        if self.n_vq <= 0:
            raise ValueError("n_vq must be positive.")
        if audio_codebook_sizes is None:
            self.audio_codebook_sizes = [int(audio_vocab_size)] * self.n_vq
        else:
            self.audio_codebook_sizes = [int(size) for size in audio_codebook_sizes]
        if len(self.audio_codebook_sizes) != self.n_vq:
            raise ValueError(
                f"audio_codebook_sizes must have length n_vq={self.n_vq}, "
                f"got {len(self.audio_codebook_sizes)}."
            )
        if any(size <= 0 for size in self.audio_codebook_sizes):
            raise ValueError("audio_codebook_sizes must contain positive integers.")
        self.audio_vocab_size = int(max(int(audio_vocab_size), max(self.audio_codebook_sizes)))
        self.audio_pad_token_id = int(audio_pad_code if audio_pad_code is not None else audio_pad_token_id)
        self.audio_pad_code = self.audio_pad_token_id
        if self.audio_pad_token_id < self.audio_vocab_size:
            raise ValueError("audio_pad_token_id/audio_pad_code must be outside the audio vocab.")

        self.pad_token_id = int(pad_token_id)
        self.im_start_token_id = int(im_start_token_id)
        self.im_end_token_id = int(im_end_token_id)
        self.audio_start_token_id = int(audio_start_token_id)
        self.audio_end_token_id = int(audio_end_token_id)
        self.audio_user_slot_token_id = int(audio_user_slot_token_id)
        self.audio_assistant_slot_token_id = int(
            audio_assistant_slot_token_id
            if audio_assistant_gen_slot_token_id is None
            else audio_assistant_gen_slot_token_id
        )
        self.audio_assistant_gen_slot_token_id = self.audio_assistant_slot_token_id

        self.sampling_rate = int(sampling_rate)
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.attn_implementation = _normalize_attention_implementation(attn_implementation)
        self.local_transformer_attn_implementation = _normalize_attention_implementation(
            local_transformer_attn_implementation,
            default=self.attn_implementation,
        )
        self.initializer_range = float(initializer_range)

        self.hidden_size = int(self.qwen3_config.hidden_size)
        self.vocab_size = int(self.qwen3_config.vocab_size)
        self.local_hidden_size = int(self.gpt2_config.hidden_size)
        if self.local_hidden_size != self.hidden_size:
            raise ValueError(
                "This MOSS-TTS-Local-Transformer-v1.5 release expects local hidden size to "
                "match Qwen3 hidden size so audio embeddings and heads are tied."
            )

        normalized_text_head_mode = str(local_text_head_mode or "full_vocab").strip().lower()
        if normalized_text_head_mode in {"full", "full-vocab", "vocab"}:
            normalized_text_head_mode = "full_vocab"
        if normalized_text_head_mode not in {"full_vocab", "binary"}:
            raise ValueError("local_text_head_mode must be 'full_vocab' or 'binary'.")
        self.local_text_head_mode = normalized_text_head_mode

        kwargs.setdefault("tie_word_embeddings", True)
        super().__init__(pad_token_id=self.pad_token_id, **kwargs)

    @property
    def language_config(self) -> Qwen3Config:
        return self.qwen3_config

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()
        output["qwen3_config"] = self.qwen3_config.to_dict()
        output["language_config"] = self.qwen3_config.to_dict()
        output["gpt2_config"] = self.gpt2_config.to_dict()
        output["audio_pad_code"] = self.audio_pad_token_id
        output["audio_assistant_gen_slot_token_id"] = self.audio_assistant_slot_token_id
        return output
