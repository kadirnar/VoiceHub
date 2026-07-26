"""ConversationTTS configuration."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig


class ConversationTTSConfig(VoiceHubConfig):
    """Configuration for the CC BY-NC ConversationTTS release."""

    model_type = "conversationtts"

    def __init__(
        self,
        *,
        checkpoint_filename: str = "ckpt1.checkpoint",
        text_tokenizer_path: str | None = None,
        audio_tokenizer_path: str | None = None,
        audio_tokenizer_repo_id: str = "kyutai/moshika-pytorch-bf16",
        audio_tokenizer_filename: str = "tokenizer-e351c8d8-checkpoint125.safetensors",
        model_args: dict | None = None,
        torch_dtype: str = "bfloat16",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.checkpoint_filename = checkpoint_filename
        self.text_tokenizer_path = text_tokenizer_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.audio_tokenizer_repo_id = audio_tokenizer_repo_id
        self.audio_tokenizer_filename = audio_tokenizer_filename
        self.model_args = model_args or {
            "backbone_flavor": "llama-1B",
            "decoder_flavor": "llama-100M",
            "text_vocab_size": 128_256,
            "audio_vocab_size": 2051,
            "audio_num_codebooks": 32,
        }
        self.torch_dtype = torch_dtype
