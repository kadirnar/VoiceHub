"""Configuration for IBM Granite Speech ASR checkpoints."""

from __future__ import annotations

from voicehub.models.asr_transformers_multimodal.configuration_asr_transformers_multimodal import (
    MultimodalTransformersASRConfig, )

_DEFAULT_TRANSCRIPTION_PROMPT = ("Please transcribe the following audio to text<|audio|>")


class GraniteSpeechASRConfig(MultimodalTransformersASRConfig):
    """Configure native Transformers Granite Speech inference and training.

    Granite Speech uses a tokenizer-rendered instruction followed by
    processor owned audio features. ``transcription_prompt`` is shared
    by inference and raw-data fine-tuning so exported checkpoints retain
    the exact task instruction used to build their supervised examples.
    """

    model_type = "asr_granite_speech"

    def __init__(
        self,
        *,
        transcription_prompt: str = _DEFAULT_TRANSCRIPTION_PROMPT,
        training_language: str | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        if not isinstance(transcription_prompt, str) or not transcription_prompt.strip():
            raise ValueError("`transcription_prompt` must be a non-empty string.")
        if "<|audio|>" not in transcription_prompt:
            raise ValueError(
                "`transcription_prompt` must contain Granite Speech's "
                "`<|audio|>` placeholder.")
        if training_language is not None:
            raise ValueError(
                "Granite Speech training is prompt-conditioned rather than "
                "language-ID conditioned. Put language guidance in "
                "`transcription_prompt` and omit `training_language`.")
        self.transcription_prompt = transcription_prompt.strip()
        super().__init__(
            training_language=None,
            sample_rate=sample_rate,
            **kwargs,
        )


__all__ = ["GraniteSpeechASRConfig"]
