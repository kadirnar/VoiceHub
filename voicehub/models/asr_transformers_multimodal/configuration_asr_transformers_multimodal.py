"""Configuration for chat-template Transformers ASR checkpoints."""

from __future__ import annotations

from voicehub.models.asr_transformers.configuration_asr_transformers import TransformersASRConfig


class MultimodalTransformersASRConfig(TransformersASRConfig):
    """Configure an ASR checkpoint driven by a multimodal chat template.

    These checkpoints still expose a native sequence-to-sequence loss,
    but their processor owns prompt construction and label masking.
    Keeping them separate from the conventional ASR pipeline prevents
    the tokenizer and feature extractor from being split apart.
    """

    model_type = "asr_transformers_multimodal"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        *,
        architecture_family: str = "speech-seq2seq",
        training_language: str | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        normalized_family = str(architecture_family).strip().lower()
        if normalized_family != self.architecture_family:
            raise ValueError(
                "Multimodal Transformers ASR checkpoints require "
                "`architecture_family='speech-seq2seq'`.")
        if training_language is not None:
            if not isinstance(training_language, str) or not training_language.strip():
                raise ValueError("`training_language` must be a non-empty string or None.")
            training_language = training_language.strip()
        self.training_language = training_language
        super().__init__(
            architecture_family=self.architecture_family,
            sample_rate=sample_rate,
            **kwargs,
        )


class Qwen3ASRConfig(MultimodalTransformersASRConfig):
    """Configuration for native Hugging Face Qwen3-ASR checkpoints."""

    model_type = "asr_qwen3"

    def __init__(
        self,
        *,
        training_language: str | None = "English",
        sample_rate: int = 16_000,
        **kwargs,
    ):
        super().__init__(
            training_language=training_language,
            sample_rate=sample_rate,
            **kwargs,
        )


class VibeVoiceASRConfig(MultimodalTransformersASRConfig):
    """Configuration for native Hugging Face VibeVoice-ASR checkpoints."""

    model_type = "asr_vibevoice"

    def __init__(
        self,
        *,
        training_language: str | None = None,
        sample_rate: int = 24_000,
        **kwargs,
    ):
        if training_language is not None:
            raise ValueError(
                "VibeVoice-ASR does not expose a language-conditioning "
                "control. Omit `training_language` and let the checkpoint "
                "infer language from the audio.")
        super().__init__(
            training_language=None,
            sample_rate=sample_rate,
            **kwargs,
        )


__all__ = [
    "MultimodalTransformersASRConfig",
    "Qwen3ASRConfig",
    "VibeVoiceASRConfig",
]
