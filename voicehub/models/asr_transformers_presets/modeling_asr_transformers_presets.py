"""Architecture-specific presets built on the universal Transformers ASR
runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.models.asr_transformers.modeling_asr_transformers import TransformersASRForSpeechRecognition
from voicehub.models.asr_transformers_presets.configuration_asr_transformers_presets import (
    HubertASRConfig,
    MoonshineASRConfig,
    SeamlessM4Tv2ASRConfig,
    Wav2Vec2ASRConfig,
    WavLMASRConfig,
)


class _TransformersASRPresetForSpeechRecognition(TransformersASRForSpeechRecognition):
    """Validate a preset's native architecture before allocating weights."""

    expected_native_model_types: frozenset[str] = frozenset()

    def _load_native_model(self, transformers: Any) -> tuple[Any, str]:
        native_model_type = str(getattr(self.native_config, "model_type", ""), ).strip().lower()
        normalized_model_type = native_model_type.replace("-", "_")
        if normalized_model_type not in self.expected_native_model_types:
            expected = ", ".join(sorted(self.expected_native_model_types))
            raise ValueError(
                f"{self.__class__.__name__} requires a Transformers "
                f"checkpoint with model type {expected}; received "
                f"{native_model_type or '<missing>'!r}. Use "
                "TransformersASRForSpeechRecognition for dynamic "
                "architecture dispatch.")
        return super()._load_native_model(transformers)


class Wav2Vec2ForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """English Wav2Vec2 CTC ASR with native differentiable fine-tuning."""

    config_class = Wav2Vec2ASRConfig
    default_model_name_or_path = "facebook/wav2vec2-base-960h"
    expected_native_model_types = frozenset({"wav2vec2"})


class HubertForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """English HuBERT CTC ASR with native differentiable fine-tuning."""

    config_class = HubertASRConfig
    default_model_name_or_path = "facebook/hubert-large-ls960-ft"
    expected_native_model_types = frozenset({"hubert"})


class WavLMForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """English WavLM CTC ASR with native differentiable fine-tuning."""

    config_class = WavLMASRConfig
    default_model_name_or_path = ("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    expected_native_model_types = frozenset({"wavlm"})


class MoonshineForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """Compact English Moonshine encoder-decoder ASR."""

    config_class = MoonshineASRConfig
    default_model_name_or_path = "UsefulSensors/moonshine-tiny"
    expected_native_model_types = frozenset({
        "moonshine",
        "moonshine_streaming",
    })


class SeamlessM4Tv2ForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """Multilingual SeamlessM4T v2 speech recognition and translation."""

    config_class = SeamlessM4Tv2ASRConfig
    default_model_name_or_path = "facebook/seamless-m4t-v2-large"
    expected_native_model_types = frozenset({"seamless_m4t_v2"})

    def _pipeline_call_options(
        self,
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        # SeamlessM4T selects both recognition and translation output with
        # `tgt_lang`; its generate method does not accept Whisper's `task` or
        # `language` keywords.
        call_options = super()._pipeline_call_options(
            language=None,
            task="transcribe",
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            options=options,
        )
        target_language = language or self.config.target_language
        generation_options = dict(call_options.pop("generate_kwargs", {}) or {}, )
        self._merge_generation_option(
            generation_options,
            "tgt_lang",
            target_language,
        )
        call_options["generate_kwargs"] = generation_options
        return call_options

    def _transcribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        **kwargs,
    ):
        return super()._transcribe(
            audio,
            language=language or self.config.target_language,
            **kwargs,
        )

    def _tokenize_training_labels(
        self,
        text: str | list[str],
    ) -> Mapping[str, Any]:
        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            self.transformers_processor,
        )
        encoded = tokenizer(
            text_target=text,
            tgt_lang=self.config.target_language,
            padding=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("The SeamlessM4T tokenizer did not return a mapping.")
        return encoded


__all__ = [
    "HubertForSpeechRecognition",
    "MoonshineForSpeechRecognition",
    "SeamlessM4Tv2ForSpeechRecognition",
    "Wav2Vec2ForSpeechRecognition",
    "WavLMForSpeechRecognition",
]
