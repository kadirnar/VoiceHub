"""Original OpenAI Whisper inference provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import (
    normalize_asr_result,
    reject_unsupported_options,
    require_supported_kwargs,
)
from voicehub.models.asr_native.configuration import OpenAIWhisperConfig


class OpenAIWhisperForSpeechRecognition(PreTrainedASRModel):
    """Load original Whisper ``.pt`` checkpoints for inference."""

    config_class = OpenAIWhisperConfig
    default_model_name_or_path = "small"

    def __init__(
        self,
        config: OpenAIWhisperConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        whisper = import_optional(
            "whisper",
            model_type=self.config.model_type,
            install_extra="openai-whisper",
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        loader = getattr(whisper, "load_model", None)
        if not callable(loader):
            raise RuntimeError("The installed openai-whisper package does not expose load_model().")
        self.model = loader(
            source,
            device=self.device,
            **self.config.model_kwargs,
        )
        if self.model is None or not callable(getattr(self.model, "transcribe", None)):
            raise RuntimeError(f"OpenAI Whisper could not load an ASR runtime from {source!r}.")

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s=None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords=None,
    ) -> ASROutput:
        reject_unsupported_options(
            "OpenAI Whisper",
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        required = []
        for requested, option_name in (
            (language is not None, "language"),
            (task != "transcribe", "task"),
            (return_timestamps == "word", "word_timestamps"),
            (num_beams is not None, "beam_size"),
        ):
            if requested:
                required.append(option_name)
        options = require_supported_kwargs(
            self.model.transcribe,
            {
                "language": language,
                "task": task,
                "word_timestamps": return_timestamps == "word",
                "beam_size": num_beams,
            },
            provider="OpenAI Whisper",
            required=tuple(required),
        )
        result = self.model.transcribe(materialized.waveform, **options)
        if not return_timestamps and isinstance(result, dict):
            result = {**result, "segments": ()}
        return normalize_asr_result(
            result,
            backend="openai-whisper",
            duration=materialized.duration,
            language=language,
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "The original Whisper runtime is inference-oriented in VoiceHub. "
            "Use `asr_transformers` for fine-tuning and convert/export afterward.")
