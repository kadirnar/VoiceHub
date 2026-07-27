"""Faster-whisper/CTranslate2 inference provider."""

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
    resolve_cpu_cuda_device,
    validate_ctranslate2_precision,
)
from voicehub.models.asr_native.configuration import FasterWhisperConfig


class FasterWhisperForSpeechRecognition(PreTrainedASRModel):
    """Optimized Whisper inference backed by CTranslate2."""

    config_class = FasterWhisperConfig
    default_model_name_or_path = "small"

    def __init__(
        self,
        config: FasterWhisperConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(
            device,
            provider="faster-whisper/CTranslate2",
            allow_cuda_index=False,
        )

    def _load_pretrained_model(self) -> None:
        validate_ctranslate2_precision(
            self.config.compute_type,
            device=self.device,
            provider="faster-whisper",
        )
        faster_whisper = import_optional(
            "faster_whisper",
            model_type=self.config.model_type,
            install_extra="faster-whisper",
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        model_class = getattr(faster_whisper, "WhisperModel", None)
        if not callable(model_class):
            raise RuntimeError("The installed faster-whisper package does not expose "
                               "WhisperModel().")
        options = {
            "device": self.device,
            "compute_type": self.config.compute_type,
            "cpu_threads": self.config.cpu_threads,
            "num_workers": self.config.num_workers,
            **self.config.model_kwargs,
        }
        self.model = model_class(source, **options)
        if self.model is None or not callable(getattr(self.model, "transcribe", None)):
            raise RuntimeError(f"faster-whisper could not load an ASR runtime from {source!r}.")

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
            "faster-whisper",
            stride_length_s=stride_length_s,
            batch_size=batch_size,
        )
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        if isinstance(hotwords, (tuple, list)):
            hotwords = " ".join(hotwords)
        required = []
        for requested, option_name in (
            (language is not None, "language"),
            (task != "transcribe", "task"),
            (return_timestamps == "word", "word_timestamps"),
            (chunk_length_s is not None, "chunk_length"),
            (num_beams is not None, "beam_size"),
            (max_new_tokens is not None, "max_new_tokens"),
            (hotwords is not None, "hotwords"),
        ):
            if requested:
                required.append(option_name)
        options = require_supported_kwargs(
            self.model.transcribe,
            {
                "language": language,
                "task": task,
                "word_timestamps": return_timestamps == "word",
                "chunk_length": chunk_length_s,
                "beam_size": num_beams,
                "max_new_tokens": max_new_tokens,
                "hotwords": hotwords,
            },
            provider="faster-whisper",
            required=tuple(required),
        )
        segment_iterator, info = self.model.transcribe(
            materialized.waveform,
            **options,
        )
        raw_segments = tuple(segment_iterator)
        result = {
            "text": " ".join(str(getattr(segment, "text", "")).strip() for segment in raw_segments).strip(),
            "segments": raw_segments if return_timestamps else (),
            "language": getattr(info, "language", language),
        }
        output = normalize_asr_result(
            result,
            backend="faster-whisper",
            duration=materialized.duration,
        )
        output.metadata.update({
            "language_probability": getattr(
                info,
                "language_probability",
                None,
            ),
            "duration_after_vad": getattr(
                info,
                "duration_after_vad",
                None,
            ),
            "artifact_format": "ctranslate2",
        })
        return output

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "CTranslate2 artifacts are inference-only. Fine-tune the matching "
            "Transformers Whisper checkpoint with `asr_transformers`, then "
            "convert it for faster-whisper inference.")
