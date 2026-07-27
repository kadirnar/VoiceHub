"""WhisperX transcription and optional alignment provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import (
    normalize_asr_result,
    preferred_keyword,
    reject_unsupported_options,
    require_supported_kwargs,
    resolve_cpu_cuda_device,
    supported_kwargs,
    validate_ctranslate2_precision,
)
from voicehub.models.asr_native.configuration import WhisperXConfig


class WhisperXForSpeechRecognition(PreTrainedASRModel):
    """WhisperX ASR with word alignment normalized to VoiceHub outputs."""

    config_class = WhisperXConfig
    default_model_name_or_path = "small"

    def __init__(
        self,
        config: WhisperXConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs,
    ):
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)
        self._token = token
        self._whisperx = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(
            device,
            provider="WhisperX/faster-whisper",
            allow_cuda_index=False,
        )

    def _load_pretrained_model(self) -> None:
        validate_ctranslate2_precision(
            self.config.compute_type,
            device=self.device,
            provider="WhisperX",
        )
        self._whisperx = import_optional(
            "whisperx",
            model_type=self.config.model_type,
            install_extra="asr-vad",
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        loader = getattr(self._whisperx, "load_model", None)
        if not callable(loader):
            raise RuntimeError("The installed WhisperX package does not expose load_model().")
        token_keyword = preferred_keyword(
            loader,
            ("use_auth_token", "token"),
            fallback="use_auth_token",
        )
        options = require_supported_kwargs(
            loader,
            {
                "compute_type": self.config.compute_type,
                token_keyword: self._token,
                **self.config.model_kwargs,
            },
            provider="WhisperX",
            required=(
                *tuple(self.config.model_kwargs),
                *((token_keyword, ) if self._token is not None else ()),
            ),
        )
        self.model = loader(
            source,
            self.device,
            **options,
        )
        if self.model is None or not callable(getattr(self.model, "transcribe", None)):
            raise RuntimeError(f"WhisperX could not load the ASR runtime from {source!r}.")

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
            "WhisperX",
            stride_length_s=stride_length_s,
            num_beams=num_beams,
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
            (chunk_length_s is not None, "chunk_size"),
            (batch_size is not None, "batch_size"),
        ):
            if requested:
                required.append(option_name)
        options = require_supported_kwargs(
            self.model.transcribe,
            {
                "batch_size": batch_size or 16,
                "language": language,
                "task": task,
                "chunk_size": chunk_length_s,
            },
            provider="WhisperX",
            required=tuple(required),
        )
        result = self.model.transcribe(materialized.waveform, **options)
        if not isinstance(result, Mapping):
            raise TypeError("WhisperX transcribe() must return a mapping.")
        result = dict(result)
        should_align = self.config.align_output or return_timestamps == "word"
        if should_align and result.get("segments"):
            language_code = result.get("language", language)
            if not language_code:
                raise ValueError("WhisperX alignment requires a detected or requested language.")
            align_loader = getattr(self._whisperx, "load_align_model", None)
            align = getattr(self._whisperx, "align", None)
            if not callable(align_loader) or not callable(align):
                raise RuntimeError(
                    "The installed WhisperX package does not expose its "
                    "alignment runtime.")
            align_options = require_supported_kwargs(
                align_loader,
                {
                    "language_code": language_code,
                    "device": self.device,
                },
                provider="WhisperX",
                required=("language_code", "device"),
            )
            align_model, metadata = align_loader(**align_options)
            alignment_options = supported_kwargs(
                align,
                {"return_char_alignments": False},
            )
            aligned = align(
                result["segments"],
                align_model,
                metadata,
                materialized.waveform,
                self.device,
                **alignment_options,
            )
            if not isinstance(aligned, Mapping):
                raise TypeError("WhisperX align() must return a mapping.")
            result = {
                **result,
                **aligned,
                "text": aligned.get("text", result.get("text", "")),
                "language": aligned.get("language", language_code),
            }
        if not return_timestamps and not self.config.align_output:
            result = {**result, "segments": ()}
        output = normalize_asr_result(
            result,
            backend="whisperx",
            duration=materialized.duration,
            language=language,
        )
        output.metadata["aligned"] = bool(should_align)
        return output

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "WhisperX is an inference/alignment pipeline. Fine-tune its "
            "Transformers Whisper checkpoint with `asr_transformers`, then "
            "load the exported model through WhisperX.")
