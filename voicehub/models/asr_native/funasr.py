"""FunASR inference provider."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import (
    materialized_audio_file_for_provider,
    normalize_asr_result,
    reject_unsupported_options,
    require_supported_kwargs,
    resolve_cpu_cuda_device,
)
from voicehub.models.asr_native.configuration import FunASRConfig


class FunASRForSpeechRecognition(PreTrainedASRModel):
    """FunASR provider for Paraformer, SenseVoice, and compatible models."""

    config_class = FunASRConfig
    default_model_name_or_path = "iic/SenseVoiceSmall"
    training_support = "upstream-custom"

    def __init__(
        self,
        config: FunASRConfig | str | Path | None = None,
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
        return resolve_cpu_cuda_device(device, provider="FunASR")

    def _load_pretrained_model(self) -> None:
        funasr = import_optional(
            "funasr",
            model_type=self.config.model_type,
            install_extra="asr-vad",
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        auto_model = getattr(funasr, "AutoModel", None)
        if not callable(auto_model):
            raise RuntimeError("The installed FunASR package does not expose AutoModel().")
        options = {
            "model": source,
            "device": self.device,
            "vad_model": self.config.vad_model,
            "punc_model": self.config.punc_model,
            "spk_model": self.config.spk_model,
            **self.config.model_kwargs,
        }
        self.model = auto_model(**{key: value for key, value in options.items() if value is not None})
        if self.model is None or not callable(getattr(self.model, "generate", None)):
            raise RuntimeError(f"FunASR could not load an ASR runtime from {source!r}.")

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
            "FunASR",
            task=task,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
        )
        with materialized_audio_file_for_provider(
                audio,
                sampling_rate=sampling_rate,
                target_sampling_rate=self.sample_rate,
        ) as (audio_path, materialized):
            required = list(self.config.generate_kwargs)
            for requested, option_name in (
                (hotwords is not None, "hotword"),
                (language is not None, "language"),
            ):
                if requested:
                    required.append(option_name)
            values = dict(self.config.generate_kwargs)
            values.update({
                "input": audio_path,
                "hotword": hotwords,
                "language": language,
            })
            options = require_supported_kwargs(
                self.model.generate,
                values,
                provider="FunASR",
                required=("input", *required),
            )
            results = self.model.generate(**options)
        result = results
        if isinstance(result, (tuple, list)):
            if len(result) > 1:
                raise ValueError("FunASR returned multiple utterances for one audio input.")
            result = result[0] if result else {}
        if not isinstance(result, Mapping):
            return ASROutput(
                text="" if result is None else str(result),
                language=language,
                duration=materialized.duration,
                metadata={"backend": "funasr"},
            )
        timestamps = result.get("timestamp", ()) if return_timestamps else ()
        segments = []
        for timestamp in timestamps:
            if isinstance(timestamp, Mapping):
                start = timestamp.get("start")
                end = timestamp.get("end", timestamp.get("stop"))
                text = timestamp.get("text", timestamp.get("word", ""))
            elif (isinstance(timestamp, Sequence) and not isinstance(timestamp, (str, bytes)) and
                  len(timestamp) >= 2):
                start = timestamp[0]
                end = timestamp[1]
                text = timestamp[2] if len(timestamp) > 2 else ""
            else:
                continue
            if start is not None and end is not None:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                })
        normalized = {
            "text": result.get("text", ""),
            "segments": segments,
            "language": result.get("language", language),
        }
        output = normalize_asr_result(
            normalized,
            backend="funasr",
            duration=materialized.duration,
            timestamp_scale=0.001,
            language=language,
        )
        output.metadata["raw_keys"] = tuple(sorted(result))
        return output

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "FunASR fine-tuning uses its upstream task/configuration runner. "
            "Use the FunASR native training backend; exported ONNX artifacts "
            "remain inference-only.")
