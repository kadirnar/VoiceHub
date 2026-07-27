"""WeNet ASR inference provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import (
    materialized_audio_file_for_provider,
    reject_unsupported_options,
    resolve_cpu_cuda_device,
)
from voicehub.models.asr_native.configuration import WeNetASRConfig


class WeNetASRForSpeechRecognition(PreTrainedASRModel):
    """WeNet runtime provider with upstream recipe-compatible artifacts."""

    config_class = WeNetASRConfig
    default_model_name_or_path = "english"
    training_support = "upstream-custom"

    def __init__(
        self,
        config: WeNetASRConfig | str | Path | None = None,
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
        return resolve_cpu_cuda_device(device, provider="WeNet")

    def _load_pretrained_model(self) -> None:
        wenet_model = import_optional(
            "wenet.cli.model",
            model_type=self.config.model_type,
            install_extra="asr-wenet",
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        loader = getattr(wenet_model, "load_model", None)
        if not callable(loader):
            raise RuntimeError("The installed WeNet package does not expose load_model().")
        self.model = loader(
            source,
            device=self.device,
            **self.config.model_kwargs,
        )
        if self.model is None or not callable(getattr(self.model, "transcribe", None)):
            raise RuntimeError(f"WeNet could not load an ASR runtime from {source!r}.")

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
            "WeNet",
            task=task,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if return_timestamps:
            raise ValueError(
                "Timestamp semantics vary across WeNet decoders and are not "
                "available through the generic CLI model wrapper.")
        with materialized_audio_file_for_provider(
                audio,
                sampling_rate=sampling_rate,
                target_sampling_rate=self.sample_rate,
        ) as (audio_path, materialized):
            result = self.model.transcribe(audio_path)
        text = result.get("text", "") if isinstance(result, dict) else str(result)
        metadata = {"backend": "wenet"}
        if isinstance(result, dict):
            metadata["raw_keys"] = tuple(sorted(result))
            for name in ("confidence", "score"):
                if result.get(name) is not None:
                    metadata[name] = result[name]
        return ASROutput(
            text=text,
            language=language or getattr(self.config, "language", None),
            duration=materialized.duration,
            metadata=metadata,
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "WeNet fine-tuning uses its upstream distributed training recipe. "
            "Use the WeNet native training backend and retain its YAML, "
            "tokenizer, and checkpoint bundle.")
