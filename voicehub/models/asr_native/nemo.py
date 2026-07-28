"""NVIDIA NeMo ASR provider for Canary, Parakeet, and related families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import (
    materialized_audio_file_for_provider,
    normalize_asr_result,
    preferred_keyword,
    reject_unsupported_options,
    require_supported_kwargs,
    resolve_cpu_cuda_device,
    supported_kwargs,
)
from voicehub.models.asr_native.configuration import NeMoASRConfig


class NeMoASRForSpeechRecognition(PreTrainedASRModel):
    """Wrap NeMo's common ASRModel checkpoint and transcription APIs."""

    config_class = NeMoASRConfig
    default_model_name_or_path = "nvidia/parakeet-tdt-0.6b-v2"
    training_support = "upstream-custom"

    def __init__(
        self,
        config: NeMoASRConfig | str | Path | None = None,
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

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="NeMo ASR")

    def _load_pretrained_model(self) -> None:
        nemo_asr = import_optional(
            "nemo.collections.asr",
            model_type=self.config.model_type,
            install_extra=None,
        )
        model_class = getattr(nemo_asr.models, self.config.model_class, None)
        if model_class is None:
            raise ValueError(f"NeMo ASR has no model class {self.config.model_class!r}.")
        source = self.config.name_or_path or self.default_model_name_or_path
        source_path = Path(source).expanduser()
        suffix = source_path.suffix.lower() if source_path.is_file() else ""
        if suffix == ".nemo":
            loader = getattr(model_class, "restore_from", None)
            if not callable(loader):
                raise RuntimeError(f"{self.config.model_class} cannot restore .nemo checkpoints.")
            self.model = loader(
                restore_path=str(source_path),
                map_location=self.device,
                **self.config.model_kwargs,
            )
        elif suffix == ".ckpt":
            loader = getattr(model_class, "load_from_checkpoint", None)
            if not callable(loader):
                raise RuntimeError(f"{self.config.model_class} cannot load .ckpt checkpoints.")
            self.model = loader(
                checkpoint_path=str(source_path),
                map_location=self.device,
                **self.config.model_kwargs,
            )
        else:
            loader = getattr(model_class, "from_pretrained", None)
            if not callable(loader):
                raise RuntimeError(f"{self.config.model_class} does not expose from_pretrained().")
            token_keyword = preferred_keyword(
                loader,
                ("token", "use_auth_token"),
                fallback="token",
            )
            options = require_supported_kwargs(
                loader,
                {
                    "model_name": source,
                    "map_location": self.device,
                    token_keyword: self._token,
                    **self.config.model_kwargs,
                },
                provider="NeMo ASR",
                required=(
                    *tuple(self.config.model_kwargs),
                    *((token_keyword, ) if self._token is not None else ()),
                ),
            )
            self.model = loader(**options)
        if self.model is None:
            raise RuntimeError(f"NeMo could not load the ASR runtime from {source!r}.")
        move = getattr(self.model, "to", None)
        if callable(move):
            move(self.device)

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
            "NeMo ASR",
            task=task,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        with materialized_audio_file_for_provider(
                audio,
                sampling_rate=sampling_rate,
                target_sampling_rate=self.sample_rate,
        ) as (audio_path, materialized):
            audio_keyword = preferred_keyword(
                self.model.transcribe,
                ("audio", "paths2audio_files"),
                fallback="audio",
            )
            required = [audio_keyword]
            if batch_size is not None:
                required.append("batch_size")
            if return_timestamps:
                required.append("timestamps")
            options = require_supported_kwargs(
                self.model.transcribe,
                {
                    audio_keyword: [audio_path],
                    "batch_size": batch_size or 1,
                    "return_hypotheses": True,
                    "timestamps": bool(return_timestamps),
                },
                provider="NeMo ASR",
                required=tuple(required),
            )
            results = self.model.transcribe(**options)
        hypothesis = results
        if isinstance(hypothesis, tuple):
            hypothesis = hypothesis[0] if hypothesis else ""
        if isinstance(hypothesis, list):
            hypothesis = hypothesis[0] if hypothesis else ""
        if isinstance(hypothesis, str):
            return ASROutput(
                text=hypothesis,
                language=language,
                duration=materialized.duration,
                metadata={"backend": "nemo"},
            )
        hypothesis_get = (
            hypothesis.get if isinstance(hypothesis, dict) else
            lambda name, default=None: getattr(hypothesis, name, default))
        timestamp = hypothesis_get("timestamp", None) or {}
        segments = timestamp.get("segment", ()) if isinstance(timestamp, dict) else ()
        result = {
            "text": hypothesis_get("text", ""),
            "segments": segments if return_timestamps else (),
            "language": hypothesis_get("language", language),
        }
        output = normalize_asr_result(
            result,
            backend="nemo",
            duration=materialized.duration,
            language=language,
        )
        output.metadata["decoder"] = type(hypothesis).__name__
        return output

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "NeMo ASR fine-tuning uses its Lightning/Hydra recipe and exact "
            "NeMo checkpoint state. Use the NeMo upstream training backend "
            "instead of VoiceHub's generic optimizer loop.")

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_to"):
            self.model.save_to(str(save_directory / "model.nemo"))
