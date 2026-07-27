"""SpeechBrain ASR inference provider."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import (
    materialized_audio_file_for_provider,
    preferred_keyword,
    reject_unsupported_options,
    require_supported_kwargs,
    resolve_cpu_cuda_device,
)
from voicehub.models.asr_native.configuration import SpeechBrainASRConfig


class SpeechBrainASRForSpeechRecognition(PreTrainedASRModel):
    """SpeechBrain EncoderDecoderASR provider with upstream recipe boundary."""

    config_class = SpeechBrainASRConfig
    default_model_name_or_path = "speechbrain/asr-crdnn-rnnlm-librispeech"
    training_support = "upstream-custom"

    def __init__(
        self,
        config: SpeechBrainASRConfig | str | Path | None = None,
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
        return resolve_cpu_cuda_device(device, provider="SpeechBrain ASR")

    def _hub_loader_options(self, loader) -> dict[str, Any]:
        needs_hub_configuration = (
            self._token is not None or self.config.revision is not None or
            self.config.cache_dir is not None or self.config.local_files_only)
        if not needs_hub_configuration:
            return {}

        try:
            fetching = import_module("speechbrain.utils.fetching")
        except ModuleNotFoundError:
            fetch_config_class = None
        else:
            fetch_config_class = getattr(fetching, "FetchConfig", None)
        if fetch_config_class is not None:
            candidate = require_supported_kwargs(
                loader,
                {
                    "fetch_config":
                    fetch_config_class(
                        token=False if self._token is None else self._token,
                        revision=self.config.revision,
                        huggingface_cache_dir=self.config.cache_dir,
                        allow_network=not self.config.local_files_only,
                    ),
                },
                provider="SpeechBrain",
            )
            if candidate:
                return candidate

        if self.config.cache_dir is not None or self.config.local_files_only:
            raise RuntimeError(
                "The installed SpeechBrain version cannot enforce cache-only "
                "Hub loading. Upgrade SpeechBrain or load a local artifact.")
        token_keyword = preferred_keyword(
            loader,
            ("use_auth_token", "token"),
            fallback="use_auth_token",
        )
        requested = []
        legacy = {
            token_keyword: self._token,
            "revision": self.config.revision,
        }
        if self._token is not None:
            requested.append(token_keyword)
        if self.config.revision is not None:
            requested.append("revision")
        return require_supported_kwargs(
            loader,
            legacy,
            provider="SpeechBrain",
            required=tuple(requested),
        )

    def _load_pretrained_model(self) -> None:
        speechbrain_asr = import_optional(
            "speechbrain.inference.ASR",
            model_type=self.config.model_type,
            install_extra="asr-speechbrain",
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        asr_class = getattr(speechbrain_asr, "EncoderDecoderASR", None)
        loader = getattr(asr_class, "from_hparams", None)
        if not callable(loader):
            raise RuntimeError(
                "The installed SpeechBrain package does not expose "
                "EncoderDecoderASR.from_hparams().")
        options = dict(self.config.model_kwargs)
        options.update({
            "source": source,
            "hparams_file": self.config.hparams_file,
            "overrides": dict(self.config.overrides),
            "run_opts": {
                "device": self.device,
            },
        })
        if self.config.savedir is not None:
            options["savedir"] = self.config.savedir
        options.update(self._hub_loader_options(loader))
        required = ("overrides", ) if self.config.overrides else ()
        options = require_supported_kwargs(
            loader,
            options,
            provider="SpeechBrain",
            required=(*tuple(self.config.model_kwargs), *required),
        )
        self.model = loader(**options)
        if self.model is None:
            raise RuntimeError(f"SpeechBrain could not load the ASR runtime from {source!r}.")

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
            "SpeechBrain EncoderDecoderASR",
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
                "SpeechBrain EncoderDecoderASR does not expose normalized "
                "timestamps through `transcribe_file`.")
        with materialized_audio_file_for_provider(
                audio,
                sampling_rate=sampling_rate,
                target_sampling_rate=self.sample_rate,
        ) as (audio_path, materialized):
            text = self.model.transcribe_file(audio_path)
        if isinstance(text, (tuple, list)) and text:
            text = text[0]
        return ASROutput(
            text=str(text),
            language=language,
            duration=materialized.duration,
            metadata={"backend": "speechbrain"},
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "SpeechBrain ASR fine-tuning is orchestrated by an upstream Brain "
            "recipe and HyperPyYAML configuration. Use the SpeechBrain native "
            "training backend for exact optimizer and checkpointer semantics.")
