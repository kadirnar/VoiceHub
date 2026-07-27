"""Configuration for the universal Hugging Face Transformers ASR provider."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import ASRInferenceConfig


class TransformersASRConfig(VoiceHubConfig):
    """Configure a native Transformers speech-recognition checkpoint.

    The provider deliberately keeps the architecture family separate
    from the checkpoint's ``model_type``. This lets one VoiceHub model
    key load any compatible CTC, sequence-to-sequence, RNN-T, or TDT
    checkpoint without copying architecture-specific loading code.
    """

    model_type = "asr_transformers"
    supported_architecture_families = frozenset({
        "auto",
        "ctc",
        "speech-seq2seq",
        "rnnt",
        "tdt",
    })

    def __init__(
        self,
        *,
        architecture_family: str = "auto",
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        pipeline_kwargs: Mapping[str, Any] | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
                "pipeline_kwargs": pipeline_kwargs,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.architecture_family = architecture_family
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.model_kwargs = self._copy_mapping(model_kwargs, name="model_kwargs")
        self.processor_kwargs = self._copy_mapping(
            processor_kwargs,
            name="processor_kwargs",
        )
        self.pipeline_kwargs = self._copy_mapping(
            pipeline_kwargs,
            name="pipeline_kwargs",
        )
        self.validate()

    @staticmethod
    def _copy_mapping(
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping or None.")
        return dict(value)

    def validate(self) -> None:
        """Validate dependency-free provider controls."""
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        inference_values = getattr(self, "inference_config", {})
        if isinstance(inference_values, ASRInferenceConfig):
            inference_values = inference_values.to_dict()
        elif isinstance(inference_values, Mapping):
            inference_values = dict(inference_values)
        else:
            raise TypeError("`inference_config` must be a mapping.")
        for name in ASRInferenceConfig._COMMON_FIELDS:
            if not hasattr(self, name):
                continue
            value = getattr(self, name)
            inference_values[name] = value
            delattr(self, name)
        self.inference_config = ASRInferenceConfig.from_dict(inference_values).to_dict()
        if not isinstance(self.architecture_family, str):
            raise TypeError("`architecture_family` must be a string.")
        self.architecture_family = self.architecture_family.strip().lower()
        if self.architecture_family.replace("_", "-") == "audio-text-to-text":
            raise ValueError(
                "`audio-text-to-text` checkpoints require prompt/chat-template "
                "preprocessing and causal label construction, which are not "
                "compatible with the Transformers ASR provider contract. "
                "Register a dedicated provider for that model family.")
        if self.architecture_family not in self.supported_architecture_families:
            supported = ", ".join(sorted(self.supported_architecture_families))
            raise ValueError(
                "`architecture_family` must be one of: "
                f"{supported}; received {self.architecture_family!r}.")
        for option_name in ("config_name_or_path", "processor_name_or_path"):
            value = getattr(self, option_name)
            if value is None:
                continue
            if (not isinstance(value, (str, Path)) or not str(value).strip()):
                raise ValueError(f"`{option_name}` must be a non-empty path or Hub ID.")
            setattr(self, option_name, str(value))
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.revision is not None and (not isinstance(self.revision, str) or not self.revision.strip()):
            raise ValueError("`revision` must be a non-empty string or None.")
        if self.revision is not None:
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be a path-like value or None.")
            self.cache_dir = str(self.cache_dir)
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is not None and not isinstance(
                self.use_safetensors,
                bool,
        ):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")

        self.model_kwargs = self._copy_mapping(
            self.model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            self.processor_kwargs,
            name="processor_kwargs",
        )
        self.pipeline_kwargs = self._copy_mapping(
            self.pipeline_kwargs,
            name="pipeline_kwargs",
        )
        reserved_model_options = {"config", "state_dict", "trust_remote_code"}
        conflicting_model_options = reserved_model_options.intersection(self.model_kwargs)
        if conflicting_model_options:
            names = ", ".join(sorted(conflicting_model_options))
            raise ValueError(f"`model_kwargs` cannot override provider-owned option(s): {names}.")
        if "trust_remote_code" in self.processor_kwargs:
            raise ValueError("`processor_kwargs` cannot override `trust_remote_code`.")
        reserved_pipeline_options = {
            "device",
            "model",
            "processor",
            "task",
            "tokenizer",
            "feature_extractor",
        }
        conflicting_pipeline_options = reserved_pipeline_options.intersection(self.pipeline_kwargs)
        if conflicting_pipeline_options:
            names = ", ".join(sorted(conflicting_pipeline_options))
            raise ValueError("`pipeline_kwargs` cannot override provider-owned option(s): "
                             f"{names}.")

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable overrides before serializing the provider."""
        self.validate()
        return super().to_dict()


__all__ = ["TransformersASRConfig"]
