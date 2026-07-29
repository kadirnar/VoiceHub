"""Fine-tuning adapter for VoiceHub's native Granite Speech runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter

_CHECKPOINT_FORMAT = "native-granite-speech-v1"
_NATIVE_OBJECTIVE = ("audio-conditioned-completion-only-causal-language-modeling")
_SOURCE_NOTEBOOK_REVISION = ("c622675f2059ad81d1b89387f4e7bda5110cfb87")
_SOURCE_NOTEBOOK_URL = (
    "https://github.com/ibm-granite/granite-speech-models/blob/"
    f"{_SOURCE_NOTEBOOK_REVISION}/notebooks/"
    "fine_tuning_granite_speech.ipynb")
_SOURCE_TRAINABLE_SCOPE = "projector-and-native-lora"
_FULL_MODEL_SCOPE = "full-model"
_SOURCE_LORA_TARGET_MODULES = (
    "language_model.model.layers.*.self_attn.q_proj",
    "language_model.model.layers.*.self_attn.v_proj",
)
_TRAINABLE_SCOPES = frozenset({
    _SOURCE_TRAINABLE_SCOPE,
    _FULL_MODEL_SCOPE,
})
_FORWARD_FIELDS = frozenset({
    "attention_mask",
    "audio_lengths",
    "ignore_index",
    "input_features",
    "input_features_mask",
    "input_ids",
    "label_smoothing",
    "labels",
    "output_attentions",
    "output_hidden_states",
    "position_ids",
    "use_cache",
})
_REQUIRED_FORWARD_FIELDS = frozenset({
    "attention_mask",
    "input_features",
    "input_features_mask",
    "input_ids",
    "labels",
})


class NativeGraniteSpeechTrainingAdapter(
        SpeechSeq2SeqTrainingAdapter, ):
    """Apply IBM's narrow recipe to VoiceHub's native Granite Speech graph."""

    supports_custom_recipe = True
    native_export_semantics = (
        "voicehub-native-granite-speech-safetensors-processor-"
        "tokenizer-and-generation-config")

    def __init__(
        self,
        model: Any,
        spec: Any,
        *,
        trainable_scope: str = _SOURCE_TRAINABLE_SCOPE,
        lora_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model, spec)
        self._trainable_scope = self._normalize_trainable_scope(trainable_scope, )
        self._lora_options = self._normalize_lora_options(lora_options)
        self._trainability_prepared = False
        self._trainable_parameter_names: tuple[str, ...] = ()
        self._trainable_parameter_count = 0

    @staticmethod
    def _normalize_trainable_scope(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Granite Speech trainable scope must be a non-empty string.")
        normalized = value.strip().lower().replace("_", "-")
        if normalized not in _TRAINABLE_SCOPES:
            choices = ", ".join(sorted(_TRAINABLE_SCOPES))
            raise ValueError(
                "Unsupported Granite Speech trainable scope "
                f"{value!r}. Expected one of: {choices}.")
        return normalized

    @staticmethod
    def _normalize_lora_options(value: Mapping[str, Any] | None, ) -> dict[str, Any]:
        if value is not None and not isinstance(value, Mapping):
            raise TypeError("Granite Speech LoRA options must be a mapping.")
        options = {
            "target_modules": _SOURCE_LORA_TARGET_MODULES,
        }
        if value is not None:
            options.update(value)
        targets = tuple(options["target_modules"])
        if any(not isinstance(target, str) or not target.startswith("language_model.") for target in targets):
            raise ValueError(
                "Granite Speech's source recipe permits native LoRA only "
                "inside `language_model`; the projector is trained densely.")
        options["target_modules"] = targets
        return options

    @property
    def trainable_scope(self) -> str:
        return self._trainable_scope

    def configure_trainable_scope(
        self,
        trainable_scope: str,
        *,
        lora_options: Mapping[str, Any] | None = None,
    ) -> NativeGraniteSpeechTrainingAdapter:
        """Select the source recipe or explicitly opt into full fine-tuning."""
        if self.is_ready:
            raise RuntimeError("Configure Granite Speech trainability before setup.")
        self._trainable_scope = self._normalize_trainable_scope(trainable_scope, )
        if lora_options is not None:
            self._lora_options = self._normalize_lora_options(lora_options)
        return self

    def _lora_configuration(self) -> dict[str, Any] | None:
        injection = getattr(self.model, "_lora_injection", None)
        if injection is None:
            return None
        config = injection.config
        return {
            "alpha": config.alpha,
            "dropout": config.dropout,
            "freeze_base": config.freeze_base,
            "module_names": list(injection.module_names),
            "rank": config.rank,
            "seed": config.seed,
            "target_modules": list(config.target_modules),
        }

    def _prepare_trainability(self, wrapper_model: Any) -> None:
        if self._trainability_prepared:
            return
        if self._trainable_scope == _FULL_MODEL_SCOPE:
            for parameter in wrapper_model.parameters():
                parameter.requires_grad_(True)
        else:
            enable_lora = getattr(self.model, "enable_lora", None)
            if getattr(self.model, "_lora_injection", None) is None:
                if not callable(enable_lora):
                    raise TypeError(
                        "The source-compatible Granite Speech recipe "
                        "requires the wrapper's native `enable_lora()` "
                        "implementation.")
                enable_lora(**self._lora_options)
            injection = self.model._lora_injection
            invalid_modules = tuple(
                name for name in injection.module_names if not name.startswith("language_model."))
            if invalid_modules:
                names = ", ".join(invalid_modules)
                raise ValueError(
                    "Granite Speech's source recipe received LoRA modules "
                    f"outside `language_model`: {names}.")
            trainable_names = []
            projector_names = []
            lora_names = []
            for name, parameter in wrapper_model.named_parameters():
                normalized = name.lower()
                trainable = (
                    "projector" in normalized or
                    (normalized.startswith("language_model.") and "lora" in normalized))
                parameter.requires_grad_(trainable)
                if trainable:
                    trainable_names.append(name)
                if "projector" in normalized:
                    projector_names.append(name)
                if "lora" in normalized:
                    lora_names.append(name)
            if not projector_names:
                raise ValueError("The native Granite Speech graph has no trainable "
                                 "projector parameters.")
            if not lora_names:
                raise ValueError(
                    "Native Granite Speech LoRA injection produced no "
                    "trainable adapter parameters.")
            self._trainable_parameter_names = tuple(trainable_names)
        if self._trainable_scope == _FULL_MODEL_SCOPE:
            self._trainable_parameter_names = tuple(
                name for name, parameter in wrapper_model.named_parameters() if parameter.requires_grad)
        self._trainable_parameter_count = sum(
            parameter.numel() for parameter in wrapper_model.parameters() if parameter.requires_grad)
        self._trainability_prepared = True

    def setup(self, ) -> NativeGraniteSpeechTrainingAdapter:
        super().setup()
        if (getattr(self.model, "architecture_family", None) != "speech-seq2seq"):
            raise ValueError("Native Granite Speech fine-tuning requires the "
                             "speech-seq2seq runtime.")
        wrapper_model = getattr(self.model, "model", None)
        if self.primary_model is not wrapper_model:
            raise ValueError(
                "Native Granite Speech fine-tuning must target the wrapper's "
                "exact `model` graph.")
        runtime = getattr(self.model, "runtime", None)
        if (runtime is not None and getattr(runtime, "model", None) is not wrapper_model):
            raise ValueError("The Granite Speech wrapper and runtime refer to different "
                             "model graphs.")
        self._prepare_trainability(wrapper_model)
        self._validate_loaded_training_graph()
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        if not isinstance(prepared, Mapping):
            raise TypeError("Granite Speech input preparation must return a mapping.")
        prepared = {name: value for name, value in prepared.items() if name in _FORWARD_FIELDS}
        missing = sorted(_REQUIRED_FORWARD_FIELDS - prepared.keys())
        if missing:
            raise ValueError("Granite Speech training inputs are missing: "
                             f"{', '.join(missing)}.")
        if prepared.get("use_cache") is True:
            raise ValueError("Granite Speech training does not support `use_cache=True`.")
        prepared.setdefault("use_cache", False)
        return prepared

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration(), )
        configuration.update({
            "checkpoint_format": _CHECKPOINT_FORMAT,
            "label_policy": "transcript-completion-only",
            "model_path": "model",
            "objective": _NATIVE_OBJECTIVE,
            "sample_rate": 16_000,
            "source_notebook_revision": _SOURCE_NOTEBOOK_REVISION,
            "source_notebook_url": _SOURCE_NOTEBOOK_URL,
            "source_recommended_bf16": True,
            "source_recommended_learning_rate": 3e-5,
            "source_recommended_warmup_ratio": 0.2,
            "trainable_scope": self._trainable_scope,
            "lora": self._lora_configuration(),
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format":
            _CHECKPOINT_FORMAT,
            "native_architecture_family":
            "granite-speech",
            "native_model_path":
            "model",
            "native_objective":
            _NATIVE_OBJECTIVE,
            "label_policy":
            "transcript-completion-only",
            "processor_runtime":
            "voicehub-native-granite-speech",
            "tokenizer_runtime":
            "voicehub-byte-bpe",
            "source_notebook": {
                "revision": _SOURCE_NOTEBOOK_REVISION,
                "url": _SOURCE_NOTEBOOK_URL,
            },
            "source_recommended_training": {
                "bf16": True,
                "learning_rate": 3e-5,
                "warmup_ratio": 0.2,
            },
            "trainable_parameter_count":
            (self._trainable_parameter_count if self._trainability_prepared else None),
            "trainable_tensor_count":
            (len(self._trainable_parameter_names) if self._trainability_prepared else None),
            "trainable_scope":
            self._trainable_scope,
            "export_scope": ("full-model-processor-tokenizer-generation-config"),
            "lora":
            self._lora_configuration(),
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "granite-speech",
            "native_objective": _NATIVE_OBJECTIVE,
            "label_policy": "transcript-completion-only",
            "trainable_scope": self._trainable_scope,
        })
        return output

    def save_pretrained(
        self,
        save_directory: str | Path,
    ) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        export = getattr(
            self.model,
            "export_native_pretrained",
            None,
        )
        if not callable(export):
            raise TypeError(
                "Native Granite Speech training requires a wrapper with "
                "`export_native_pretrained()`.")
        export(destination)


__all__ = ["NativeGraniteSpeechTrainingAdapter"]
