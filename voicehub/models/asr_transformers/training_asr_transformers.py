"""Training adapter for VoiceHub's native generic ASR dispatcher."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import UpstreamNativeTrainingAdapter

_SUPPORTED_FAMILIES = frozenset({
    "ctc",
    "speech-seq2seq",
})


class TransformersASRTrainingAdapter(UpstreamNativeTrainingAdapter):
    """Preserve the selected VoiceHub graph's differentiable native loss."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-family-safetensors-and-processor")

    @property
    def native_family(self) -> str:
        resolved = getattr(self.model, "architecture_family", None)
        if resolved in _SUPPORTED_FAMILIES:
            return resolved
        configured = getattr(
            getattr(self.model, "config", None),
            "architecture_family",
            "auto",
        )
        return str(configured)

    @property
    def objective_name(self) -> str:
        family = self.native_family
        names = {
            "ctc": "CTC",
            "speech-seq2seq": "speech sequence-to-sequence",
        }
        return names.get(family, "VoiceHub native ASR")

    def setup(self):
        super().setup()
        family = self.native_family
        if family not in _SUPPORTED_FAMILIES:
            choices = ", ".join(sorted(_SUPPORTED_FAMILIES))
            raise ValueError(
                "The generic ASR checkpoint did not resolve to a "
                f"trainable family. Expected one of: {choices}; received "
                f"{family!r}.")
        return self

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration["resolved_architecture_family"] = self.native_family
        configuration["resolved_native_model_type"] = getattr(
            self.model,
            "native_model_type",
            None,
        )
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest["native_architecture_family"] = self.native_family
        manifest["native_model_type"] = getattr(
            self.model,
            "native_model_type",
            None,
        )
        manifest["processor_runtime"] = "voicehub-native"
        return manifest

    def prepare_training_inputs(self, inputs, context):
        """Keep only tensors accepted by the selected native graph."""
        import torch

        prepared = dict(super().prepare_training_inputs(inputs, context))
        native_model_type = getattr(self.model, "native_model_type", None)
        if native_model_type in {"hubert", "wav2vec2", "wavlm"}:
            input_values = prepared.get("input_values")
            if (isinstance(input_values, torch.Tensor) and input_values.ndim == 1):
                prepared["input_values"] = input_values.unsqueeze(0)
                for name in ("attention_mask", "labels"):
                    value = prepared.get(name)
                    if isinstance(value, torch.Tensor) and value.ndim == 1:
                        prepared[name] = value.unsqueeze(0)
            accepted = {
                "attention_mask",
                "generator",
                "input_values",
                "labels",
                "mask_time_indices",
                "output_attentions",
                "output_hidden_states",
                "past_key_values",
                "use_cache",
            }
        elif native_model_type == "whisper":
            input_features = prepared.get("input_features")
            if (isinstance(input_features, torch.Tensor) and input_features.ndim == 2):
                prepared["input_features"] = input_features.unsqueeze(0)
                for name in (
                        "attention_mask",
                        "decoder_attention_mask",
                        "decoder_input_ids",
                        "labels",
                ):
                    value = prepared.get(name)
                    if isinstance(value, torch.Tensor) and value.ndim == 1:
                        prepared[name] = value.unsqueeze(0)
            accepted = {
                "attention_mask",
                "decoder_attention_mask",
                "decoder_input_ids",
                "input_features",
                "labels",
                "output_attentions",
                "output_hidden_states",
                "past_key_values",
                "use_cache",
            }
        elif native_model_type == "moonshine":
            input_values = prepared.get("input_values")
            if (isinstance(input_values, torch.Tensor) and input_values.ndim == 1):
                prepared["input_values"] = input_values.unsqueeze(0)
                for name in (
                        "attention_mask",
                        "decoder_attention_mask",
                        "decoder_input_ids",
                        "labels",
                ):
                    value = prepared.get(name)
                    if isinstance(value, torch.Tensor) and value.ndim == 1:
                        prepared[name] = value.unsqueeze(0)
            accepted = {
                "attention_mask",
                "decoder_attention_mask",
                "decoder_input_ids",
                "input_values",
                "labels",
                "output_attentions",
                "output_hidden_states",
                "past_key_values",
                "use_cache",
            }
        elif native_model_type is not None:
            raise ValueError("The generic ASR runtime has not resolved a verified native "
                             "model type.")
        else:
            raise ValueError(
                "The generic ASR runtime must resolve a verified native "
                "model type before preparing a training batch.")
        return {name: value for name, value in prepared.items() if name in accepted}

    def execute_training_phase(self, context):
        output = super().execute_training_phase(context)
        output.metadata["native_architecture_family"] = self.native_family
        output.metadata["native_model_type"] = getattr(
            self.model,
            "native_model_type",
            None,
        )
        output.metadata["native_objective_required"] = True
        return output

    def save_pretrained(self, save_directory) -> None:
        """Export one self-contained native family artifact."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        if getattr(self.model, "native_model_type", None) is None:
            raise RuntimeError(
                "The generic ASR runtime has not resolved a native model "
                "type, so it cannot export a checkpoint.")
        self.model.export_native_pretrained(destination)


__all__ = ["TransformersASRTrainingAdapter"]
