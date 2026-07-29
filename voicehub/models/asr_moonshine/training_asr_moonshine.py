"""Fine-tuning adapter for VoiceHub's native Moonshine runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeMoonshineTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train and export Moonshine without an upstream model runtime."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-moonshine-safetensors-and-processor")

    def setup(self) -> NativeMoonshineTrainingAdapter:
        super().setup()
        if (getattr(self.model, "architecture_family", None) != "speech-seq2seq"):
            raise ValueError("Native Moonshine fine-tuning requires the speech-seq2seq "
                             "runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native Moonshine fine-tuning must target the wrapper's exact "
                "`model` graph.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        import torch

        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        input_values = prepared.get("input_values")
        if isinstance(input_values, torch.Tensor) and input_values.ndim == 1:
            prepared["input_values"] = input_values.unsqueeze(0)
            for name in (
                    "attention_mask",
                    "decoder_attention_mask",
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
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "native-moonshine-seq2seq-v1",
            "objective": "teacher-forced-cross-entropy",
            "sample_rate": 16_000,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "native-moonshine-seq2seq-v1",
            "native_architecture_family": "moonshine",
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "moonshine",
            "native_objective": "teacher-forced-cross-entropy",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["NativeMoonshineTrainingAdapter"]
