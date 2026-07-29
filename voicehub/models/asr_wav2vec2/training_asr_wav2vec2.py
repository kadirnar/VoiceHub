"""Fine-tuning adapter for VoiceHub's native Wav2Vec2 CTC runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CTCTrainingAdapter


class NativeWav2Vec2TrainingAdapter(CTCTrainingAdapter):
    """Train and export Wav2Vec2 without an upstream model runtime."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-wav2vec2-ctc-safetensors-and-processor")
    runtime_name = "Wav2Vec2"
    checkpoint_format = "native-wav2vec2-ctc-v1"
    native_architecture_family = "wav2vec2"

    def setup(self) -> NativeWav2Vec2TrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "ctc":
            raise ValueError(f"Native {self.runtime_name} fine-tuning requires the CTC "
                             "runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                f"Native {self.runtime_name} fine-tuning must target the "
                "wrapper's exact `model` graph.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context,
    ) -> Mapping[str, Any]:
        import torch

        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
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
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": self.checkpoint_format,
            "objective": "connectionist-temporal-classification",
            "sample_rate": 16_000,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": self.checkpoint_format,
            "native_architecture_family": self.native_architecture_family,
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def execute_training_phase(self, context):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": self.native_architecture_family,
            "native_objective": "ctc",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export an inference-ready native Wav2Vec2 directory."""
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["NativeWav2Vec2TrainingAdapter"]
