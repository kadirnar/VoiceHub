"""Full-model native CTC fine-tuning adapter for MedASR."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CTCTrainingAdapter


class NativeMedASRTrainingAdapter(CTCTrainingAdapter):
    """Reproduce the official full-model MedASR fine-tuning boundary."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-medasr-ctc-safetensors-and-processor")

    def setup(self) -> NativeMedASRTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "ctc":
            raise ValueError("Native MedASR fine-tuning requires the CTC runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native MedASR fine-tuning must target the wrapper's exact "
                             "`model` graph.")
        for parameter in self.primary_model.parameters():
            parameter.requires_grad_(True)
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        accepted = {
            "attention_mask",
            "input_features",
            "labels",
            "output_hidden_states",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": ("voicehub-native-medasr-ctc-v1"),
            "full_model":
            True,
            "learning_rate":
            3e-5,
            "objective":
            "connectionist-temporal-classification",
            "optimizer":
            "adamw",
            "sample_rate":
            16_000,
            "source_recipe": ("Google-Health/medasr/notebooks/"
                              "fine_tune_with_hugging_face.ipynb"),
            "source_recipe_revision": ("ad843cb81b3e610e1868ed38f7230a70b66ed7e8"),
            "warmup_steps":
            300,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": ("voicehub-native-medasr-ctc-v1"),
            "fine_tuning_scope": "full-model",
            "native_architecture_family": "lasr-ctc",
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def execute_training_phase(self, context):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "fine_tuning_scope": "full-model",
            "native_architecture_family": "lasr-ctc",
            "native_objective": "ctc",
        })
        return output

    def save_pretrained(
        self,
        save_directory: str | Path,
    ) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        self.model.export_native_pretrained(destination)


__all__ = ["NativeMedASRTrainingAdapter"]
