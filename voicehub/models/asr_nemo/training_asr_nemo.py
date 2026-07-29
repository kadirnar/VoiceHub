"""Fine-tuning adapter for VoiceHub's native NeMo QuartzNet CTC runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import CTCTrainingAdapter


class NativeNeMoCTCTrainingAdapter(CTCTrainingAdapter):
    """Train and export QuartzNet without NeMo, Lightning, or Hydra."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-nemo-quartznet-ctc-safetensors"

    def setup(self) -> NativeNeMoCTCTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "ctc":
            raise ValueError("Native NeMo fine-tuning requires the CTC runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native NeMo fine-tuning must target the wrapper's exact "
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
        input_signal = prepared.get("input_signal")
        if isinstance(input_signal, torch.Tensor) and input_signal.ndim == 1:
            prepared["input_signal"] = input_signal.unsqueeze(0)
            for name in (
                    "input_signal_length",
                    "labels",
                    "label_lengths",
            ):
                value = prepared.get(name)
                if not isinstance(value, torch.Tensor):
                    continue
                if (name in {"input_signal_length", "label_lengths"} and value.ndim == 0):
                    prepared[name] = value.unsqueeze(0)
                elif name == "labels" and value.ndim == 1:
                    prepared[name] = value.unsqueeze(0)
        accepted = {
            "input_signal",
            "input_signal_length",
            "label_lengths",
            "labels",
            "processed_signal",
            "processed_signal_length",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "voicehub-nemo-quartznet-ctc-v1",
            "objective": "connectionist-temporal-classification",
            "sample_rate": 16_000,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-nemo-quartznet-ctc-v1",
            "native_architecture_family": "nemo-quartznet-ctc",
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "nemo-quartznet-ctc",
            "native_objective": "ctc",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["NativeNeMoCTCTrainingAdapter"]
