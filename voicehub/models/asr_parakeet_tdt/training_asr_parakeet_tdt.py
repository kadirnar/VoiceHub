"""Fine-tuning adapter for VoiceHub-native Parakeet TDT."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import TDTTrainingAdapter


class NativeParakeetTDTTrainingAdapter(TDTTrainingAdapter):
    """Train the complete FastConformer/TDT graph with its native objective."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-parakeet-tdt-safetensors-and-processor")
    runtime_name = "Parakeet TDT"
    checkpoint_format = "native-parakeet-tdt-v1"
    native_architecture_family = "parakeet-tdt"

    def setup(self) -> NativeParakeetTDTTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "tdt":
            raise ValueError("Native Parakeet fine-tuning requires TDT runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native Parakeet fine-tuning must target the wrapper's exact graph.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context,
    ) -> Mapping[str, Any]:
        import torch

        required = {
            "input_features",
            "attention_mask",
            "labels",
            "decoder_input_ids",
        }
        supplied_preprocessed = required & set(inputs)
        if supplied_preprocessed and supplied_preprocessed != required:
            missing = tuple(sorted(required - set(inputs)))
            raise ValueError(
                "A preprocessed Parakeet TDT batch must provide all required "
                f"tensors; missing {', '.join(missing)}.")
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        missing = tuple(sorted(required - set(prepared)))
        if missing:
            raise ValueError(
                "Parakeet TDT training input preparation is incomplete; "
                f"missing {', '.join(missing)}.")
        dimensions = {
            "input_features": 2,
            "attention_mask": 1,
            "labels": 1,
            "decoder_input_ids": 1,
        }
        for name, unbatched_rank in dimensions.items():
            value = prepared.get(name)
            if (isinstance(value, torch.Tensor) and value.ndim == unbatched_rank):
                prepared[name] = value.unsqueeze(0)
        accepted = {
            "attention_mask",
            "decoder_input_ids",
            "input_features",
            "labels",
            "reduction",
            "sigma",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": self.checkpoint_format,
            "objective": "token-and-duration-transducer",
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

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        self.model.export_native_pretrained(destination)


__all__ = ["NativeParakeetTDTTrainingAdapter"]
