"""Full-model native fine-tuning adapter for SeamlessM4T-v2 S2T."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeSeamlessM4Tv2TrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train the complete Conformer/decoder graph with sequence CE."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-seamless-m4t-v2-s2t-safetensors-and-processor")
    runtime_name = "SeamlessM4T-v2 S2T"
    checkpoint_format = "native-seamless-m4t-v2-s2t-v1"
    native_architecture_family = "seamless-m4t-v2-s2t"

    def setup(self) -> NativeSeamlessM4Tv2TrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "speech-seq2seq":
            raise ValueError("Native SeamlessM4T-v2 fine-tuning requires speech seq2seq.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Fine-tuning must target the wrapper's exact native graph.")
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
        required = {
            "attention_mask",
            "input_features",
            "labels",
        }
        missing = tuple(sorted(required - set(prepared)))
        if missing:
            raise ValueError(
                "SeamlessM4T-v2 training preparation is incomplete; "
                f"missing {', '.join(missing)}.")
        accepted = {
            "attention_mask",
            "decoder_attention_mask",
            "decoder_input_ids",
            "input_features",
            "labels",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": self.checkpoint_format,
            "fine_tuning_scope": "full-model",
            "objective": "language-conditioned-sequence-cross-entropy",
            "sample_rate": 16_000,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": self.checkpoint_format,
            "fine_tuning_scope": "full-model",
            "native_architecture_family": self.native_architecture_family,
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(Path(save_directory).expanduser())


__all__ = ["NativeSeamlessM4Tv2TrainingAdapter"]
