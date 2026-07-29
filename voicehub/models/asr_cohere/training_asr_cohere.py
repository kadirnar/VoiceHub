"""Fine-tuning adapter for VoiceHub-native Cohere Transcribe."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeCohereASRTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train the exact native encoder-decoder graph and export Safetensors."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-cohere-asr-safetensors-tokenizer-and-processor")

    def setup(self) -> NativeCohereASRTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "speech-seq2seq":
            raise ValueError("Native Cohere ASR fine-tuning requires the "
                             "speech-seq2seq runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native Cohere ASR fine-tuning must target the wrapper's "
                "exact `model` graph.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        """Normalize raw records and keep only graph-supported inputs."""
        import torch

        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        features = prepared.get("input_features")
        if isinstance(features, torch.Tensor) and features.ndim == 2:
            prepared["input_features"] = features.unsqueeze(0)
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
            "reduction",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        native_config = getattr(self.model, "native_config", None)
        configuration.update({
            "checkpoint_format": "native-cohere-asr-v1",
            "language_policy": "explicit-shared-language-per-batch",
            "mask_prompt_loss": getattr(
                native_config,
                "mask_prompt_loss",
                False,
            ),
            "objective": ("prompt-conditioned-teacher-forced-cross-entropy"),
            "sample_rate": 16_000,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "native-cohere-asr-v1",
            "native_architecture_family": "cohere-asr",
            "processor_runtime": "voicehub-native",
            "tokenizer_runtime": "voicehub-byte-fallback-bpe",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "cohere-asr",
            "native_objective": ("prompt-conditioned-teacher-forced-cross-entropy"),
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export one inference-ready directory after strict validation."""
        self.setup()
        self.model.export_native_pretrained(Path(save_directory).expanduser())


__all__ = ["NativeCohereASRTrainingAdapter"]
