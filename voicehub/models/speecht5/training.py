"""Source-faithful fine-tuning adapter for native SpeechT5."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import Seq2SeqTrainingAdapter
from voicehub.training.contracts import TrainingContext


class NativeSpeechT5TrainingAdapter(Seq2SeqTrainingAdapter):
    """Train the complete text-to-spectrogram graph and freeze HiFi-GAN.

    SpeechT5's published fine-tuning objective is already implemented by
    the acoustic model.  The adapter owns lifecycle validation, raw-data
    preprocessing, and creation of an inference-reloadable native
    artifact.
    """

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-safetensors"
    RECIPE_VERSION = 1

    def setup(self) -> NativeSpeechT5TrainingAdapter:
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "SpeechT5 fine-tuning must target the wrapper's complete "
                "text-to-spectrogram model.")
        vocoder = getattr(self.model, "vocoder", None)
        if vocoder is None:
            raise TypeError("SpeechT5 fine-tuning requires its paired vocoder.")
        vocoder.eval()
        vocoder.requires_grad_(False)
        processor = getattr(self.model, "training_processor", None)
        if processor is None:
            raise TypeError("SpeechT5 fine-tuning requires its native processor.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        accepted = {
            "attention_mask",
            "decoder_attention_mask",
            "decoder_input_values",
            "input_ids",
            "labels",
            "speaker_embeddings",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export the trained acoustic graph and frozen native vocoder."""
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format":
            "voicehub-speecht5-v1",
            "native_architecture_family":
            "speecht5-text-to-speech",
            "training_scope":
            "complete-text-to-spectrogram",
            "objective": [
                "pre-postnet-and-postnet-l1",
                "weighted-stop-token-bce",
                "guided-multihead-cross-attention",
            ],
            "raw_data_fine_tuning":
            True,
            "speaker_embedding_dimension":
            int(self.model.native_config.speaker_embedding_dim),
            "frozen_components": ["vocoder"],
            "inference_reloadable":
            True,
        })
        return manifest


__all__ = ["NativeSpeechT5TrainingAdapter"]
