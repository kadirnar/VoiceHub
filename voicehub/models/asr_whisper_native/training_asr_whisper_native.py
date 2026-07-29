"""Fine-tuning adapter for VoiceHub's native Whisper runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeWhisperTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train and export Whisper without an upstream model runtime.

    Whisper's native forward pass returns its teacher-forced cross-entropy
    loss directly. The generic sequence objective remains available only as a
    defensive fallback for compatible custom graphs.
    """

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-whisper-safetensors-and-processor"

    def setup(self) -> NativeWhisperTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "speech-seq2seq":
            raise ValueError(
                "Native Whisper fine-tuning requires the speech-seq2seq "
                "runtime."
            )
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native Whisper fine-tuning must target the wrapper's exact "
                "`model` graph."
            )
        return self

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update(
            {
                "checkpoint_format": "native-whisper-v1",
                "objective": "teacher-forced-cross-entropy",
                "sample_rate": 16_000,
            }
        )
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update(
            {
                "native_architecture_family": "whisper",
                "checkpoint_format": "native-whisper-v1",
                "processor_runtime": "voicehub-native",
            }
        )
        return manifest

    def execute_training_phase(self, context):
        output = super().execute_training_phase(context)
        output.metadata.update(
            {
                "native_architecture_family": "whisper",
                "native_objective": "teacher-forced-cross-entropy",
            }
        )
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Export an inference-ready native Whisper directory."""
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["NativeWhisperTrainingAdapter"]
