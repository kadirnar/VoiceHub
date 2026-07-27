"""Training adapter for dynamically dispatched Transformers ASR families."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import UpstreamNativeTrainingAdapter

_SUPPORTED_FAMILIES = frozenset({
    "ctc",
    "speech-seq2seq",
    "rnnt",
    "tdt",
})


class TransformersASRTrainingAdapter(UpstreamNativeTrainingAdapter):
    """Preserve each Transformers ASR checkpoint's native differentiable loss.

    The wrapper can resolve four architecture families at load time. A
    static registry family would therefore be misleading; this adapter
    records the resolved family while consistently requiring the model's
    own loss for CTC blank handling, encoder-decoder alignment, and
    transducer objectives.
    """

    supports_custom_recipe = True
    native_export_semantics = "huggingface-safetensors-and-processor"

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
            "rnnt": "RNN-T",
            "tdt": "TDT",
        }
        return names.get(family, "Transformers ASR native")

    def setup(self):
        super().setup()
        family = self.native_family
        if family not in _SUPPORTED_FAMILIES:
            choices = ", ".join(sorted(_SUPPORTED_FAMILIES))
            raise ValueError(
                "The loaded Transformers ASR checkpoint did not resolve to a "
                f"trainable family. Expected one of: {choices}; received "
                f"{family!r}.")
        return self

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration["resolved_architecture_family"] = self.native_family
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest["native_architecture_family"] = self.native_family
        return manifest

    def execute_training_phase(self, context):
        output = super().execute_training_phase(context)
        output.metadata["native_architecture_family"] = self.native_family
        output.metadata["native_objective_required"] = True
        return output

    def save_pretrained(self, save_directory) -> None:
        """Export native weights and processor in a reusable HF layout."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        self.primary_model.save_pretrained(
            destination,
            safe_serialization=True,
        )
        processor = getattr(self.model, "training_processor", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)


__all__ = ["TransformersASRTrainingAdapter"]
