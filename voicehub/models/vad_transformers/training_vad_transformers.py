"""Training adapter for native clip- and frame-level VAD models."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import AudioClassificationTrainingAdapter

_SUPPORTED_FAMILIES = frozenset({
    "audio-classification",
    "frame-classification",
})


class TransformersVADTrainingAdapter(AudioClassificationTrainingAdapter):
    """Expose native classifier losses with a safe explicit fallback."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-safetensors-and-processor"

    @property
    def native_family(self) -> str:
        return str(getattr(self.model, "architecture_family", "auto"))

    def setup(self):
        super().setup()
        if self.native_family not in _SUPPORTED_FAMILIES:
            choices = ", ".join(sorted(_SUPPORTED_FAMILIES))
            raise ValueError(
                "The loaded native VAD checkpoint did not resolve to a "
                f"trainable family. Expected one of: {choices}; received "
                f"{self.native_family!r}.")
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
        return output

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        exporter = getattr(self.model, "_save_pretrained", None)
        if not callable(exporter):
            raise TypeError("Native VAD wrapper does not expose `_save_pretrained()`.")
        exporter(destination)


__all__ = ["TransformersVADTrainingAdapter"]
