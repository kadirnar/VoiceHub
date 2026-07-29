"""Training adapter for VoiceHub-native PyanNet providers."""

from __future__ import annotations

from pathlib import Path

from voicehub.training.adapters import FrameClassificationTrainingAdapter


class NativePyanNetTrainingAdapter(FrameClassificationTrainingAdapter):
    """Preserve model-owned objectives and native Safetensors export."""

    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native PyanNet training requires a wrapper with "
                            "export_native_pretrained().")
        export(Path(save_directory))


__all__ = ["NativePyanNetTrainingAdapter"]
