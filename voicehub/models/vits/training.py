"""Explicitly opt-in waveform-reconstruction experiments for VITS.

This module does not implement the complete VITS training objective. The
Transformers inference graph omits the posterior encoder,
discriminators, and the duration, KL, feature-matching, and adversarial
losses required for native VITS fine-tuning.  The adapter below only
makes VoiceHub's deliberately limited reconstruction facade reachable
through the shared Trainer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.training.adapters import VITSTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class VitsReconstructionTrainingAdapter(VITSTrainingAdapter):
    """Run the opt-in waveform-only objective through VoiceHub's Trainer.

    ``VITSTrainingAdapter`` remains strict for arbitrary VITS-family
    models. Only this model-local specialization claims support for
    VoiceHub's custom reconstruction experiment, and only when the
    wrapper configuration explicitly enables it.
    """

    native_export_semantics = "transformers-vits-weight-warm-start"

    @property
    def experimental_reconstruction_enabled(self) -> bool:
        config = getattr(self.model, "config", None)
        return (getattr(
            config,
            "enable_experimental_reconstruction_training",
            False,
        ) is True)

    @property
    def supports_custom_recipe(self) -> bool:
        """Advertise the limited custom recipe only after explicit opt-in."""
        return self.experimental_reconstruction_enabled

    def validate_support(self) -> None:
        if not self.experimental_reconstruction_enabled:
            raise ValueError(
                "Transformers VITS does not expose the complete source "
                "training recipe. Set "
                "`enable_experimental_reconstruction_training=True` only "
                "to opt into VoiceHub's non-equivalent waveform "
                "reconstruction experiment.")
        super().validate_support()

    def create_dataset(self, records, **kwargs):
        """Create a portable dataset after validating the explicit opt-in."""
        self.validate_support()
        return SpeechDataset(records, **kwargs)

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "experimental_reconstruction_opt_in": True,
            "objective_scope": "waveform-reconstruction-only",
            "full_vits_fine_tuning": False,
        })
        return configuration

    def on_training_phase_end(self, context, output):
        """Mark every result so downstream reporters cannot imply full FT."""
        output = super().on_training_phase_end(context, output)
        output.metadata.update({
            "experimental": True,
            "objective": "waveform-reconstruction-only",
            "full_vits_fine_tuning": False,
        })
        return output


__all__ = ["VitsReconstructionTrainingAdapter"]
