"""Trainer integration for the native OpenVoice V2 converter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import VITSTrainingAdapter
from voicehub.training.datasets import SpeechDataset


class OpenVoiceTrainingCollator:
    """Preserve variable-length paired waveforms until native preprocessing."""

    _AUDIO_FIELDS = (
        "source_audio",
        "target_audio",
        "source_reference_audio",
        "target_reference_audio",
    )
    _RATE_FIELDS = (
        "sampling_rate",
        "source_sampling_rate",
        "target_sampling_rate",
        "source_reference_sampling_rate",
        "target_reference_sampling_rate",
    )

    @staticmethod
    def _same_value(rows: list[dict[str, Any]], name: str) -> Any:
        values = [row[name] for row in rows if name in row]
        if not values:
            return None
        if len(values) != len(rows) or any(value != values[0] for value in values):
            raise ValueError(f"Every OpenVoice sample must use the same `{name}` value.")
        return values[0]

    @staticmethod
    def _embeddings(rows: list[dict[str, Any]], name: str) -> Any:
        values = [row.get(name) for row in rows]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"OpenVoice `{name}` must be present for every sample or none.")
        import torch

        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"Batched OpenVoice `{name}` values must be tensors.")
        normalized = [
            value.squeeze(0) if value.ndim == 3 and value.shape[0] == 1 else value for value in values
        ]
        if any(value.shape != (256, 1) for value in normalized):
            raise ValueError(f"Every OpenVoice `{name}` must have shape [256, 1] "
                             "or [1, 256, 1].")
        return torch.stack(normalized)

    def __call__(
        self,
        features: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not features:
            return {}
        rows = [dict(feature) for feature in features]
        for index, row in enumerate(rows):
            if "source_audio" not in row and "audio" in row:
                row["source_audio"] = row["audio"]
            if "target_audio" not in row and "target_waveform" in row:
                row["target_audio"] = row["target_waveform"]
            missing = [name for name in ("source_audio", "target_audio") if row.get(name) is None]
            if missing:
                raise ValueError(f"OpenVoice sample {index} is missing: " + ", ".join(missing) + ".")
        result: dict[str, Any] = {
            "source_audio": tuple(row["source_audio"] for row in rows),
            "target_audio": tuple(row["target_audio"] for row in rows),
        }
        for name in self._AUDIO_FIELDS[2:]:
            values = [row.get(name) for row in rows]
            if all(value is None for value in values):
                continue
            if any(value is None for value in values):
                raise ValueError(f"OpenVoice `{name}` must be present for every sample or "
                                 "none.")
            result[name] = tuple(values)
        for name in self._RATE_FIELDS:
            value = self._same_value(rows, name)
            if value is not None:
                result[name] = value
        for name in ("source_embedding", "target_embedding"):
            value = self._embeddings(rows, name)
            if value is not None:
                result[name] = value
        for name, default in (("tau", 0.3), ("reduction", "mean")):
            values = [row.get(name, default) for row in rows]
            if any(value != values[0] for value in values):
                raise ValueError(f"Every OpenVoice sample must use the same `{name}`.")
            result[name] = values[0]
        return result

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "type": "openvoice-paired-waveform-v1",
            "batching": "variable-length-tuples-before-native-stft",
            "sample_rate": 22_050,
        }


class OpenVoiceTrainingAdapter(VITSTrainingAdapter):
    """Run VoiceHub's explicit reconstructed paired-conversion objective."""

    supports_custom_recipe = True
    native_export_semantics = "inference-reloadable-openvoice-safetensors"

    def __init__(self, model: Any, spec: Any) -> None:
        super().__init__(model, spec)
        self.data_collator = OpenVoiceTrainingCollator()

    def setup(self) -> OpenVoiceTrainingAdapter:
        super().setup()
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("OpenVoice fine-tuning must target the exact loaded converter.")
        self.primary_model.train()
        return self

    def create_dataset(
        self,
        records: Any,
        **kwargs: Any,
    ) -> SpeechDataset:
        self.validate_support()
        return SpeechDataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        accepted = {
            "source_spectrogram",
            "source_lengths",
            "source_embedding",
            "target_embedding",
            "source_reference_spectrogram",
            "source_reference_lengths",
            "target_reference_spectrogram",
            "target_reference_lengths",
            "target_waveform",
            "target_lengths",
            "tau",
            "reduction",
        }
        return {name: value for name, value in prepared.items() if name in accepted and value is not None}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "voicehub-openvoice-v2-v1",
            "objective": "reconstructed-paired-waveform-smooth-l1",
            "upstream_training_recipe_available": False,
            "upstream_training_parity": False,
            "discriminator_available": False,
            "sample_rate": 22_050,
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "native_architecture_family":
            "openvoice-v2-converter",
            "training_scope":
            "reconstructed-paired-waveform",
            "upstream_training_parity":
            False,
            "inference_reloadable":
            True,
            "required_preprocessing": [
                "paired 22.05 kHz source and target waveforms with "
                "matching linguistic content",
                "optional source and target reference waveforms or "
                "256-channel speaker embeddings",
            ],
        })
        return manifest

    def on_training_phase_end(self, context: Any, output: Any) -> Any:
        output = super().on_training_phase_end(context, output)
        output.metadata.update({
            "objective": "reconstructed-paired-waveform-smooth-l1",
            "upstream_training_parity": False,
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = [
    "OpenVoiceTrainingAdapter",
    "OpenVoiceTrainingCollator",
]
