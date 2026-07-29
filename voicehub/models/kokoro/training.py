"""Preprocessed fine-tuning for Kokoro's released decoder-side graph."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.training.adapters import AcousticTrainingAdapter
from voicehub.training.datasets import SpeechDataset

from .model import KModel


def _finite_non_negative(value: Any, *, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not 0 <= float(value) < float("inf")):
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return float(value)


class KokoroPreprocessedTrainingModel(nn.Module):
    """Differentiable objectives over the exact released inference graph.

    This is an architecture-consistent reconstruction, not the
    unpublished author recipe. Duration supervision trains PL-
    BERT/prosody. Acoustic supervision additionally trains the text
    encoder and iSTFTNet from precomputed alignment, F0, energy, and
    waveform targets.
    """

    def __init__(
        self,
        model: KModel,
        *,
        duration_loss_weight: float = 1.0,
        f0_loss_weight: float = 1.0,
        energy_loss_weight: float = 1.0,
        waveform_loss_weight: float = 1.0,
        spectral_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(model, KModel):
            raise TypeError("Kokoro training requires the native KModel.")
        self.native_model = model
        for name, value in (
            ("duration_loss_weight", duration_loss_weight),
            ("f0_loss_weight", f0_loss_weight),
            ("energy_loss_weight", energy_loss_weight),
            ("waveform_loss_weight", waveform_loss_weight),
            ("spectral_loss_weight", spectral_loss_weight),
        ):
            setattr(
                self,
                name,
                _finite_non_negative(value, name=name),
            )

    @staticmethod
    def _tensor(
        value: Any,
        *,
        name: str,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Kokoro fine-tuning requires `{name}`.")
        tensor = (value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
        if tensor.is_complex():
            raise TypeError(f"Kokoro `{name}` cannot use a complex dtype.")
        return tensor.to(device=device, dtype=dtype or tensor.dtype)

    @classmethod
    def _integer_tensor(
        cls,
        value: Any,
        *,
        name: str,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = cls._tensor(
            value,
            name=name,
            device=device,
        )
        if tensor.dtype == torch.bool or tensor.is_floating_point():
            raise TypeError(f"Kokoro `{name}` must use an integer dtype.")
        return tensor.long()

    @staticmethod
    def _masked_smooth_l1(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        error = torch.nn.functional.smooth_l1_loss(
            prediction.float(),
            target.float(),
            reduction="none",
        )
        weights = mask.to(device=error.device, dtype=error.dtype)
        while weights.ndim < error.ndim:
            weights = weights.unsqueeze(-1)
        denominator = weights.expand_as(error).sum().clamp_min(1)
        return (error * weights).sum() / denominator

    @staticmethod
    def _waveform_target(
        value: Any,
        *,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        target = (value if isinstance(value, torch.Tensor) else torch.as_tensor(value))
        target = target.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if target.ndim == 1:
            target = target[None, None, :]
        elif target.ndim == 2:
            target = target[:, None, :]
        if target.ndim != 3 or target.shape[1] != 1:
            raise ValueError(
                "Kokoro `audio_values` must have shape [batch, samples] or "
                "[batch, 1, samples].")
        if target.shape[0] != prediction.shape[0]:
            raise ValueError("Kokoro waveform target batch does not match input IDs.")
        if not bool(torch.isfinite(target).all()):
            raise ValueError("Kokoro waveform targets contain NaN or infinity.")
        return target

    @staticmethod
    def _spectral_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        maximum = min(prediction.shape[-1], target.shape[-1])
        if lengths is None:
            lengths = torch.full(
                (prediction.shape[0], ),
                maximum,
                device=prediction.device,
                dtype=torch.long,
            )
        losses = []
        for batch_index, item_length in enumerate(lengths):
            length = int(item_length.item())
            item_prediction = prediction[
                batch_index,
                0,
                :length,
            ].float()
            item_target = target[
                batch_index,
                0,
                :length,
            ].float()
            for n_fft in (64, 128, 256, 512, 1_024):
                if length < n_fft:
                    continue
                window = torch.hann_window(
                    n_fft,
                    device=prediction.device,
                    dtype=item_prediction.dtype,
                )
                prediction_stft = torch.stft(
                    item_prediction,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    window=window,
                    return_complex=True,
                )
                target_stft = torch.stft(
                    item_target,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    window=window,
                    return_complex=True,
                )
                losses.append(
                    torch.nn.functional.l1_loss(
                        torch.log1p(prediction_stft.abs()),
                        torch.log1p(target_stft.abs()),
                    ))
        if not losses:
            return prediction.new_zeros(())
        return torch.stack(losses).mean()

    def _audio_lengths(
        self,
        value: Any,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        maximum = min(prediction.shape[-1], target.shape[-1])
        if value is None:
            return torch.full(
                (prediction.shape[0], ),
                maximum,
                device=prediction.device,
                dtype=torch.long,
            )
        lengths = self._tensor(
            value,
            name="audio_lengths",
            device=prediction.device,
        )
        if lengths.dtype == torch.bool or lengths.is_floating_point():
            raise TypeError("Kokoro `audio_lengths` must use an integer dtype.")
        if tuple(lengths.shape) != (prediction.shape[0], ):
            raise ValueError("Kokoro `audio_lengths` must have shape [batch].")
        lengths = lengths.long()
        if bool(((lengths < 1) | (lengths > maximum)).any()):
            raise ValueError(
                "Kokoro `audio_lengths` must be positive and no greater "
                "than both predicted and target waveform lengths.")
        return lengths

    def _encoded(
        self,
        input_ids: Any,
        *,
        ref_s: Any,
        input_lengths: Any = None,
    ) -> dict[str, torch.Tensor]:
        input_ids = self._integer_tensor(
            input_ids,
            name="input_ids",
            device=self.native_model.device,
        )
        ref_s = self._tensor(
            ref_s,
            name="ref_s",
            device=self.native_model.device,
            dtype=self.native_model.dtype,
        )
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if ref_s.ndim == 1:
            ref_s = ref_s.unsqueeze(0)
        lengths = None
        if input_lengths is not None:
            lengths = self._integer_tensor(
                input_lengths,
                name="input_lengths",
                device=self.native_model.device,
            )
        return self.native_model.encode_text(
            input_ids,
            input_lengths=lengths,
            ref_s=ref_s,
        )

    def _duration_loss(
        self,
        encoded: Mapping[str, torch.Tensor],
        durations: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = self._tensor(
            durations,
            name="durations",
            device=self.native_model.device,
        )
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if target.shape != encoded["text_mask"].shape:
            raise ValueError("Kokoro `durations` must match [batch, text] input shape.")
        if target.dtype == torch.bool or target.is_floating_point():
            raise TypeError("Kokoro `durations` must use an integer dtype.")
        if bool((target < 0).any()):
            raise ValueError("Kokoro `durations` cannot be negative.")
        target = target.to(dtype=torch.float32)
        prediction = torch.sigmoid(encoded["duration_logits"]).sum(dim=-1)
        loss = self._masked_smooth_l1(
            prediction,
            target,
            ~encoded["text_mask"],
        )
        return loss, target.long()

    def duration_objective(
        self,
        input_ids: Any,
        *,
        ref_s: Any,
        durations: Any,
        input_lengths: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train duration/PL-BERT without running the waveform decoder."""
        del kwargs
        encoded = self._encoded(
            input_ids,
            ref_s=ref_s,
            input_lengths=input_lengths,
        )
        duration_loss, _ = self._duration_loss(encoded, durations)
        loss = self.duration_loss_weight * duration_loss
        return {
            "loss": loss,
            "logits": encoded["duration_logits"],
            "losses": {
                "loss": loss,
                "duration_loss": duration_loss,
            },
            "metadata": {
                "objective": "preprocessed-duration-reconstruction",
                "recipe_status": "reconstructed-not-author-verified",
                "raw_audio_training": False,
            },
        }

    def acoustic_objective(
        self,
        input_ids: Any,
        *,
        ref_s: Any,
        durations: Any,
        audio_values: Any,
        input_lengths: Any = None,
        alignment: Any = None,
        f0_targets: Any = None,
        energy_targets: Any = None,
        audio_lengths: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Train the complete released decoder from prepared supervision."""
        del kwargs
        encoded = self._encoded(
            input_ids,
            ref_s=ref_s,
            input_lengths=input_lengths,
        )
        duration_loss, duration_values = self._duration_loss(
            encoded,
            durations,
        )
        if alignment is None:
            alignment_tensor = self.native_model.alignment_from_durations(
                duration_values,
                text_mask=encoded["text_mask"],
            )
        else:
            alignment_tensor = self._tensor(
                alignment,
                name="alignment",
                device=self.native_model.device,
                dtype=self.native_model.dtype,
            )
            if alignment_tensor.ndim == 2:
                alignment_tensor = alignment_tensor.unsqueeze(0)
        decoded = self.native_model.decode_aligned(
            dict(encoded),
            alignment_tensor,
        )
        waveform = decoded["waveform"]
        target_waveform = self._waveform_target(
            audio_values,
            prediction=waveform,
        )
        common_length = min(
            waveform.shape[-1],
            target_waveform.shape[-1],
        )
        if common_length < 1:
            raise ValueError("Kokoro waveform targets cannot be empty.")
        waveform_lengths = self._audio_lengths(
            audio_lengths,
            prediction=waveform,
            target=target_waveform,
        )
        waveform_error = torch.nn.functional.smooth_l1_loss(
            waveform[..., :common_length].float(),
            target_waveform[..., :common_length].float(),
            reduction="none",
        )
        waveform_mask = (
            torch.arange(common_length, device=waveform.device)[None, :] < waveform_lengths[:, None])
        waveform_loss = (waveform_error * waveform_mask[:, None, :]).sum() / waveform_mask.sum().clamp_min(1)
        spectral_loss = self._spectral_loss(
            waveform,
            target_waveform,
            lengths=waveform_lengths,
        )
        zero = waveform.new_zeros((), dtype=torch.float32)
        f0_loss = zero
        frame_mask = alignment_tensor.sum(dim=1) > 0
        if f0_targets is not None:
            f0_target = self._tensor(
                f0_targets,
                name="f0_targets",
                device=self.native_model.device,
                dtype=decoded["f0"].dtype,
            )
            if f0_target.ndim == 1:
                f0_target = f0_target.unsqueeze(0)
            if f0_target.shape != decoded["f0"].shape:
                raise ValueError(
                    "Kokoro `f0_targets` must match predicted shape "
                    f"{tuple(decoded['f0'].shape)}.")
            f0_mask = torch.nn.functional.interpolate(
                frame_mask[:, None, :].float(),
                size=decoded["f0"].shape[-1],
                mode="nearest",
            ).squeeze(1).bool()
            f0_loss = self._masked_smooth_l1(
                decoded["f0"],
                f0_target,
                f0_mask,
            )
        energy_loss = zero
        if energy_targets is not None:
            energy_target = self._tensor(
                energy_targets,
                name="energy_targets",
                device=self.native_model.device,
                dtype=decoded["energy"].dtype,
            )
            if energy_target.ndim == 1:
                energy_target = energy_target.unsqueeze(0)
            if energy_target.shape != decoded["energy"].shape:
                raise ValueError(
                    "Kokoro `energy_targets` must match predicted shape "
                    f"{tuple(decoded['energy'].shape)}.")
            energy_mask = torch.nn.functional.interpolate(
                frame_mask[:, None, :].float(),
                size=decoded["energy"].shape[-1],
                mode="nearest",
            ).squeeze(1).bool()
            energy_loss = self._masked_smooth_l1(
                decoded["energy"],
                energy_target,
                energy_mask,
            )
        loss = (
            self.duration_loss_weight * duration_loss + self.f0_loss_weight * f0_loss +
            self.energy_loss_weight * energy_loss + self.waveform_loss_weight * waveform_loss +
            self.spectral_loss_weight * spectral_loss)
        return {
            "loss": loss,
            "logits": waveform,
            "audio_values": waveform,
            "losses": {
                "loss": loss,
                "duration_loss": duration_loss,
                "f0_loss": f0_loss,
                "energy_loss": energy_loss,
                "waveform_loss": waveform_loss,
                "spectral_loss": spectral_loss,
            },
            "metadata": {
                "objective": "preprocessed-decoder-reconstruction",
                "recipe_status": "reconstructed-not-author-verified",
                "released_graph_fine_tuning": True,
                "raw_audio_training": False,
            },
        }

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.acoustic_objective(*args, **kwargs)


class KokoroTrainingAdapter(AcousticTrainingAdapter):
    """VoiceHub trainer integration for reconstructed Kokoro objectives."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-kokoro-safetensors"

    def validate_support(self) -> None:
        if not bool(getattr(
                self.model.config,
                "enable_preprocessed_training",
                False,
        )):
            raise ValueError(
                "Kokoro exposes an explicitly reconstructed preprocessed "
                "decoder fine-tuning path. Set "
                "`enable_preprocessed_training=True` and supply phoneme IDs, "
                "a 256-D style vector, durations/alignment, and prepared "
                "acoustic targets. This does not enable the unpublished "
                "StyleTTS2 raw-audio recipe.")
        super().validate_support()

    def setup(self) -> KokoroTrainingAdapter:
        super().setup()
        if self.primary_model is not getattr(
                self.model,
                "training_model",
                None,
        ):
            raise ValueError("Kokoro training must target its exact preprocessed "
                             "objective facade.")
        if self.primary_model.native_model is not self.model.model:
            raise ValueError("Kokoro training facade is detached from the loaded graph.")
        return self

    def create_dataset(self, records: Any, **kwargs: Any) -> SpeechDataset:
        self.validate_support()
        return SpeechDataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        return self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format":
            "voicehub-kokoro-v1",
            "objective_scope":
            "preprocessed-decoder-reconstruction",
            "recipe_status":
            "reconstructed-not-author-verified",
            "raw_audio_styletts2_recipe":
            False,
            "blocking_requirements": [
                "unreleased Kokoro alignment/phonemization data pipeline",
                "unreleased style encoder and diffusion checkpoints",
                "unreleased discriminator and optimizer schedule",
            ],
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-kokoro-v1",
            "native_architecture_family": "kokoro",
            "training_scope": "preprocessed-decoder-reconstruction",
            "author_verified_full_recipe": False,
        })
        return manifest

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model._save_pretrained(destination)


__all__ = [
    "KokoroPreprocessedTrainingModel",
    "KokoroTrainingAdapter",
]
