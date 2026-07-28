"""Native Hugging Face VITS and MMS-TTS inference integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.audio import AudioInput, load_audio
from voicehub.modeling_outputs import TTSOutput
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models._transformers_tts import TransformersTTSConfigBase, TransformersTTSModelBase


class VitsConfig(TransformersTTSConfigBase):
    """Loading and synthesis controls for VITS-compatible checkpoints."""

    model_type = "vits"

    def __init__(
        self,
        *,
        speaking_rate: float = 1.0,
        noise_scale: float | None = None,
        noise_scale_duration: float | None = None,
        enable_experimental_reconstruction_training: bool = False,
        training_spectral_loss_weight: float = 0.1,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.speaking_rate = speaking_rate
        self.noise_scale = noise_scale
        self.noise_scale_duration = noise_scale_duration
        self.enable_experimental_reconstruction_training = (enable_experimental_reconstruction_training)
        self.training_spectral_loss_weight = training_spectral_loss_weight
        self.validate()

    def validate(self) -> None:
        super().validate()
        speaking_rate = getattr(self, "speaking_rate", 1.0)
        self.speaking_rate = self._validate_real(
            speaking_rate,
            name="speaking_rate",
        )
        for name in ("noise_scale", "noise_scale_duration"):
            value = getattr(self, name, None)
            if value is not None:
                setattr(
                    self,
                    name,
                    self._validate_real(
                        value,
                        name=name,
                        allow_zero=True,
                    ),
                )
        enabled = getattr(
            self,
            "enable_experimental_reconstruction_training",
            False,
        )
        if not isinstance(enabled, bool):
            raise TypeError("`enable_experimental_reconstruction_training` must be a boolean.")
        self.enable_experimental_reconstruction_training = enabled
        weight = getattr(self, "training_spectral_loss_weight", 0.1)
        self.training_spectral_loss_weight = self._validate_real(
            weight,
            name="training_spectral_loss_weight",
            allow_zero=True,
        )

    @staticmethod
    def _validate_real(
        value: Any,
        *,
        name: str,
        allow_zero: bool = False,
    ) -> float:
        from math import isfinite

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{name}` must be a real number.")
        value = float(value)
        valid = value >= 0 if allow_zero else value > 0
        if not isfinite(value) or not valid:
            qualifier = "non-negative" if allow_zero else "greater than zero"
            raise ValueError(f"`{name}` must be finite and {qualifier}.")
        return value


def _build_vits_training_model(
    torch: Any,
    model: Any,
    *,
    spectral_loss_weight: float,
):
    """Wrap VITS synthesis in an experimental waveform objective.

    This is deliberately not advertised as native VITS fine-tuning.
    Upstream Transformers raises when ``labels`` are passed, and
    inference synthesis does not reproduce the source posterior,
    duration, KL, discriminator, feature-matching, or adversarial
    objectives. The facade exists only for explicitly enabled
    reconstruction experiments.
    """

    class VitsReconstructionModel(torch.nn.Module):

        def __init__(self, native_model):
            super().__init__()
            self.native_model = native_model
            self.spectral_loss_weight = float(spectral_loss_weight)

        @staticmethod
        def _waveform(outputs):
            if isinstance(outputs, Mapping):
                return outputs.get("waveform")
            return getattr(outputs, "waveform", None)

        @staticmethod
        def _as_waveform_batch(value, reference):
            if hasattr(value, "detach"):
                target = value
            else:
                target = torch.as_tensor(value)
            target = target.to(
                device=reference.device,
                dtype=reference.dtype,
            )
            while target.ndim > 2 and 1 in target.shape:
                target = target.squeeze()
            if target.ndim == 1:
                target = target.unsqueeze(0)
            if target.ndim != 2:
                raise ValueError(
                    "VITS waveform targets must have shape (batch, samples); "
                    f"received {tuple(target.shape)}.")
            if target.shape[0] != reference.shape[0]:
                raise ValueError("VITS waveform target batch size does not match the text "
                                 "batch.")
            return target

        @staticmethod
        def _spectral_loss(prediction, target):
            losses = []
            sample_count = prediction.shape[-1]
            for n_fft in (256, 512, 1024):
                if sample_count < n_fft:
                    continue
                window = torch.hann_window(
                    n_fft,
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                predicted_stft = torch.stft(
                    prediction,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=window,
                    return_complex=True,
                )
                target_stft = torch.stft(
                    target,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    win_length=n_fft,
                    window=window,
                    return_complex=True,
                )
                losses.append(
                    torch.nn.functional.l1_loss(
                        torch.log1p(predicted_stft.abs()),
                        torch.log1p(target_stft.abs()),
                    ))
            if not losses:
                return prediction.new_zeros(())
            return torch.stack(losses).mean()

        def forward(
            self,
            input_ids,
            *,
            audio_values=None,
            labels=None,
            attention_mask=None,
            speaker_id=None,
            speaking_rate=None,
            **kwargs,
        ):
            del kwargs
            target = audio_values if audio_values is not None else labels
            if target is None:
                raise ValueError("VITS fine-tuning requires `audio_values` waveform "
                                 "targets.")
            outputs = self.native_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speaker_id=speaker_id,
                speaking_rate=speaking_rate,
                return_dict=True,
            )
            waveform = self._waveform(outputs)
            if waveform is None:
                raise RuntimeError("The native VITS model returned no waveform for training.")
            target = self._as_waveform_batch(target, waveform)
            common_length = min(waveform.shape[-1], target.shape[-1])
            if common_length <= 0:
                raise ValueError("VITS waveform targets cannot be empty.")
            prediction = waveform[..., :common_length]
            target = target[..., :common_length]
            waveform_loss = torch.nn.functional.smooth_l1_loss(
                prediction,
                target,
            )
            spectral_loss = self._spectral_loss(prediction, target)
            loss = waveform_loss + self.spectral_loss_weight * spectral_loss
            return {
                "loss": loss,
                "waveform": waveform,
                "audio_values": waveform,
                "losses": {
                    "waveform_loss": waveform_loss,
                    "spectral_loss": spectral_loss,
                },
            }

    return VitsReconstructionModel(model)


class VitsForTextToSpeech(TransformersTTSModelBase):
    """Load VITS or any of Meta's 1,100+ MMS-TTS language checkpoints."""

    config_class = VitsConfig
    default_model_name_or_path = "facebook/mms-tts-eng"
    transformers_model_class = "VitsModel"
    transformers_processor_class = "VitsTokenizer"
    passthrough_generation_options = frozenset({
        "output_attentions",
        "output_hidden_states",
    })

    def __init__(
        self,
        config: VitsConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
            token=token,
        )

    def _load_pretrained_model(self) -> None:
        _, model, _ = self._load_transformers_model_and_processor()
        self.model = model
        sample_rate = getattr(
            getattr(model, "config", None),
            "sampling_rate",
            self.config.sample_rate,
        )
        self.config.sample_rate = int(sample_rate)

    def _prepare_for_training(self) -> None:
        if not self.config.enable_experimental_reconstruction_training:
            raise ValueError(
                "Transformers VITS does not expose the complete source training "
                "recipe. Set `enable_experimental_reconstruction_training=True` "
                "only to opt into VoiceHub's non-equivalent waveform "
                "reconstruction experiment.")
        super()._prepare_for_training()
        if self.training_model is None:
            self.training_model = _build_vits_training_model(
                self._torch,
                self.model,
                spectral_loss_weight=(self.config.training_spectral_loss_weight),
            )
        self.training_model.train()

    def _prepare_for_inference(self) -> None:
        super()._prepare_for_inference()
        if self.training_model is not None:
            self.training_model.eval()

    @staticmethod
    def _sampling_rate_value(value: Any) -> int | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            values = value.detach().reshape(-1)
            if values.numel() == 0:
                return None
            first = int(values[0].item())
            if values.numel() > 1 and not bool((values == first).all().item()):
                raise ValueError("Every VITS waveform in a batch must share one sampling "
                                 "rate.")
            return first
        return int(value)

    def _materialize_training_audio(
        self,
        audio: Any,
        *,
        sampling_rate: Any,
    ):
        if hasattr(audio, "detach") and getattr(audio, "ndim", 0) >= 2:
            source_rate = self._sampling_rate_value(sampling_rate)
            if source_rate is not None and source_rate != self.sample_rate:
                raise ValueError(
                    "Batched VITS waveform tensors must already be resampled "
                    f"to {self.sample_rate} Hz.")
            return audio
        if (isinstance(audio, (list, tuple)) and audio and not isinstance(audio[0], (int, float))):
            materialized = [
                load_audio(
                    value,
                    sampling_rate=self._sampling_rate_value(sampling_rate),
                    target_sampling_rate=self.sample_rate,
                ).waveform for value in audio
            ]
            return self._torch.as_tensor(materialized, dtype=self._torch.float32)
        loaded = load_audio(
            audio,
            sampling_rate=self._sampling_rate_value(sampling_rate),
            target_sampling_rate=self.sample_rate,
        )
        return self._torch.as_tensor(
            loaded.waveform,
            dtype=self._torch.float32,
        ).unsqueeze(0)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Tokenize text and normalize waveform targets for VITS training."""
        if phase != "waveform_reconstruction":
            raise ValueError(f"Unknown VITS training phase {phase!r}.")
        prepared = dict(inputs)
        if "input_ids" not in prepared:
            text = prepared.pop("text", None)
            if text is None:
                raise ValueError("VITS fine-tuning requires `input_ids` or raw `text`.")
            tokenized = self.transformers_processor(
                text=text,
                padding=True,
                return_tensors="pt",
            )
            if not isinstance(tokenized, Mapping):
                raise TypeError("VitsTokenizer must return a mapping.")
            prepared.update(dict(tokenized))
        audio = prepared.pop(
            "audio",
            prepared.get("audio_values", prepared.get("labels")),
        )
        if audio is None:
            raise ValueError("VITS fine-tuning requires `audio_values` or raw `audio`.")
        prepared["audio_values"] = self._materialize_training_audio(
            audio,
            sampling_rate=prepared.pop("sampling_rate", None),
        )
        prepared.pop("labels", None)
        return prepared

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_id = model_inputs.get("speaker_id")
        if speaker_id is not None and (isinstance(speaker_id, bool) or not isinstance(speaker_id, int) or
                                       speaker_id < 0):
            raise ValueError("`speaker_id` must be a non-negative integer or None.")

    def _generate(
        self,
        text: str,
        *,
        speaker_id: int | None = None,
        speaking_rate: float | None = None,
        speed: float | None = None,
        noise_scale: float | None = None,
        noise_scale_duration: float | None = None,
        normalize: bool | None = None,
        output_file: str | Path | None = None,
        seed: int | None = None,
        **model_options,
    ) -> TTSOutput:
        if speaking_rate is not None and speed is not None:
            raise ValueError("Pass `speaking_rate` or `speed`, not both.")
        speaking_rate = (
            speaking_rate
            if speaking_rate is not None else speed if speed is not None else self.config.speaking_rate)
        speaking_rate = self._positive_real(
            speaking_rate,
            name="speaking_rate",
        )
        if noise_scale is None:
            noise_scale = self.config.noise_scale
        if noise_scale_duration is None:
            noise_scale_duration = self.config.noise_scale_duration
        if noise_scale is not None:
            noise_scale = self._positive_real(
                noise_scale,
                name="noise_scale",
                allow_zero=True,
            )
        if noise_scale_duration is not None:
            noise_scale_duration = self._positive_real(
                noise_scale_duration,
                name="noise_scale_duration",
                allow_zero=True,
            )

        processor_options = {}
        if normalize is not None:
            if not isinstance(normalize, bool):
                raise TypeError("`normalize` must be a boolean or None.")
            processor_options["normalize"] = normalize
        inputs = self._processor_inputs(text, **processor_options)
        num_speakers = int(getattr(getattr(self.model, "config", None), "num_speakers", 1))
        if speaker_id is not None and speaker_id >= num_speakers:
            raise ValueError(f"`speaker_id` must be smaller than {num_speakers} for this "
                             "checkpoint.")

        previous_noise = getattr(self.model, "noise_scale", None)
        previous_duration_noise = getattr(
            self.model,
            "noise_scale_duration",
            None,
        )
        if noise_scale is not None:
            self.model.noise_scale = noise_scale
        if noise_scale_duration is not None:
            self.model.noise_scale_duration = noise_scale_duration
        try:
            with seeded_inference(
                    seed,
                    device=self.device,
                    model_type=self.config.model_type,
            ) as effective_seed:
                with self._torch.inference_mode():
                    generated = self.model(
                        **inputs,
                        speaker_id=speaker_id,
                        speaking_rate=speaking_rate,
                        return_dict=True,
                        **model_options,
                    )
        finally:
            if previous_noise is not None:
                self.model.noise_scale = previous_noise
            if previous_duration_noise is not None:
                self.model.noise_scale_duration = previous_duration_noise

        waveform_value = (
            generated.get("waveform") if isinstance(generated, Mapping) else getattr(
                generated, "waveform", None))
        output_lengths = (
            generated.get("sequence_lengths") if isinstance(generated, Mapping) else getattr(
                generated, "sequence_lengths", None))
        if waveform_value is None:
            raise RuntimeError("The native VITS model returned no waveform.")
        waveform = self._normalize_waveform(
            waveform_value,
            output_length=output_lengths,
        )
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend":
                "transformers",
                "checkpoint_family":
                "mms-tts/vits",
                "speaker_id":
                speaker_id,
                "speaking_rate":
                speaking_rate,
                "noise_scale": (noise_scale if noise_scale is not None else previous_noise),
                "noise_scale_duration":
                (noise_scale_duration if noise_scale_duration is not None else previous_duration_noise),
                "seed":
                effective_seed,
                "requested_seed":
                seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self._save_native_bundle(save_directory)


MmsTTSForTextToSpeech = VitsForTextToSpeech
VitsTTS = VitsForTextToSpeech

__all__ = [
    "MmsTTSForTextToSpeech",
    "VitsConfig",
    "VitsForTextToSpeech",
    "VitsTTS",
]
