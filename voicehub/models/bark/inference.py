"""Native Bark inference and pre-tokenized stage-objective integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.modeling_outputs import TTSOutput
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models._transformers_tts import TransformersTTSConfigBase, TransformersTTSModelBase


class BarkConfig(TransformersTTSConfigBase):
    """Loading controls for ``transformers.BarkModel`` checkpoints."""

    model_type = "bark"

    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)


def _build_bark_training_model(torch: Any, model: Any):
    """Build differentiable losses around Bark's three public submodels.

    Transformers currently exposes logits for Bark's submodels but
    rejects ``labels``. VoiceHub computes stage-aligned token cross-
    entropies outside the upstream modules, retaining their parameter
    topology and safetensors state-dict names. This consumes pre-
    tokenized, stage-specific batches; it is not an end-to-end raw-audio
    Bark recipe.
    """

    class BarkTokenObjective(torch.nn.Module):

        def __init__(self, component, *, shift_labels: bool, fine: bool = False):
            super().__init__()
            self.component = component
            self.shift_labels = shift_labels
            self.fine = fine

        @staticmethod
        def _codebook_index(value):
            if value is None:
                raise ValueError("Bark fine-codebook training requires `codebook_idx`.")
            if hasattr(value, "detach"):
                flattened = value.detach().reshape(-1)
                if flattened.numel() == 0:
                    raise ValueError("`codebook_idx` cannot be empty.")
                first = int(flattened[0].item())
                if flattened.numel() > 1 and not bool((flattened == first).all().item()):
                    raise ValueError(
                        "Every sample in a Bark fine-codebook batch must use "
                        "the same `codebook_idx`.")
                return first
            return int(value)

        def forward(
            self,
            input_ids,
            *,
            labels,
            attention_mask=None,
            codebook_idx=None,
            **kwargs,
        ):
            del kwargs
            options = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            }
            if self.fine:
                options["codebook_idx"] = self._codebook_index(codebook_idx)
            else:
                options["use_cache"] = False
            outputs = self.component(**options)
            logits = (
                outputs.get("logits") if isinstance(outputs, Mapping) else getattr(outputs, "logits", None))
            if logits is None:
                raise RuntimeError("The Bark training component returned no token logits.")
            common_length = min(logits.shape[-2], labels.shape[-1])
            if self.shift_labels:
                if common_length < 2:
                    raise ValueError("Bark causal training requires at least two aligned "
                                     "tokens.")
                logits_for_loss = logits[..., :common_length - 1, :].contiguous()
                labels_for_loss = labels[..., 1:common_length].contiguous()
            else:
                logits_for_loss = logits[..., :common_length, :].contiguous()
                labels_for_loss = labels[..., :common_length].contiguous()
            loss = torch.nn.functional.cross_entropy(
                logits_for_loss.reshape(-1, logits_for_loss.shape[-1]),
                labels_for_loss.reshape(-1).long(),
                ignore_index=-100,
            )
            return {
                "loss": loss,
                "logits": logits,
            }

    class BarkTrainingModel(torch.nn.Module):

        def __init__(self, native_model):
            super().__init__()
            self.semantic = BarkTokenObjective(
                native_model.semantic,
                shift_labels=True,
            )
            self.coarse = BarkTokenObjective(
                native_model.coarse_acoustics,
                shift_labels=True,
            )
            self.fine = BarkTokenObjective(
                native_model.fine_acoustics,
                shift_labels=False,
                fine=True,
            )

    return BarkTrainingModel(model)


class BarkForTextToSpeech(TransformersTTSModelBase):
    """Generate speech and train Bark stages from aligned token batches."""

    config_class = BarkConfig
    default_model_name_or_path = "suno/bark-small"
    transformers_model_class = "BarkModel"
    transformers_processor_class = "BarkProcessor"

    def __init__(
        self,
        config: BarkConfig | str | Path | None = None,
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
        codec_config = getattr(self.native_config, "codec_config", None)
        sample_rate = getattr(codec_config, "sampling_rate", None)
        if sample_rate is None:
            generation_config = getattr(model, "generation_config", None)
            sample_rate = getattr(
                generation_config,
                "sample_rate",
                self.config.sample_rate,
            )
        self.config.sample_rate = int(sample_rate)

    def _prepare_for_training(self) -> None:
        super()._prepare_for_training()
        if self.training_model is None:
            self.training_model = _build_bark_training_model(
                self._torch,
                self.model,
            )
        self.training_model.train()

    def _prepare_for_inference(self) -> None:
        super()._prepare_for_inference()
        if self.training_model is not None:
            self.training_model.eval()

    @staticmethod
    def _phase_inputs(
        inputs: Mapping[str, Any],
        *,
        prefix: str,
    ) -> dict[str, Any]:
        output = {}
        names = {
            f"{prefix}_input_ids": "input_ids",
            f"{prefix}_attention_mask": "attention_mask",
            f"{prefix}_labels": "labels",
        }
        for source, target in names.items():
            if source in inputs:
                output[target] = inputs[source]
        for name in ("input_ids", "attention_mask", "labels"):
            if name in inputs and name not in output:
                output[name] = inputs[name]
        if prefix == "fine" and "codebook_idx" in inputs:
            output["codebook_idx"] = inputs["codebook_idx"]
        return output

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Select the precomputed token batch for one Bark training stage."""
        if phase not in {"semantic", "coarse", "fine"}:
            raise ValueError(f"Unknown Bark training phase {phase!r}.")
        prepared = self._phase_inputs(inputs, prefix=phase)
        missing = [name for name in ("input_ids", "labels") if name not in prepared]
        if phase == "fine" and "codebook_idx" not in prepared:
            missing.append("codebook_idx")
        if missing:
            expected_prefix = (
                f"{phase}_input_ids/{phase}_labels"
                if phase != "fine" else "fine_input_ids/fine_labels/codebook_idx")
            raise ValueError(
                f"Bark {phase!r} fine-tuning requires {expected_prefix}; "
                f"missing: {', '.join(missing)}.")
        return prepared

    def _generate(
        self,
        text: str,
        *,
        voice_preset: str | Mapping[str, Any] | None = None,
        output_file: str | Path | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        processor_options = {}
        if voice_preset is not None:
            processor_options["voice_preset"] = voice_preset
        inputs = self._processor_inputs(text, **processor_options)

        if "return_output_lengths" in generation_options:
            raise ValueError("`return_output_lengths` is managed by VoiceHub and cannot "
                             "be overridden.")
        if temperature is not None:
            generation_options["temperature"] = temperature
        if top_p is not None:
            generation_options["top_p"] = top_p
        if max_new_tokens is not None:
            generation_options["semantic_max_new_tokens"] = max_new_tokens
        generation_options["return_output_lengths"] = True

        with seeded_inference(
                seed,
                device=self.device,
                model_type=self.config.model_type,
        ) as effective_seed:
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    **generation_options,
                )
        if not isinstance(generated, tuple) or len(generated) < 2:
            raise RuntimeError("Bark did not return the waveform lengths requested by "
                               "VoiceHub.")
        waveform = self._normalize_waveform(
            generated[0],
            output_length=generated[1],
        )
        preset_name = (
            voice_preset if isinstance(voice_preset, str) else "custom" if voice_preset is not None else None)
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": "transformers",
                "voice_preset": preset_name,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self._save_native_bundle(save_directory)


BarkTTS = BarkForTextToSpeech

__all__ = [
    "BarkConfig",
    "BarkForTextToSpeech",
    "BarkTTS",
]
