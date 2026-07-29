"""Fine-tuning adapter for VoiceHub-native WeNet U2++."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeWeNetU2PPTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train and export the exact CTC/dual-decoder graph."""

    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-wenet-u2pp-safetensors"

    def setup(self) -> NativeWeNetU2PPTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "speech-seq2seq":
            raise ValueError("Native WeNet requires the speech-seq2seq runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native WeNet fine-tuning must target the wrapper's exact "
                             "`model` graph.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        import torch

        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        signal = prepared.get("input_signal")
        if isinstance(signal, torch.Tensor) and signal.ndim == 1:
            prepared["input_signal"] = signal.unsqueeze(0)
            for name in ("input_signal_length", "labels", "label_lengths"):
                value = prepared.get(name)
                if not isinstance(value, torch.Tensor):
                    continue
                if name == "labels" and value.ndim == 1:
                    prepared[name] = value.unsqueeze(0)
                elif name != "labels" and value.ndim == 0:
                    prepared[name] = value.unsqueeze(0)
        accepted = {
            "features",
            "feature_lengths",
            "input_signal",
            "input_signal_length",
            "labels",
            "label_lengths",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        values = dict(super().recipe_resume_configuration())
        native_config = getattr(self.model, "native_config", None)
        values.update({
            "checkpoint_format": "voicehub-wenet-gigaspeech-u2pp-v1",
            "ctc_weight": getattr(native_config, "ctc_weight", 0.3),
            "gradient_clip_norm": getattr(
                native_config,
                "gradient_clip_norm",
                5.0,
            ),
            "label_smoothing": getattr(
                native_config,
                "label_smoothing",
                0.1,
            ),
            "learning_rate": getattr(native_config, "learning_rate", 0.001),
            "optimizer": getattr(native_config, "optimizer", "adam"),
            "reverse_weight": getattr(native_config, "reverse_weight", 0.3),
            "sample_rate": getattr(native_config, "sampling_rate", 16_000),
            "scheduler": "warmuplr",
            "warmup_steps": getattr(native_config, "warmup_steps", 80_000),
        })
        return values

    def create_optimizer(
        self,
        name: str,
        parameters: list[tuple[str, Any]],
        training_args: Any,
    ):
        import torch

        if name not in {"default", "model"}:
            raise ValueError("Native WeNet declares only the `model` optimizer, "
                             f"found {name!r}.")
        trainable = [parameter for _, parameter in parameters if parameter.requires_grad]
        if not trainable:
            raise ValueError("Native WeNet has no trainable parameters.")
        return torch.optim.Adam(
            trainable,
            lr=training_args.learning_rate,
            betas=(
                training_args.adam_beta1,
                training_args.adam_beta2,
            ),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )

    def create_scheduler(
        self,
        name: str,
        optimizer: Any,
        num_training_steps: int,
        training_args: Any,
    ):
        import torch

        del num_training_steps
        if name not in {"default", "model"}:
            raise ValueError("Native WeNet declares only the `model` scheduler, "
                             f"found {name!r}.")
        native_config = getattr(self.model, "native_config", None)
        warmup_steps = (
            training_args.warmup_steps if training_args.warmup_steps > 0 else int(
                getattr(native_config, "warmup_steps", 80_000)))
        if warmup_steps <= 0:
            raise ValueError("Native WeNet WarmupLR requires warmup steps.")
        scale = warmup_steps**0.5

        def schedule(current_step: int) -> float:
            step = current_step + 1
            return scale * min(
                step**-0.5,
                step * warmup_steps**-1.5,
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-wenet-gigaspeech-u2pp-v1",
            "native_architecture_family": "wenet-gigaspeech-u2pp",
            "processor_runtime": "voicehub-native",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "wenet-gigaspeech-u2pp",
            "native_objective": "hybrid-ctc-bidirectional-attention",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["NativeWeNetU2PPTrainingAdapter"]
