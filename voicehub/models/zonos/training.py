"""Delayed-codebook causal fine-tuning for the released Zonos graph."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import CausalLMTrainingAdapter
from voicehub.training.contracts import TrainingContext


class ZonosTrainingAdapter(CausalLMTrainingAdapter):
    """Train Zonos from source-shaped conditioning and DAC code tensors."""

    supports_custom_recipe = True
    native_export_semantics = "source-compatible-checkpoint-weight-warm-start"

    def setup(self):
        super().setup()
        autoencoder = getattr(self.primary_model, "autoencoder", None)
        if autoencoder is not None:
            codec = getattr(autoencoder, "dac", autoencoder)
            if hasattr(codec, "eval"):
                codec.eval()
            if hasattr(codec, "parameters"):
                for parameter in codec.parameters():
                    parameter.requires_grad_(False)
        self.primary_model.train()
        return self

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        torch = import_optional(
            "torch",
            model_type="zonos",
            install_extra="training",
        )
        functional = import_optional(
            "torch.nn.functional",
            model_type="zonos",
            install_extra="training",
        )
        batch = dict(context.inputs)
        prefix = batch.get("prefix_conditioning")
        audio_codes = batch.get("audio_codes")
        if not torch.is_tensor(prefix) or prefix.ndim != 3:
            raise ValueError(
                "Zonos `prefix_conditioning` must have shape "
                "[batch, prefix_time, hidden_size].")
        if not torch.is_tensor(audio_codes) or audio_codes.ndim != 3:
            raise ValueError("Zonos `audio_codes` must have shape "
                             "[batch, codebook, audio_time].")

        prefix = prefix.to(
            device=self.primary_model.device,
            dtype=next(self.primary_model.parameters()).dtype,
        )
        audio_codes = audio_codes.to(
            device=self.primary_model.device,
            dtype=torch.long,
        )
        code_lengths = batch.get("audio_code_lengths")
        if code_lengths is not None:
            if (not torch.is_tensor(code_lengths) or code_lengths.shape != (audio_codes.shape[0], )):
                raise ValueError("Zonos `audio_code_lengths` must have shape [batch].")
            code_lengths = code_lengths.to(
                device=audio_codes.device,
                dtype=torch.long,
            )

        logits, targets = self.primary_model.teacher_forced_logits(
            prefix,
            audio_codes,
            audio_code_lengths=code_lengths,
        )
        valid_targets = targets.ne(int(self.primary_model.masked_token_id))

        if not bool(valid_targets.any()):
            raise ValueError("Zonos training batch contains no supervised codec token.")
        labels = targets.masked_fill(~valid_targets, -100)
        codec_loss = functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return TTSTrainingOutput(
            loss=codec_loss,
            logits=logits,
            losses={
                "loss": codec_loss,
                "codec_ce_loss": codec_loss,
            },
            metadata={
                "model_type": "zonos",
                "objective": "delayed-codebook-causal-ce",
                "supervised_tokens": int(valid_targets.sum().item()),
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory) -> None:
        """Write the ``config.json`` and safetensors used by ``from_local``."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        safetensors = import_optional(
            "safetensors.torch",
            model_type="zonos",
            install_extra="training",
        )
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.primary_model.state_dict().items()
        }
        safetensors.save_file(state, destination / "model.safetensors")
        (destination / "config.json").write_text(
            json.dumps(
                asdict(self.primary_model.config),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )


__all__ = ["ZonosTrainingAdapter"]
