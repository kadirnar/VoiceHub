"""Source-faithful GPT-only fine-tuning for native XTTS v2."""

from __future__ import annotations

from pathlib import Path

from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import CompositeTrainingAdapter


class XTTSTrainingAdapter(CompositeTrainingAdapter):
    supports_custom_recipe = True
    native_export_semantics = "voicehub-native-xtts2-safetensors"

    def setup(self):
        super().setup()
        runtime = getattr(self.model, "native_runtime", None)
        if runtime is None:
            raise ValueError("XTTS fine-tuning requires the native runtime.")
        for parameter in runtime.parameters():
            parameter.requires_grad_(False)
        for parameter in runtime.gpt.parameters():
            parameter.requires_grad_(True)
        runtime.hifigan_decoder.eval()
        self.primary_model = runtime.gpt
        return self

    def prepare_training_inputs(self, inputs, context):
        del context
        required = {
            "text_inputs",
            "text_lengths",
            "audio_codes",
            "wav_lengths",
        }
        if not required <= set(inputs):
            raise ValueError(
                "Native XTTS fine-tuning requires precomputed text/audio tokens; "
                "the frozen legacy DVAE conversion is an offline data boundary.", )
        allowed = required | {"cond_mels", "cond_idxs", "cond_lens", "cond_latents"}
        return {name: value for name, value in inputs.items() if name in allowed}

    def execute_training_phase(self, context):
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        raw_text, raw_mel, logits = self.primary_model(**prepared)
        text_loss = raw_text * self.model.config.training_text_loss_weight
        mel_loss = raw_mel * self.model.config.training_mel_loss_weight
        loss = text_loss + mel_loss
        return TTSTrainingOutput(
            loss=loss,
            logits=logits,
            losses={
                "loss": loss,
                "loss_text_ce": text_loss,
                "loss_mel_ce": mel_loss,
                "raw_text_ce": raw_text,
                "raw_mel_ce": raw_mel,
            },
            metadata={
                "backend": "voicehub-native",
                "objective": "source-text-ce-plus-acoustic-token-ce",
                "gpt_trainable": True,
                "speaker_encoder_frozen": True,
                "vocoder_frozen": True,
                "dvae_boundary": "offline-precomputed-audio-codes",
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(save_directory)


__all__ = ["XTTSTrainingAdapter"]
