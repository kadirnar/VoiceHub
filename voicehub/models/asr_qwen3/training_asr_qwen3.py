"""Fine-tuning adapter for VoiceHub's native Qwen3-ASR runtime."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeQwen3ASRTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train and export Qwen3-ASR without an upstream model runtime."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-qwen3-asr-safetensors-tokenizer-and-processor")

    def _lora_configuration(self) -> dict[str, Any] | None:
        injection = getattr(self.model, "_lora_injection", None)
        if injection is None:
            return None
        config = injection.config
        return {
            "alpha": config.alpha,
            "dropout": config.dropout,
            "freeze_base": config.freeze_base,
            "rank": config.rank,
            "seed": config.seed,
            "target_modules": list(config.target_modules),
        }

    def setup(self) -> NativeQwen3ASRTrainingAdapter:
        super().setup()
        if (getattr(self.model, "architecture_family", None) != "speech-seq2seq"):
            raise ValueError("Native Qwen3-ASR fine-tuning requires the "
                             "speech-seq2seq runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native Qwen3-ASR fine-tuning must target the wrapper's "
                "exact `model` graph.")
        return self

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
            "attention_mask",
            "audio_feature_lengths",
            "feature_attention_mask",
            "ignore_index",
            "input_features",
            "input_ids",
            "label_smoothing",
            "labels",
            "output_attentions",
            "output_hidden_states",
            "position_ids",
            "use_cache",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "native-qwen3-asr-v1",
            "label_policy": "assistant-completion-only",
            "objective": "causal-language-modeling",
            "sample_rate": 16_000,
            "lora": self._lora_configuration(),
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "native-qwen3-asr-v1",
            "native_architecture_family": "qwen3-asr",
            "processor_runtime": "voicehub-native",
            "tokenizer_runtime": "voicehub-byte-bpe",
            "lora": self._lora_configuration(),
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "qwen3-asr",
            "native_objective": "causal-language-modeling",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self.model.export_native_pretrained(destination)


__all__ = ["NativeQwen3ASRTrainingAdapter"]
