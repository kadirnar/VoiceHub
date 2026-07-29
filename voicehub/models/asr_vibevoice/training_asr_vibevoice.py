"""Fine-tuning adapter for VoiceHub-native VibeVoice ASR."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import SpeechSeq2SeqTrainingAdapter


class NativeVibeVoiceASRTrainingAdapter(SpeechSeq2SeqTrainingAdapter):
    """Train the complete author-permitted ASR graph and export safely."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-vibevoice-asr-safetensors-tokenizer-and-processor")

    def setup(self) -> NativeVibeVoiceASRTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "causal-multimodal-lm":
            raise ValueError(
                "Native VibeVoice ASR fine-tuning requires the causal "
                "multimodal LM runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError(
                "Native VibeVoice ASR fine-tuning must target the wrapper's "
                "exact model graph.")
        runtime = getattr(self.model, "runtime", None)
        if (runtime is not None and getattr(runtime, "model", None) is not self.primary_model):
            raise ValueError("VibeVoice ASR wrapper and runtime refer to different graphs.")
        speech_encoders = (
            self.primary_model.model.acoustic_tokenizer_encoder,
            self.primary_model.model.semantic_tokenizer_encoder,
        )
        for encoder in speech_encoders:
            encoder.eval()
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise ValueError("Published VibeVoice ASR training keeps both speech "
                                 "encoders frozen.")
        return self

    def train(self, mode: bool = True) -> NativeVibeVoiceASRTrainingAdapter:
        super().train(mode)
        if mode:
            for encoder in (
                    self.primary_model.model.acoustic_tokenizer_encoder,
                    self.primary_model.model.semantic_tokenizer_encoder,
            ):
                encoder.eval()
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
        ranks = {
            "input_ids": 1,
            "attention_mask": 1,
            "input_values": 2,
            "padding_mask": 1,
            "labels": 1,
        }
        for name, rank in ranks.items():
            value = prepared.get(name)
            if isinstance(value, torch.Tensor) and value.ndim == rank:
                prepared[name] = value.unsqueeze(0)
        required = set(ranks)
        missing = required - set(prepared)
        if missing:
            raise ValueError(
                "VibeVoice ASR training inputs are incomplete; missing " + ", ".join(sorted(missing)) + ".")
        prepared["use_cache"] = False
        accepted = {
            *required,
            "generator",
            "logits_to_keep",
            "use_cache",
        }
        return {name: value for name, value in prepared.items() if name in accepted}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "checkpoint_format": "native-vibevoice-asr-v1",
            "label_policy": "assistant-completion-only",
            "objective": "shifted-causal-cross-entropy",
            "sample_rate": 24_000,
            "speech_encoder_policy": "author-frozen-no-grad",
            "trainable_scope": "multimodal-projector-language-model-lm-head",
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "native-vibevoice-asr-v1",
            "native_architecture_family": "vibevoice-asr",
            "processor_runtime": "voicehub-native",
            "tokenizer_runtime": "voicehub-byte-bpe",
            "trainable_scope": "multimodal-projector-language-model-lm-head",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": "vibevoice-asr",
            "native_objective": "shifted-causal-cross-entropy",
            "speech_encoder_policy": "author-frozen-no-grad",
            "trainable_scope": "multimodal-projector-language-model-lm-head",
        })
        return output

    def save_pretrained(self, save_directory: str | Path) -> None:
        self.setup()
        self.model.export_native_pretrained(Path(save_directory).expanduser())


__all__ = ["NativeVibeVoiceASRTrainingAdapter"]
