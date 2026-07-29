"""Fine-tuning adapter for VoiceHub-native Nemotron 3.5 ASR."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.training.adapters import RNNTTrainingAdapter


class NativeNemotronASRTrainingAdapter(RNNTTrainingAdapter):
    """Train the exact prompt-conditioned FastConformer/RNN-T graph."""

    supports_custom_recipe = True
    native_export_semantics = ("voicehub-native-nemotron-rnnt-safetensors-and-processor")
    runtime_name = "Nemotron 3.5 ASR"
    checkpoint_format = "voicehub-nemotron-3.5-rnnt-v1"
    native_architecture_family = "nemotron-3.5-rnnt"

    def setup(self) -> NativeNemotronASRTrainingAdapter:
        super().setup()
        if getattr(self.model, "architecture_family", None) != "rnnt":
            raise ValueError("Native Nemotron fine-tuning requires the RNN-T runtime.")
        if self.primary_model is not getattr(self.model, "model", None):
            raise ValueError("Native Nemotron fine-tuning must target the wrapper's "
                             "exact model graph.")
        runtime = getattr(self.model, "runtime", None)
        if (runtime is not None and getattr(runtime, "model", None) is not self.primary_model):
            raise ValueError("Nemotron wrapper and runtime refer to different graphs.")
        return self

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: Any,
    ) -> Mapping[str, Any]:
        import torch

        required = {
            "input_features",
            "attention_mask",
            "prompt_ids",
            "labels",
            "label_lengths",
            "decoder_input_ids",
        }
        supplied = required & set(inputs)
        if supplied and supplied != required:
            missing = ", ".join(sorted(required - set(inputs)))
            raise ValueError(
                "A preprocessed Nemotron batch must provide all RNN-T "
                f"tensors; missing {missing}.")
        prepared = self.model.prepare_training_inputs(
            dict(inputs),
            phase=context.phase.name,
        )
        missing = tuple(sorted(required - set(prepared)))
        if missing:
            raise ValueError(
                "Nemotron training input preparation is incomplete; "
                f"missing {', '.join(missing)}.")
        unbatched_ranks = {
            "input_features": 2,
            "attention_mask": 1,
            "prompt_ids": 0,
            "labels": 1,
            "label_lengths": 0,
            "decoder_input_ids": 1,
        }
        for name, rank in unbatched_ranks.items():
            value = prepared[name]
            if isinstance(value, torch.Tensor) and value.ndim == rank:
                prepared[name] = value.unsqueeze(0)
        prepared.setdefault("num_lookahead_tokens", None)
        prepared["use_cache"] = False
        accepted = {
            "attention_mask",
            "decoder_input_ids",
            "input_features",
            "label_lengths",
            "labels",
            "num_lookahead_tokens",
            "prompt_ids",
            "use_cache",
        }
        return {name: value for name, value in prepared.items() if name in accepted and value is not None}

    def recipe_resume_configuration(self) -> Mapping[str, Any]:
        configuration = dict(super().recipe_resume_configuration())
        native_config = getattr(self.model, "native_config", None)
        encoder = getattr(native_config, "encoder_config", None)
        configuration.update({
            "checkpoint_format":
            self.checkpoint_format,
            "label_policy":
            "blank-padded-rnnt-targets",
            "model_blank_token_id":
            getattr(
                native_config,
                "blank_token_id",
                13_087,
            ),
            "objective":
            "exact-rnnt-forward-backward",
            "sample_rate":
            16_000,
            "supported_num_lookahead_tokens":
            list(getattr(
                encoder,
                "supported_num_lookahead_tokens",
                (3, 0, 6, 13),
            )),
            "trainable_scope":
            "full-model",
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": self.checkpoint_format,
            "export_scope": "full-model-processor-tokenizer-generation-config",
            "label_policy": "blank-padded-rnnt-targets",
            "native_architecture_family": self.native_architecture_family,
            "native_objective": "exact-rnnt-forward-backward",
            "processor_runtime": "voicehub-native-nemotron",
            "trainable_scope": "full-model",
        })
        return manifest

    def execute_training_phase(self, context: Any):
        output = super().execute_training_phase(context)
        output.metadata.update({
            "native_architecture_family": self.native_architecture_family,
            "native_objective": "exact-rnnt-forward-backward",
            "trainable_scope": "full-model",
        })
        return output

    def save_pretrained(
        self,
        save_directory: str | Path,
    ) -> None:
        self.setup()
        destination = Path(save_directory).expanduser()
        self.model.export_native_pretrained(destination)


__all__ = ["NativeNemotronASRTrainingAdapter"]
