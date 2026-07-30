"""Built-in source-native fine-tuning recipes.

These adapters preserve objectives published by the model authors while
using VoiceHub's common optimization, checkpoint, and strategy layers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import BaseTrainingAdapter, CausalLMTrainingAdapter, FlowMatchingTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.ema import ExponentialMovingAverage


class SourceRecipeTrainingAdapter(BaseTrainingAdapter):
    """Base class for model-author training recipes integrated by VoiceHub."""

    supports_custom_recipe = True

    def _training_output(
        self,
        context: TrainingContext,
        *,
        loss,
        losses: Mapping[str, Any] | None = None,
        logits=None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TTSTrainingOutput:
        output_metadata = {
            "model_type": self.model_type,
            "training_family": self.spec.family_name,
            "training_support": self.spec.support.value,
            "training_phase": context.phase.name,
            "optimizer_names": context.phase.optimizer_names,
            "source_native_recipe": True,
        }
        output_metadata.update(dict(metadata or {}))
        return TTSTrainingOutput(
            loss=loss,
            logits=logits,
            losses=dict(losses or {"loss": loss}),
            metadata=output_metadata,
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )


class CodecCausalLMTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Causal codec LMs with frozen audio tokenizers and native HF loss."""

    supports_custom_recipe = True

    def setup(self):
        super().setup()
        codec = getattr(self.model, "codec", None)
        if codec is None:
            runtime = getattr(self.model, "model", None)
            codec = getattr(runtime, "codec", None)
            if codec is None:
                codec = getattr(runtime, "audio_codec", None)
        if codec is not None:
            if hasattr(codec, "eval"):
                codec.eval()
            if hasattr(codec, "parameters"):
                for parameter in codec.parameters():
                    parameter.requires_grad_(False)
        return self

    def create_dataset(self, records, **kwargs):
        self.setup()
        if self.model_type == "orpheustts":
            training = import_optional(
                "voicehub.models.orpheustts.training",
                model_type=self.model_type,
                install_extra=self.spec.install_extra,
            )
            return training.OrpheusSFTDataset(
                records,
                tokenizer=self.model.tokenizer,
                codec=self.model.codec,
                completion_only=bool(kwargs.get("completion_only", False)),
                max_length=kwargs.get("max_length"),
            )
        if self.model_type == "llasa":
            training = import_optional(
                "voicehub.models.llasa.training",
                model_type=self.model_type,
                install_extra=self.spec.install_extra,
            )
            return training.LlasaSFTDataset(
                records,
                tokenizer=self.model.tokenizer,
                codec=self.model.codec,
                sample_rate=self.model.sample_rate,
                max_length=int(kwargs.get("max_length", 2048)),
            )
        if self.model_type == "neutts":
            training = import_optional(
                "voicehub.models.neutts.training",
                model_type=self.model_type,
                install_extra=self.spec.install_extra,
            )
            return training.NeuTTSSFTDataset(
                records,
                runtime=self.model.model,
                max_length=int(kwargs.get("max_length", 2048)),
            )
        if self.model_type == "outetts":
            training = import_optional(
                "voicehub.models.outetts.training",
                model_type=self.model_type,
                install_extra=self.spec.install_extra,
            )
            return training.OuteTTSSFTDataset(
                records,
                interface=self.model.model,
                completion_only=bool(kwargs.get("completion_only", True)),
                whisper_model=str(kwargs.get("whisper_model", "turbo")),
                whisper_device=kwargs.get("whisper_device"),
            )
        return super().create_dataset(records, **kwargs)

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        destination = Path(save_directory)
        if hasattr(self.primary_model, "save_pretrained"):
            self.primary_model.save_pretrained(
                destination,
                safe_serialization=True,
            )
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = getattr(getattr(self.model, "model", None), "tokenizer", None)
        if tokenizer is None and self.model_type == "outetts":
            tokenizer = self.model.model.prompt_processor.tokenizer
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(destination)


class OrpheusTrainingAdapter(CodecCausalLMTrainingAdapter):
    """Fine-tune and export the complete VoiceHub-native Orpheus runtime."""

    native_export_semantics = "inference-export"

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native Orpheus training requires a wrapper with "
                            "export_native_pretrained().")
        export(save_directory)


class ConversationTTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Published two-level codec language-model objective and export."""

    native_export_semantics = "inference-reloadable-safetensors"

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        if not isinstance(prepared, Mapping):
            raise TypeError("ConversationTTS input preparation must return a mapping.")
        prepared = dict(prepared)
        required = ("tokens", "labels", "tokens_mask")
        missing = [name for name in required if name not in prepared]
        if missing:
            raise ValueError("ConversationTTS fine-tuning requires: " + ", ".join(missing))
        model = self.primary_model
        c0_logits, residual_logits, residual_labels = model(
            tokens=prepared["tokens"],
            labels=prepared["labels"],
            tokens_mask=prepared["tokens_mask"],
            input_pos=prepared.get("input_pos"),
        )
        source = import_optional(
            "voicehub.models.conversationtts.source.conversationtts."
            "models.model_new",
            model_type="conversationtts",
            install_extra="training",
        )
        labels = prepared["labels"]
        zero_labels = labels[..., 0]
        zero_mask = prepared.get("loss_mask")
        if zero_mask is None:
            zero_mask = prepared["tokens_mask"][:, 1:, 0].bool()
        zero_length = min(c0_logits.shape[1], zero_labels.shape[1], zero_mask.shape[1])
        default_padding_id = int(getattr(
            getattr(model, "config", None),
            "audio_vocab_size",
            2_051,
        )) - 1
        loss_zero, zero_metrics = source.CrossEntropyAndAccuracy_zero(
            c0_logits[:, :zero_length],
            zero_labels[:, :zero_length],
            zero_mask[:, :zero_length],
            ignore_id=int(prepared.get("ignore_id", default_padding_id)),
        )
        residual_weights = prepared.get("residual_loss_weights")
        if residual_weights is None:
            residual_weights = [1.0] * int(residual_labels.shape[-1])
        loss_residual, residual_metrics = source.CrossEntropyAndAccuracy_residual(
            residual_logits,
            residual_labels,
            loss_weights=residual_weights,
            ignore_id=int(prepared.get(
                "residual_ignore_id",
                default_padding_id,
            )),
        )
        loss = loss_zero + loss_residual
        metrics = {
            **zero_metrics,
            **residual_metrics,
        }
        return self._training_output(
            context,
            loss=loss,
            losses={
                "loss": loss,
                "codebook0_loss": loss_zero,
                "residual_loss": loss_residual,
            },
            logits=(c0_logits, residual_logits),
            metadata={"metrics": metrics},
        )

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError(
                "Native ConversationTTS training requires a wrapper with "
                "`export_native_pretrained()`.")
        export(save_directory)

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        manifest.update({
            "checkpoint_format": "voicehub-conversationtts-v1",
            "native_architecture_family": "conversationtts",
            "objective": "published-two-level-masked-codebook-cross-entropy",
            "objective_author_verified": True,
            "training_scope": "full-acoustic-language-model",
            "frozen_components": ["mimi-codec"],
            "raw_audio_preprocessing": "optional-frozen-native-mimi",
            "inference_reloadable": True,
            "commercial_use": False,
        })
        return manifest


class F5TTSTrainingAdapter(
        FlowMatchingTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Native conditional-flow objective with update-coupled EMA."""

    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._ema: ExponentialMovingAverage | None = None

    def _use_ema(self) -> bool:
        config = getattr(self.model, "config", None)
        enabled = getattr(config, "use_ema", True)
        if not isinstance(enabled, bool):
            raise TypeError("F5-TTS `use_ema` must be a boolean.")
        return enabled

    def setup(self):
        super().setup()
        if not self._use_ema():
            self._ema = None
        elif self._ema is None:
            config = getattr(self.model, "config", None)
            self._ema = ExponentialMovingAverage(
                self.primary_model,
                decay=float(getattr(config, "ema_decay", 0.9999)),
                update_after_step=int(getattr(config, "ema_update_after_step", 0)),
                update_every=int(getattr(config, "ema_update_every", 1)),
            )
        return self

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        config = getattr(self.model, "config", None)
        configuration.update({
            "resolved_use_ema":
            self._use_ema(),
            "resolved_ema_decay":
            float(getattr(config, "ema_decay", 0.9999), ),
            "resolved_ema_update_after_step":
            int(getattr(config, "ema_update_after_step", 0), ),
            "resolved_ema_update_every":
            int(getattr(config, "ema_update_every", 1), ),
        })
        return configuration

    def prepare_training_inputs(
        self,
        inputs: Mapping[str, Any],
        context: TrainingContext,
    ) -> Mapping[str, Any]:
        del context
        prepared = dict(inputs)
        aliases = {
            "mel": "inp",
            "mel_spec": "inp",
            "input_values": "inp",
            "input_ids": "text",
            "mel_lengths": "lens",
            "lengths": "lens",
        }
        for source, target in aliases.items():
            if source in prepared and target not in prepared:
                value = prepared.pop(source)
                if (source in ("mel", "mel_spec") and getattr(value, "ndim", None) == 3):
                    value = value.permute(0, 2, 1)
                prepared[target] = value
        allowed = ("inp", "text", "lens", "noise_scheduler")
        return {name: prepared[name] for name in allowed if name in prepared}

    def on_optimizer_step(
        self,
        *,
        optimizer_names: tuple[str, ...] | None,
        step: int,
    ) -> None:
        del optimizer_names
        self.setup()
        if self._ema is not None:
            self._ema.update(step=step)

    def recipe_state_dict(self) -> Mapping[str, Any]:
        self.setup()
        if self._ema is None:
            return {}
        return {"ema": self._ema.state_dict()}

    def load_recipe_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        self.setup()
        if not isinstance(state_dict, Mapping):
            raise TypeError("F5-TTS recipe state must be a mapping.")
        if self._ema is None:
            if strict and state_dict:
                raise ValueError("F5-TTS recipe state cannot contain EMA data when "
                                 "`use_ema=False`.")
            return
        if not state_dict:
            return
        if strict and set(state_dict) != {"ema"}:
            raise ValueError("F5-TTS recipe state must contain only 'ema'.")
        if "ema" in state_dict:
            self._ema.load_state_dict(state_dict["ema"], strict=strict)

    def save_pretrained(self, save_directory) -> None:
        """Export EMA weights when enabled, otherwise explicit raw weights."""
        self.setup()
        from voicehub.architectures.f5tts.checkpoint import export_f5tts_checkpoint

        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        state = self.primary_model.state_dict()
        prefix = ""
        export_state = state
        if self._ema is not None:
            ema_state = self._ema.state_dict()["shadow"]
            export_state = {name: ema_state.get(name, value) for name, value in state.items()}
            prefix = "ema_model."
        export_f5tts_checkpoint(
            self.primary_model,
            destination / "model.safetensors",
            prefix=prefix,
            state_override=export_state,
        )
        runtime = getattr(self.model, "model", None)
        frontend = getattr(runtime, "frontend", None)
        vocabulary = getattr(frontend, "vocabulary", None)
        if vocabulary is None or not callable(getattr(vocabulary, "save", None)):
            raise TypeError("Native F5-TTS export requires the loaded vocabulary.")
        vocabulary.save(destination / "vocab.txt")
        self.model.config.save_pretrained(destination)


class Qwen3TTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Official Qwen3-TTS 12 Hz single-speaker SFT objective."""

    native_export_semantics = "inference-export"
    _DEFAULT_LORA_TARGETS = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._target_speaker_embedding = None
        self._lora_injection = None

    def _configured_lora_targets(self) -> tuple[str, ...]:
        return tuple(
            getattr(
                self.model.config,
                "training_lora_target_modules",
                self._DEFAULT_LORA_TARGETS,
            ))

    def setup(self):
        if self.is_ready:
            return super().setup()
        super().setup()
        model_type = str(getattr(self.primary_model.config, "tts_model_type", "")).lower()
        speaker_encoder = getattr(self.primary_model, "speaker_encoder", None)
        if model_type != "base" or speaker_encoder is None:
            raise ValueError(
                "Qwen3-TTS fine-tuning requires a 12 Hz Base checkpoint with "
                "its speaker encoder. CustomVoice and VoiceDesign artifacts "
                "are inference/export targets, not valid SFT starting points.")
        for parameter in speaker_encoder.parameters():
            parameter.requires_grad_(False)
        lora_rank = getattr(self.model.config, "training_lora_rank", None)
        if lora_rank is not None:
            from voicehub.models.qwen3tts.lora import inject_qwen3_tts_lora

            self._lora_injection = inject_qwen3_tts_lora(
                self.primary_model,
                rank=lora_rank,
                alpha=getattr(
                    self.model.config,
                    "training_lora_alpha",
                    16.0,
                ),
                dropout=getattr(
                    self.model.config,
                    "training_lora_dropout",
                    0.0,
                ),
                target_modules=self._configured_lora_targets(),
                seed=getattr(
                    self.model.config,
                    "training_lora_seed",
                    0,
                ),
            )
        return self

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        configuration["resolved_sub_talker_loss_weight"] = float(
            getattr(self.model.config, "sub_talker_loss_weight", 0.3), )
        lora_rank = getattr(self.model.config, "training_lora_rank", None)
        configuration.update({
            "parameter_efficient": lora_rank is not None,
            "lora_topology": None if lora_rank is None else {
                "adapter_library":
                "voicehub-native",
                "base_model_frozen":
                True,
                "speaker_encoder_frozen":
                True,
                "decoder_stacks": (
                    "talker",
                    "residual-code-predictor",
                ),
                "injected_module_names":
                ([] if self._lora_injection is None else list(self._lora_injection.module_names)),
            },
        })
        return configuration

    def artifact_manifest(self) -> dict[str, Any]:
        manifest = super().artifact_manifest()
        if getattr(self.model.config, "training_lora_rank", None) is not None:
            manifest["checkpoint_semantics"]["lora_adapter"] = ("strict-adapter-only-safetensors")
            manifest["checkpoint_semantics"]["save_pretrained"] = (
                "clone-merged-inference-export-plus-adapter")
        return manifest

    def create_dataset(self, records, **kwargs):
        del kwargs
        self.setup()
        training = import_optional(
            "voicehub.models.qwen3tts.training",
            model_type="qwen3tts",
            install_extra="training",
        )
        return training.Qwen3TTSSFTDataset(
            records,
            self.model.model.processor,
            self.primary_model.config,
        )

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        batch = dict(context.inputs)
        required = (
            "input_ids",
            "codec_ids",
            "ref_mels",
            "text_embedding_mask",
            "codec_embedding_mask",
            "attention_mask",
            "codec_0_labels",
            "codec_mask",
        )
        missing = [name for name in required if name not in batch]
        if missing:
            raise ValueError("Qwen3-TTS fine-tuning requires: " + ", ".join(missing))
        model = self.primary_model
        speaker_embedding = model.speaker_encoder(
            batch["ref_mels"].to(
                device=model.device,
                dtype=model.dtype,
            )).detach()
        if self._target_speaker_embedding is None:
            self._target_speaker_embedding = speaker_embedding[0].detach().clone()

        input_ids = batch["input_ids"]
        codec_ids = batch["codec_ids"]
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        text_embeddings = model.talker.get_text_embeddings()(input_text_ids)
        text_embeddings = (model.talker.text_projection(text_embeddings) * batch["text_embedding_mask"])
        codec_embeddings = (
            model.talker.model.codec_embedding(input_codec_ids) * batch["codec_embedding_mask"])
        codec_embeddings = codec_embeddings.clone()
        codec_embeddings[:, 6, :] = speaker_embedding
        input_embeddings = text_embeddings + codec_embeddings

        codec_mask = batch["codec_mask"]
        for channel_index in range(1, codec_ids.shape[-1]):
            channel_embedding = (
                model.talker.code_predictor.get_input_embeddings()[channel_index - 1](
                    codec_ids[:, :, channel_index]))
            input_embeddings = (input_embeddings + channel_embedding * codec_mask.unsqueeze(-1))

        outputs = model.talker(
            inputs_embeds=input_embeddings,
            attention_mask=batch["attention_mask"],
            labels=batch["codec_0_labels"],
            output_hidden_states=False,
        )
        hidden_states = outputs.last_hidden_state
        # The native causal loss uses hidden state t - 1 to predict target t.
        # Preserve that same pairing for the per-frame sub-talker objective.
        next_codec_mask = codec_mask[:, 1:]
        talker_hidden = hidden_states[:, :-1][next_codec_mask]
        talker_codec_ids = codec_ids[:, 1:][next_codec_mask]
        sub_logits, sub_loss = model.talker.forward_sub_talker_finetune(
            talker_codec_ids,
            talker_hidden,
        )
        sub_weight = float(getattr(self.model.config, "sub_talker_loss_weight", 0.3))
        if not math.isfinite(sub_weight) or sub_weight < 0:
            raise ValueError("Qwen3-TTS sub-talker loss weight must be finite and "
                             "non-negative.")
        loss = outputs.loss + sub_weight * sub_loss
        return self._training_output(
            context,
            loss=loss,
            losses={
                "loss": loss,
                "talker_loss": outputs.loss,
                "sub_talker_loss": sub_loss,
            },
            logits=(outputs.logits, sub_logits),
            metadata={
                "parameter_efficient": self._lora_injection is not None,
                "lora_adapter_library": (None if self._lora_injection is None else "voicehub-native"),
            },
        )

    def recipe_state_dict(self) -> Mapping[str, Any]:
        return {
            "target_speaker_embedding": (
                None if self._target_speaker_embedding is None else
                self._target_speaker_embedding.detach().clone())
        }

    def load_recipe_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if not state_dict:
            return
        if strict and set(state_dict) != {"target_speaker_embedding"}:
            raise ValueError("Qwen3-TTS recipe state must contain only "
                             "'target_speaker_embedding'.")
        embedding = state_dict.get("target_speaker_embedding")
        self._target_speaker_embedding = (None if embedding is None else embedding.detach().clone())

    def save_pretrained(self, save_directory) -> None:
        """Write a directly loadable Hugging Face safetensors directory."""
        self.setup()
        if self._target_speaker_embedding is None:
            return
        model = self.primary_model
        speaker_id = int(getattr(self.model.config, "training_speaker_id", 3000))
        speaker_name = str(getattr(self.model.config, "training_speaker_name", "voicehub"))
        embedding = model.talker.model.codec_embedding.weight
        if not 0 <= speaker_id < embedding.shape[0]:
            raise ValueError(
                f"Qwen3-TTS speaker id {speaker_id} is outside the codec "
                f"embedding table of size {embedding.shape[0]}.")
        talker_config = model.config.talker_config
        destination = Path(save_directory)
        if self._lora_injection is None:
            state_dict = {
                name: value
                for name, value in model.state_dict().items() if not name.startswith("speaker_encoder.")
            }
        else:
            from voicehub.models.qwen3tts.lora import merged_qwen3_tts_state_dict

            state_dict = merged_qwen3_tts_state_dict(
                model,
                self._lora_injection,
                drop_prefixes=("speaker_encoder.", ),
            )
        embedding_module = model.talker.model.codec_embedding
        embedding_key = next(
            (f"{name}.weight" for name, module in model.named_modules() if module is embedding_module),
            None,
        )
        if embedding_key is None or embedding_key not in state_dict:
            raise RuntimeError(
                "Qwen3-TTS export could not locate the talker codec "
                "embedding in the model state.")
        exported_embedding = state_dict[embedding_key].detach().clone()
        exported_embedding[speaker_id].copy_(
            self._target_speaker_embedding.to(
                device=exported_embedding.device,
                dtype=exported_embedding.dtype,
            ))
        state_dict[embedding_key] = exported_embedding

        original_speaker_ids = talker_config.spk_id
        original_dialects = talker_config.spk_is_dialect
        original_model_type = model.config.tts_model_type
        try:
            talker_config.spk_id = {speaker_name: speaker_id}
            talker_config.spk_is_dialect = {speaker_name: False}
            model.config.tts_model_type = "custom_voice"
            model.save_pretrained(
                destination,
                state_dict=state_dict,
                safe_serialization=True,
            )
        finally:
            talker_config.spk_id = original_speaker_ids
            talker_config.spk_is_dialect = original_dialects
            model.config.tts_model_type = original_model_type
        processor = getattr(self.model.model, "processor", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)
        speech_tokenizer = getattr(model, "speech_tokenizer", None)
        speech_model = getattr(speech_tokenizer, "model", None)
        feature_extractor = getattr(speech_tokenizer, "feature_extractor", None)
        speech_directory = destination / "speech_tokenizer"
        if speech_model is None or feature_extractor is None:
            raise ValueError(
                "Qwen3-TTS export requires the loaded speech tokenizer model "
                "and feature extractor.")
        speech_model.save_pretrained(
            speech_directory,
            safe_serialization=True,
        )
        feature_extractor.save_pretrained(speech_directory)
        if (self._lora_injection is not None and bool(getattr(
                self.model.config,
                "training_lora_export_adapter",
                True,
        ))):
            from voicehub.models.qwen3tts.lora import save_qwen3_tts_lora_adapter

            save_qwen3_tts_lora_adapter(
                self._lora_injection,
                destination / "lora_adapter",
                target_modules=self._configured_lora_targets(),
                base_model=(getattr(self.model.config, "name_or_path", None) or None),
                target_speaker_embedding=self._target_speaker_embedding,
                speaker_name=speaker_name,
                speaker_id=speaker_id,
            )

    def load_lora_adapter(self, directory):
        """Restore a strict adapter-only export into this training graph."""
        self.setup()
        if self._lora_injection is None:
            raise RuntimeError(
                "Enable Qwen3-TTS LoRA in the training configuration before "
                "loading an adapter-only checkpoint.")
        from voicehub.models.qwen3tts.lora import load_qwen3_tts_lora_adapter

        result = load_qwen3_tts_lora_adapter(
            self._lora_injection,
            directory,
            target_modules=self._configured_lora_targets(),
            expected_base_model=(getattr(self.model.config, "name_or_path", None) or None),
            expected_speaker_name=str(getattr(
                self.model.config,
                "training_speaker_name",
                "voicehub",
            )),
            expected_speaker_id=int(getattr(
                self.model.config,
                "training_speaker_id",
                3000,
            )),
            expected_speaker_embedding_size=int(
                self.primary_model.talker.model.codec_embedding.weight.shape[1]),
        )
        self._target_speaker_embedding = result.target_speaker_embedding
        return result


def _native_fish_s2_adapter(model, spec):
    # Keep the exact two-head objective model-local while registry discovery
    # remains dependency-light.
    from voicehub.models.fishtts.training import FishSpeechTrainingAdapter

    return FishSpeechTrainingAdapter(model, spec)


def _dia_adapter(model, spec):
    from voicehub.models.dia.training import DiaTrainingAdapter

    return DiaTrainingAdapter(model, spec)


def _cosyvoice_adapter(model, spec):
    from voicehub.models.cosyvoice.training import CosyVoiceTrainingAdapter

    return CosyVoiceTrainingAdapter(model, spec)


def _xtts_adapter(model, spec):
    from voicehub.models.xtts_native.training_xtts import XTTSTrainingAdapter

    return XTTSTrainingAdapter(model, spec)


def _csm_adapter(model, spec):
    # Keep the specialized two-level objective and frozen Mimi preprocessing
    # model-local while ordinary registry discovery remains dependency-light.
    from voicehub.models.csm.training import CSMTrainingAdapter

    return CSMTrainingAdapter(model, spec)


def _echo_adapter(model, spec):
    from voicehub.models.echo.training import EchoTrainingAdapter

    return EchoTrainingAdapter(model, spec)


def _chatterbox_adapter(model, spec):
    from voicehub.models.chatterbox.training import ChatterboxTrainingAdapter

    return ChatterboxTrainingAdapter(model, spec)


def _vui_adapter(model, spec):
    from voicehub.models.vui.training import VuiTrainingAdapter

    return VuiTrainingAdapter(model, spec)


def _zonos_adapter(model, spec):
    from voicehub.models.zonos.training import ZonosTrainingAdapter

    return ZonosTrainingAdapter(model, spec)


def _zonos2_adapter(model, spec):
    from voicehub.models.zonos2.training import Zonos2TrainingAdapter

    return Zonos2TrainingAdapter(model, spec)


def _vibevoice_adapter(model, spec):
    from voicehub.models.vibevoice.training import VibeVoiceTrainingAdapter

    return VibeVoiceTrainingAdapter(model, spec)


def _voxcpm_adapter(model, spec):
    from voicehub.models.voxcpm.training import VoxCPMTrainingAdapter

    return VoxCPMTrainingAdapter(model, spec)


def _omnivoice_adapter(model, spec):
    from voicehub.models.omnivoice.training import OmniVoiceTrainingAdapter

    return OmniVoiceTrainingAdapter(model, spec)


def _higgs_adapter(model, spec):
    from voicehub.models.higgstts.training import HiggsTrainingAdapter

    return HiggsTrainingAdapter(model, spec)


def _irodori_adapter(model, spec):
    from voicehub.models.irodoritts.training import NativeIrodoriTrainingAdapter

    return NativeIrodoriTrainingAdapter(model, spec)


def _vits_adapter(model, spec):
    # Keep the explicitly partial generator recipe model-local. The generic
    # VITS family adapter must continue to require an architecture-specific
    # implementation of a complete adversarial objective.
    from voicehub.models.vits.training import NativeVitsGeneratorTrainingAdapter

    return NativeVitsGeneratorTrainingAdapter(model, spec)


def _kokoro_adapter(model, spec):
    from voicehub.models.kokoro.training import KokoroTrainingAdapter

    return KokoroTrainingAdapter(model, spec)


def _parlertts_adapter(model, spec):
    from voicehub.models.parlertts.training import ParlerTTSTrainingAdapter

    return ParlerTTSTrainingAdapter(model, spec)


def _native_speecht5_adapter(model, spec):
    from voicehub.models.speecht5.training import NativeSpeechT5TrainingAdapter

    return NativeSpeechT5TrainingAdapter(model, spec)


def _supertonic_adapter(model, spec):
    from voicehub.models.supertonic.training import SupertonicTrainingAdapter

    return SupertonicTrainingAdapter(model, spec)


def _bark_adapter(model, spec):
    from voicehub.architectures.bark.training import BarkTrainingAdapter

    return BarkTrainingAdapter(model, spec)


def _inflecttts_adapter(model, spec):
    from voicehub.models.inflecttts.training import InflectTTSTrainingAdapter

    return InflectTTSTrainingAdapter(model, spec)


def _styletts2_adapter(model, spec):
    from voicehub.models.styletts2.training import StyleTTS2TrainingAdapter

    return StyleTTS2TrainingAdapter(model, spec)


def _melotts_adapter(model, spec):
    from voicehub.models.melotts.training import MeloTTSTrainingAdapter

    return MeloTTSTrainingAdapter(model, spec)


def _openvoice_adapter(model, spec):
    from voicehub.models.openvoice.training import OpenVoiceTrainingAdapter

    return OpenVoiceTrainingAdapter(model, spec)


def _gptsovits_adapter(model, spec):
    from voicehub.models.gptsovits.training import GPTSoVITSTrainingAdapter

    return GPTSoVITSTrainingAdapter(model, spec)


def _native_mosstts_adapter(model, spec):
    from voicehub.architectures.mosstts.training import NativeMossTTSTrainingAdapter

    return NativeMossTTSTrainingAdapter(model, spec)


def _neutts_adapter(model, spec):
    from voicehub.models.neutts.training import NeuTTSTrainingAdapter

    return NeuTTSTrainingAdapter(model, spec)


def _outetts_adapter(model, spec):
    from voicehub.models.outetts.training import OuteTTSTrainingAdapter

    return OuteTTSTrainingAdapter(model, spec)


def _transformers_asr_adapter(model, spec):
    # The historical name remains public API. The generic provider dispatches
    # only to verified VoiceHub-native graphs; dedicated providers temporarily
    # reuse this adapter until their architecture ports are complete.
    from voicehub.models.asr_transformers.training_asr_transformers import TransformersASRTrainingAdapter

    return TransformersASRTrainingAdapter(model, spec)


def _transformers_vad_adapter(model, spec):
    # Keep Transformers optional until this provider is selected.
    from voicehub.models.vad_transformers.training_vad_transformers import TransformersVADTrainingAdapter

    return TransformersVADTrainingAdapter(model, spec)


def _native_whisper_adapter(model, spec):
    from voicehub.models.asr_whisper_native.training_asr_whisper_native import NativeWhisperTrainingAdapter

    return NativeWhisperTrainingAdapter(model, spec)


def _native_wav2vec2_adapter(model, spec):
    from voicehub.models.asr_wav2vec2.training_asr_wav2vec2 import NativeWav2Vec2TrainingAdapter

    return NativeWav2Vec2TrainingAdapter(model, spec)


def _native_nemo_ctc_adapter(model, spec):
    from voicehub.models.asr_nemo.training_asr_nemo import NativeNeMoCTCTrainingAdapter

    return NativeNeMoCTCTrainingAdapter(model, spec)


def _native_wenet_u2pp_adapter(model, spec):
    from voicehub.models.asr_wenet.training_asr_wenet import NativeWeNetU2PPTrainingAdapter

    return NativeWeNetU2PPTrainingAdapter(model, spec)


def _native_espnet_adapter(model, spec):
    from voicehub.architectures.espnet_transformer.training import NativeESPnetASRTrainingAdapter

    return NativeESPnetASRTrainingAdapter(model, spec)


def _native_hubert_adapter(model, spec):
    from voicehub.models.asr_hubert.training_asr_hubert import NativeHubertTrainingAdapter

    return NativeHubertTrainingAdapter(model, spec)


def _native_wavlm_adapter(model, spec):
    from voicehub.models.asr_wavlm.training_asr_wavlm import NativeWavLMTrainingAdapter

    return NativeWavLMTrainingAdapter(model, spec)


def _native_moonshine_adapter(model, spec):
    from voicehub.models.asr_moonshine.training_asr_moonshine import NativeMoonshineTrainingAdapter

    return NativeMoonshineTrainingAdapter(model, spec)


def _native_qwen3_asr_adapter(model, spec):
    from voicehub.models.asr_qwen3.training_asr_qwen3 import NativeQwen3ASRTrainingAdapter

    return NativeQwen3ASRTrainingAdapter(model, spec)


def _native_granite_speech_adapter(model, spec):
    from voicehub.models.asr_granite_speech.training_asr_granite_speech import NativeGraniteSpeechTrainingAdapter

    return NativeGraniteSpeechTrainingAdapter(model, spec)


def _native_parakeet_tdt_adapter(model, spec):
    from voicehub.models.asr_parakeet_tdt.training_asr_parakeet_tdt import NativeParakeetTDTTrainingAdapter

    return NativeParakeetTDTTrainingAdapter(model, spec)


def _native_nemotron_asr_adapter(model, spec):
    from voicehub.models.asr_nemotron.training_asr_nemotron import NativeNemotronASRTrainingAdapter

    return NativeNemotronASRTrainingAdapter(model, spec)


def _native_cohere_asr_adapter(model, spec):
    from voicehub.models.asr_cohere.training_asr_cohere import NativeCohereASRTrainingAdapter

    return NativeCohereASRTrainingAdapter(model, spec)


def _native_seamless_m4t_v2_adapter(model, spec):
    from voicehub.models.asr_seamless_m4t_v2.training_asr_seamless_m4t_v2 import NativeSeamlessM4Tv2TrainingAdapter

    return NativeSeamlessM4Tv2TrainingAdapter(model, spec)


def _native_vibevoice_asr_adapter(model, spec):
    from voicehub.models.asr_vibevoice.training_asr_vibevoice import NativeVibeVoiceASRTrainingAdapter

    return NativeVibeVoiceASRTrainingAdapter(model, spec)


def _native_medasr_adapter(model, spec):
    from voicehub.models.asr_medasr.training_asr_medasr import NativeMedASRTrainingAdapter

    return NativeMedASRTrainingAdapter(model, spec)


def _native_sensevoice_adapter(model, spec):
    from voicehub.architectures.sensevoice.training import NativeSenseVoiceTrainingAdapter

    return NativeSenseVoiceTrainingAdapter(model, spec)


def _llasa_adapter(model, spec):
    from voicehub.models.llasa.training import LlasaTrainingAdapter

    return LlasaTrainingAdapter(model, spec)


def _native_silero_vad_adapter(model, spec):
    from voicehub.models.vad_silero.training_vad_silero import NativeSileroVADTrainingAdapter

    return NativeSileroVADTrainingAdapter(model, spec)


def _native_sherpa_vad_adapter(model, spec):
    from voicehub.models.vad_sherpa_onnx.training_vad_sherpa_onnx import create_sherpa_native_vad_training_adapter

    return create_sherpa_native_vad_training_adapter(model, spec)


def _native_pyannet_adapter(model, spec):
    from voicehub.models.vad_pyannote.training_vad_pyannote import NativePyanNetTrainingAdapter

    return NativePyanNetTrainingAdapter(model, spec)


def _native_fsmn_vad_adapter(model, spec):
    from voicehub.models.vad_funasr.training_vad_funasr import NativeFSMNVADTrainingAdapter

    return NativeFSMNVADTrainingAdapter(model, spec)


def _native_speechbrain_vad_adapter(model, spec):
    from voicehub.models.vad_speechbrain.training_vad_speechbrain import NativeSpeechBrainVADTrainingAdapter

    return NativeSpeechBrainVADTrainingAdapter(model, spec)


def _native_speechbrain_asr_adapter(model, spec):
    from voicehub.models.asr_native.speechbrain_training import NativeSpeechBrainASRTrainingAdapter

    return NativeSpeechBrainASRTrainingAdapter(model, spec)


def _native_marblenet_vad_adapter(model, spec):
    from voicehub.models.vad_nemo.training_vad_nemo import NativeMarbleNetVADTrainingAdapter

    return NativeMarbleNetVADTrainingAdapter(model, spec)


BUILTIN_MODEL_ADAPTERS = {
    "orpheustts": OrpheusTrainingAdapter,
    "dia": _dia_adapter,
    "conversationtts": ConversationTTSTrainingAdapter,
    "cosyvoice": _cosyvoice_adapter,
    "llasa": _llasa_adapter,
    "f5tts": F5TTSTrainingAdapter,
    "gptsovits": _gptsovits_adapter,
    "mosstts": _native_mosstts_adapter,
    "neutts": _neutts_adapter,
    "outetts": _outetts_adapter,
    "qwen3tts": Qwen3TTSTrainingAdapter,
    "higgstts": _higgs_adapter,
    "irodoritts": _irodori_adapter,
    "xtts": _xtts_adapter,
    "fishtts": _native_fish_s2_adapter,
    "csm": _csm_adapter,
    "echo": _echo_adapter,
    "chatterbox": _chatterbox_adapter,
    "vui": _vui_adapter,
    "zonos": _zonos_adapter,
    "zonos2": _zonos2_adapter,
    "vibevoice": _vibevoice_adapter,
    "voxcpm": _voxcpm_adapter,
    "omnivoice": _omnivoice_adapter,
    "vits": _vits_adapter,
    "kokoro": _kokoro_adapter,
    "parlertts": _parlertts_adapter,
    "speecht5": _native_speecht5_adapter,
    "supertonic": _supertonic_adapter,
    "bark": _bark_adapter,
    "inflecttts": _inflecttts_adapter,
    "styletts2": _styletts2_adapter,
    "melotts": _melotts_adapter,
    "openvoice": _openvoice_adapter,
    "asr_whisper": _native_whisper_adapter,
    "asr_whisperx": _native_whisper_adapter,
    "asr_openai_whisper": _native_whisper_adapter,
    "asr_faster_whisper": _native_whisper_adapter,
    "asr_transformers": _transformers_asr_adapter,
    "asr_tiron": _native_whisper_adapter,
    "asr_qwen3": _native_qwen3_asr_adapter,
    "asr_funasr": _native_sensevoice_adapter,
    "asr_vibevoice": _native_vibevoice_asr_adapter,
    "asr_granite_speech": _native_granite_speech_adapter,
    "asr_parakeet_tdt": _native_parakeet_tdt_adapter,
    "asr_nemotron": _native_nemotron_asr_adapter,
    "asr_cohere": _native_cohere_asr_adapter,
    "asr_seamless_m4t_v2": _native_seamless_m4t_v2_adapter,
    "asr_medasr": _native_medasr_adapter,
    "asr_wav2vec2": _native_wav2vec2_adapter,
    "asr_nemo": _native_nemo_ctc_adapter,
    "asr_espnet": _native_espnet_adapter,
    "asr_wenet": _native_wenet_u2pp_adapter,
    "asr_speechbrain": _native_speechbrain_asr_adapter,
    "asr_hubert": _native_hubert_adapter,
    "asr_wavlm": _native_wavlm_adapter,
    "asr_moonshine": _native_moonshine_adapter,
    "vad_transformers": _transformers_vad_adapter,
    "vad_silero": _native_silero_vad_adapter,
    "vad_sherpa_onnx": _native_sherpa_vad_adapter,
    "vad_funasr": _native_fsmn_vad_adapter,
    "vad_speechbrain": _native_speechbrain_vad_adapter,
    "vad_nemo": _native_marblenet_vad_adapter,
    "vad_pyannote": _native_pyannet_adapter,
    "vad_pyannote_segmentation": _native_pyannet_adapter,
    "vad_pyannote_brouhaha": _native_pyannet_adapter,
}
