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
from voicehub.models.cosyvoice.training import CosyVoiceTrainingAdapter
from voicehub.models.dia.training import DiaTrainingAdapter
from voicehub.models.higgstts.training import HiggsTrainingAdapter
from voicehub.models.xtts.training import XTTSTrainingAdapter
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


class ConversationTTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Official two-level codec language-model objective."""

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        prepared = dict(context.inputs)
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
        loss_zero, zero_metrics = source.CrossEntropyAndAccuracy_zero(
            c0_logits[:, :zero_length],
            zero_labels[:, :zero_length],
            zero_mask[:, :zero_length],
            ignore_id=int(prepared.get("ignore_id", 0)),
        )
        residual_weights = prepared.get("residual_loss_weights")
        if residual_weights is None:
            residual_weights = [1.0] * int(residual_labels.shape[-1])
        loss_residual, residual_metrics = source.CrossEntropyAndAccuracy_residual(
            residual_logits,
            residual_labels,
            loss_weights=residual_weights,
            ignore_id=int(prepared.get("residual_ignore_id", 0)),
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


class F5TTSTrainingAdapter(
        FlowMatchingTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Native conditional-flow objective with update-coupled EMA."""

    supports_custom_recipe = True

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._ema: ExponentialMovingAverage | None = None

    def setup(self):
        super().setup()
        if self._ema is None:
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
                prepared[target] = prepared.pop(source)
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
        self._ema.update(step=step)

    def recipe_state_dict(self) -> Mapping[str, Any]:
        self.setup()
        return {"ema": self._ema.state_dict()}

    def load_recipe_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        self.setup()
        if not state_dict:
            return
        if strict and set(state_dict) != {"ema"}:
            raise ValueError("F5-TTS recipe state must contain only 'ema'.")
        if "ema" in state_dict:
            self._ema.load_state_dict(state_dict["ema"], strict=strict)

    def save_pretrained(self, save_directory) -> None:
        """Export an upstream-compatible EMA safetensors checkpoint."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        safetensors = import_optional(
            "safetensors.torch",
            model_type="f5tts",
            install_extra="training",
        )
        state = self.primary_model.state_dict()
        ema_state = self._ema.state_dict()["shadow"]
        exported = {}
        for name, value in state.items():
            averaged = ema_state.get(name, value)
            exported[f"ema_model.{name}"] = averaged.detach().cpu().contiguous()
        safetensors.save_file(
            exported,
            str(destination / "model_ema.safetensors"),
        )


class MossTTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """MOSS family adapter including Local Transformer v1.5 channel losses."""

    supports_custom_recipe = True

    def _is_local_v15(self) -> bool:
        resolver = getattr(self.model, "_resolve_variant", None)
        variant = resolver() if callable(resolver) else getattr(self.model, "_variant", "")
        return str(variant).lower().replace(".", "_") in {
            "local_v1_5",
            "local_v15",
        }

    def create_dataset(self, records, **kwargs):
        self.setup()
        if not self._is_local_v15():
            return super().create_dataset(records, **kwargs)
        dataset_module = import_optional(
            "voicehub.models.mosstts.source.moss_tts_local_v1_5."
            "finetuning.dataset",
            model_type="mosstts",
            install_extra="training",
        )
        return dataset_module.MossTTSLocalV15SFTDataset(
            records,
            self.model._processor,
            n_vq=kwargs.get("n_vq"),
        )

    def create_optimizer(self, name, parameters, training_args):
        del name
        if not self._is_local_v15():
            return None
        torch = import_optional(
            "torch",
            model_type="mosstts",
            install_extra="training",
        )
        decay = []
        no_decay = []
        for parameter_name, parameter in parameters:
            normalized = parameter_name.lower()
            target = (
                no_decay
                if parameter_name.endswith(".bias") or "norm" in normalized or "ln_" in normalized else decay)
            target.append(parameter)
        groups = []
        if decay:
            groups.append({
                "params": decay,
                "weight_decay": training_args.weight_decay,
            })
        if no_decay:
            groups.append({
                "params": no_decay,
                "weight_decay": 0.0,
            })
        config = self.model.config
        return torch.optim.AdamW(
            groups,
            lr=training_args.learning_rate,
            betas=(
                float(config.training_adam_beta1),
                float(config.training_adam_beta2),
            ),
            eps=float(config.training_adam_epsilon),
        )

    @staticmethod
    def _loss_weights(config, n_heads: int) -> list[float]:
        values = getattr(config, "training_channelwise_loss_weights", None)
        if values is None:
            values = (1.0, 32.0)
        if isinstance(values, str):
            values = tuple(float(item.strip()) for item in values.split(",") if item.strip())
        values = [float(item) for item in values]
        if len(values) == 2 and n_heads > 1:
            text_weight, total_audio_weight = values
            values = [text_weight] + [total_audio_weight / (n_heads - 1)] * (n_heads - 1)
        if len(values) != n_heads:
            raise ValueError(
                "MOSS channelwise loss weights must contain two values or "
                f"one value per head ({n_heads}).")
        if (any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0):
            raise ValueError(
                "MOSS channelwise loss weights must be finite, non-negative, "
                "and sum to a positive value.")
        return values

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        if not self._is_local_v15():
            return super().execute_training_phase(context)
        self.setup()
        prepared = dict(context.inputs)
        required = ("input_ids", "attention_mask", "labels")
        missing = [name for name in required if name not in prepared]
        if missing:
            raise ValueError("MOSS Local v1.5 fine-tuning requires: " + ", ".join(missing))
        outputs = self.primary_model(
            input_ids=prepared["input_ids"],
            attention_mask=prepared["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        loss, per_head = self._compute_local_v15_loss(
            hidden,
            prepared["labels"],
        )
        return self._training_output(
            context,
            loss=loss,
            losses={
                "loss": loss,
                **per_head,
            },
        )

    def _compute_local_v15_loss(self, hidden, labels):
        torch = import_optional(
            "torch",
            model_type="mosstts",
            install_extra="training",
        )
        functional = torch.nn.functional
        model = self.primary_model
        batch_size, seq_len, hidden_size = hidden.shape
        n_vq = int(model.config.n_vq)
        if labels.shape[-1] != n_vq + 1:
            raise ValueError(
                f"MOSS Local v1.5 expects {n_vq + 1} label channels, "
                f"received {labels.shape[-1]}.")

        weights = self._loss_weights(self.model.config, n_vq + 1)
        flat_hidden = hidden.reshape(batch_size * seq_len, hidden_size)
        flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)
        local_dtype = model.local_transformer.ln_f.weight.dtype
        prefix = model._global_hidden_to_local(flat_hidden).to(dtype=local_dtype)
        local_inputs = torch.zeros(
            (batch_size * seq_len, n_vq, prefix.shape[-1]),
            dtype=local_dtype,
            device=flat_hidden.device,
        )
        local_inputs[:, 0, :] = prefix

        audio_targets = flat_labels[:, 1:]
        for channel_index in range(n_vq - 1):
            teacher_ids = audio_targets[:, channel_index]
            embedding = model.audio_embeddings[channel_index]
            valid = (teacher_ids >= 0) & (teacher_ids < embedding.num_embeddings)
            safe_ids = teacher_ids.masked_fill(~valid, 0)
            embedded = embedding(safe_ids).to(dtype=local_dtype)
            local_inputs[:, channel_index + 1, :] = embedded * valid.unsqueeze(-1)

        local_hidden = model.local_transformer(
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            inputs_embeds=local_inputs,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cu_seqlens=None,
            num_sequences=None,
        ).last_hidden_state

        total = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
        total_weight = 0.0
        losses = {}
        text_targets = flat_labels[:, 0]
        if (hasattr(model, "_use_binary_local_text_head") and model._use_binary_local_text_head() and
                getattr(model, "local_text_lm_head", None) is not None):
            logits = model.local_text_lm_head(local_hidden[:, 0, :])
            targets = torch.full_like(text_targets, -100)
            targets = targets.masked_fill(
                text_targets.eq(int(model.config.audio_assistant_slot_token_id)),
                0,
            )
            targets = targets.masked_fill(
                text_targets.eq(int(model.config.audio_end_token_id)),
                1,
            )
        else:
            logits = model.text_lm_head(local_hidden[:, 0, :])
            targets = text_targets
        if (targets != -100).any():
            text_loss = functional.cross_entropy(
                logits.float(),
                targets,
                ignore_index=-100,
            )
            losses["text_loss"] = text_loss
            total = total + weights[0] * text_loss.float()
            total_weight += weights[0]

        for channel_index in range(n_vq):
            targets = audio_targets[:, channel_index]
            if not (targets != -100).any():
                continue
            logits = model.audio_lm_heads[channel_index](local_hidden[:, channel_index, :])
            channel_loss = functional.cross_entropy(
                logits.float(),
                targets,
                ignore_index=-100,
            )
            losses[f"audio_loss_{channel_index}"] = channel_loss
            weight = weights[channel_index + 1]
            total = total + weight * channel_loss.float()
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("MOSS Local v1.5 received a batch with all labels ignored.")
        return total / total_weight, losses

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        destination = Path(save_directory)
        if hasattr(self.primary_model, "save_pretrained"):
            self.primary_model.save_pretrained(
                destination,
                safe_serialization=True,
            )
        processor = getattr(self.model, "_processor", None)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(destination)


class Qwen3TTSTrainingAdapter(
        CausalLMTrainingAdapter,
        SourceRecipeTrainingAdapter,
):
    """Official Qwen3-TTS 12 Hz single-speaker SFT objective."""

    native_export_semantics = "inference-export"

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._target_speaker_embedding = None

    def setup(self):
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
        return self

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        configuration["resolved_sub_talker_loss_weight"] = float(
            getattr(self.model.config, "sub_talker_loss_weight", 0.3), )
        return configuration

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
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[0][-1]
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
        state_dict = {
            name: value
            for name, value in model.state_dict().items() if not name.startswith("speaker_encoder.")
        }
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


def _fish_speech_adapter(model, spec):
    # Fish's adapter extends SourceRecipeTrainingAdapter, so importing it at
    # module import time would create a cycle.
    from voicehub.models.fishtts.training import FishSpeechTrainingAdapter

    return FishSpeechTrainingAdapter(model, spec)


def _csm_adapter(model, spec):
    # CSM's adapter is kept model-local so the optional Transformers backend
    # remains lazy during ordinary VoiceHub imports.
    from voicehub.models.csm.training import CSMTrainingAdapter

    return CSMTrainingAdapter(model, spec)


def _echo_adapter(model, spec):
    from voicehub.models.echo.training import EchoTrainingAdapter

    return EchoTrainingAdapter(model, spec)


def _vui_adapter(model, spec):
    from voicehub.models.vui.training import VuiTrainingAdapter

    return VuiTrainingAdapter(model, spec)


def _zonos_adapter(model, spec):
    from voicehub.models.zonos.training import ZonosTrainingAdapter

    return ZonosTrainingAdapter(model, spec)


def _vibevoice_adapter(model, spec):
    from voicehub.models.vibevoice.training import VibeVoiceTrainingAdapter

    return VibeVoiceTrainingAdapter(model, spec)


def _vits_adapter(model, spec):
    # Keep the experimental reconstruction recipe model-local. The generic
    # VITS family adapter must continue to require an architecture-specific
    # implementation of the complete adversarial objective.
    from voicehub.models.vits.training import VitsReconstructionTrainingAdapter

    return VitsReconstructionTrainingAdapter(model, spec)


def _transformers_asr_adapter(model, spec):
    # Keep Transformers optional until this provider is selected.
    from voicehub.models.asr_transformers.training_asr_transformers import TransformersASRTrainingAdapter

    return TransformersASRTrainingAdapter(model, spec)


def _transformers_vad_adapter(model, spec):
    # Keep Transformers optional until this provider is selected.
    from voicehub.models.vad_transformers.training_vad_transformers import TransformersVADTrainingAdapter

    return TransformersVADTrainingAdapter(model, spec)


BUILTIN_MODEL_ADAPTERS = {
    "orpheustts": CodecCausalLMTrainingAdapter,
    "dia": DiaTrainingAdapter,
    "conversationtts": ConversationTTSTrainingAdapter,
    "cosyvoice": CosyVoiceTrainingAdapter,
    "llasa": CodecCausalLMTrainingAdapter,
    "f5tts": F5TTSTrainingAdapter,
    "mosstts": MossTTSTrainingAdapter,
    "neutts": CodecCausalLMTrainingAdapter,
    "outetts": CodecCausalLMTrainingAdapter,
    "qwen3tts": Qwen3TTSTrainingAdapter,
    "higgstts": HiggsTrainingAdapter,
    "xtts": XTTSTrainingAdapter,
    "fishtts": _fish_speech_adapter,
    "csm": _csm_adapter,
    "echo": _echo_adapter,
    "vui": _vui_adapter,
    "zonos": _zonos_adapter,
    "vibevoice": _vibevoice_adapter,
    "vits": _vits_adapter,
    "asr_transformers": _transformers_asr_adapter,
    "asr_whisper": _transformers_asr_adapter,
    "asr_tiron": _transformers_asr_adapter,
    "asr_qwen3": _transformers_asr_adapter,
    "asr_vibevoice": _transformers_asr_adapter,
    "asr_granite_speech": _transformers_asr_adapter,
    "asr_parakeet_tdt": _transformers_asr_adapter,
    "asr_nemotron": _transformers_asr_adapter,
    "asr_cohere": _transformers_asr_adapter,
    "asr_medasr": _transformers_asr_adapter,
    "asr_wav2vec2": _transformers_asr_adapter,
    "asr_hubert": _transformers_asr_adapter,
    "asr_wavlm": _transformers_asr_adapter,
    "asr_moonshine": _transformers_asr_adapter,
    "asr_seamless_m4t_v2": _transformers_asr_adapter,
    "vad_transformers": _transformers_vad_adapter,
}
