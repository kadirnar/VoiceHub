"""Official GPT fine-tuning path for XTTS v2."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.contracts import TrainingContext


class XTTSTrainingAdapter(CompositeTrainingAdapter):
    """Fine-tune the XTTS GPT while keeping DVAE and vocoder frozen."""

    supports_custom_recipe = True

    def __init__(self, model, spec):
        super().__init__(model, spec)
        self._dvae = None
        self._style_mel = None
        self._dvae_mel = None

    def setup(self):
        super().setup()
        runtime = self.model.model
        for parameter in runtime.parameters():
            parameter.requires_grad_(False)
        for parameter in runtime.gpt.parameters():
            parameter.requires_grad_(True)
        return self

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        configuration.update({
            "resolved_text_loss_weight":
            float(getattr(
                self.model.config,
                "training_text_loss_weight",
                0.01,
            )),
            "resolved_mel_loss_weight":
            float(getattr(
                self.model.config,
                "training_mel_loss_weight",
                1.0,
            )),
            "resolved_lr_milestones": [
                int(value) for value in getattr(
                    self.model.config,
                    "training_lr_milestones",
                    (900_000, 2_700_000, 5_400_000),
                )
            ],
            "resolved_lr_gamma":
            float(getattr(
                self.model.config,
                "training_lr_gamma",
                0.5,
            )),
        })
        return configuration

    def create_dataset(self, records, **kwargs):
        """Build the author-provided XTTS dataset and install its collator."""
        self.setup()
        source = import_optional(
            "voicehub.models.xtts.source.TTS.tts.layers.xtts.trainer."
            "dataset",
            model_type="xtts",
            install_extra="xtts",
        )
        model_args = SimpleNamespace(
            debug_loading_failures=bool(kwargs.get("debug_loading_failures", False)),
            max_conditioning_length=int(kwargs.get("max_conditioning_length", 132_300)),
            min_conditioning_length=int(kwargs.get("min_conditioning_length", 66_150)),
            max_wav_length=int(kwargs.get("max_wav_length", 255_995)),
            max_text_length=int(kwargs.get("max_text_length", 200)),
            gpt_use_masking_gt_prompt_approach=bool(kwargs.get("mask_ground_truth_prompt", True)),
        )
        dataset_config = SimpleNamespace(
            model_args=model_args,
            training_seed=int(kwargs.get("seed", 1)),
        )
        dataset = source.XTTSDataset(
            dataset_config,
            list(records),
            self.model.model.tokenizer,
            int(kwargs.get("sample_rate", 22_050)),
            is_eval=bool(kwargs.get("is_eval", False)),
        )
        self.data_collator = dataset.collate_fn
        return dataset

    def prepare_training_inputs(
        self,
        inputs,
        context: TrainingContext,
    ):
        del context
        batch = dict(inputs)
        required = {
            "text_inputs",
            "text_lengths",
            "audio_codes",
            "wav_lengths",
            "cond_mels",
            "cond_idxs",
            "cond_lens",
        }
        if required.issubset(batch):
            return {name: batch[name] for name in required}
        raw_required = {
            "padded_text",
            "text_lengths",
            "wav",
            "wav_lengths",
            "conditioning",
            "cond_idxs",
            "cond_lens",
        }
        missing = sorted(raw_required - set(batch))
        if missing:
            raise ValueError(
                "XTTS fine-tuning requires a source-preprocessed batch or "
                "these raw collator fields: " + ", ".join(missing))
        return self._format_batch_on_device(batch)

    def _build_preprocessors(self) -> None:
        if self._dvae is not None:
            return
        self.setup()
        torch = import_optional(
            "torch",
            model_type="xtts",
            install_extra="xtts",
        )
        dvae_module = import_optional(
            "voicehub.models.xtts.source.TTS.tts.layers.xtts.dvae",
            model_type="xtts",
            install_extra="xtts",
        )
        mel_module = import_optional(
            "voicehub.models.xtts.source.TTS.tts.layers.tortoise."
            "arch_utils",
            model_type="xtts",
            install_extra="xtts",
        )
        model_directory = self.model._model_directory
        dvae_path = Path(
            getattr(self.model.config, "training_dvae_checkpoint", None) or model_directory / "dvae.pth")
        mel_norm_path = Path(
            getattr(self.model.config, "training_mel_norm_file", None) or model_directory / "mel_stats.pth")
        if not dvae_path.is_file() or not mel_norm_path.is_file():
            raise FileNotFoundError(
                "XTTS raw-batch fine-tuning requires dvae.pth and "
                "mel_stats.pth in the checkpoint directory (or explicit "
                "training_dvae_checkpoint/training_mel_norm_file paths).")

        runtime = self.model.model
        args = runtime.args
        self._dvae = dvae_module.DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=int(args.gpt_num_audio_tokens) - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )
        try:
            state = torch.load(
                dvae_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state = torch.load(dvae_path, map_location="cpu")
        self._dvae.load_state_dict(state, strict=False)
        self._dvae.to(runtime.device).eval()
        for parameter in self._dvae.parameters():
            parameter.requires_grad_(False)

        sample_rate = int(self.model._xtts_config.audio.sample_rate)
        dvae_sample_rate = int(getattr(self.model._xtts_config.audio, "dvae_sample_rate", 22_050))
        common = {
            "mel_norm_file": str(mel_norm_path),
        }
        if args.gpt_use_perceiver_resampler:
            self._style_mel = mel_module.TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                **common,
            )
        else:
            self._style_mel = mel_module.TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                **common,
            )
        self._dvae_mel = mel_module.TorchMelSpectrogram(
            sampling_rate=dvae_sample_rate,
            **common,
        )
        self._style_mel.to(runtime.device).eval()
        self._dvae_mel.to(runtime.device).eval()

    def _format_batch_on_device(self, batch):
        """Mirror ``GPTTrainer.format_batch_on_device`` exactly."""
        self._build_preprocessors()
        torch = import_optional(
            "torch",
            model_type="xtts",
            install_extra="xtts",
        )
        torchaudio = import_optional(
            "torchaudio",
            model_type="xtts",
            install_extra="xtts",
        )
        conditioning = batch["conditioning"]
        batch_size, samples, channels, time = conditioning.size()
        reshaped = conditioning.view(
            batch_size * samples,
            channels,
            time,
        )
        with torch.no_grad():
            cond_mels = self._style_mel(reshaped)
            cond_mels = cond_mels.view(
                batch_size,
                samples,
                self._style_mel.n_mel_channels,
                cond_mels.size(2),
            )
            source_rate = int(self.model._xtts_config.audio.sample_rate)
            dvae_rate = int(getattr(self.model._xtts_config.audio, "dvae_sample_rate", 22_050))
            waveform = batch["wav"]
            if source_rate != dvae_rate:
                waveform = torchaudio.functional.resample(
                    waveform,
                    orig_freq=source_rate,
                    new_freq=dvae_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="kaiser_window",
                    beta=14.769656459379492,
                )
            audio_codes = self._dvae.get_codebook_indices(self._dvae_mel(waveform))
        return {
            "text_inputs": batch["padded_text"],
            "text_lengths": batch["text_lengths"],
            "audio_codes": audio_codes,
            "wav_lengths": batch["wav_lengths"],
            "cond_mels": cond_mels,
            "cond_idxs": batch["cond_idxs"],
            "cond_lens": batch["cond_lens"],
        }

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        prepared = self.prepare_batch(context.inputs, context)
        loss_text, loss_mel, logits = self.primary_model(**prepared)
        text_weight = float(getattr(self.model.config, "training_text_loss_weight", 0.01))
        mel_weight = float(getattr(self.model.config, "training_mel_loss_weight", 1.0))
        if (not math.isfinite(text_weight) or not math.isfinite(mel_weight) or text_weight < 0 or
                mel_weight < 0 or text_weight + mel_weight <= 0):
            raise ValueError(
                "XTTS training loss weights must be finite, non-negative, "
                "and include at least one positive value.")
        text_loss = text_weight * loss_text
        mel_loss = mel_weight * loss_mel
        loss = text_loss + mel_loss
        return TTSTrainingOutput(
            loss=loss,
            logits=logits,
            losses={
                "loss": loss,
                "loss_text_ce": text_loss,
                "loss_mel_ce": mel_loss,
                "raw_text_ce": loss_text,
                "raw_mel_ce": loss_mel,
            },
            metadata={
                "model_type": self.model_type,
                "training_family": self.spec.family_name,
                "training_support": self.spec.support.value,
                "training_phase": context.phase.name,
                "optimizer_names": context.phase.optimizer_names,
                "source_native_recipe": True,
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def execute_prediction_phase(self, context: TrainingContext):
        """Evaluate source batches whose targets are derived from raw audio."""
        fields = set(context.inputs)
        prepared_fields = {
            "text_inputs",
            "text_lengths",
            "audio_codes",
            "wav_lengths",
            "cond_mels",
            "cond_idxs",
            "cond_lens",
        }
        raw_fields = {
            "padded_text",
            "text_lengths",
            "wav",
            "wav_lengths",
            "conditioning",
            "cond_idxs",
            "cond_lens",
        }
        if prepared_fields <= fields or raw_fields <= fields:
            return self.execute_training_phase(context)
        return super().execute_prediction_phase(context)

    def create_optimizer(self, name, parameters, training_args):
        del name
        torch = import_optional(
            "torch",
            model_type="xtts",
            install_extra="xtts",
        )
        decay = []
        no_decay = []
        for parameter_name, parameter in parameters:
            normalized = parameter_name.lower()
            target = (
                no_decay if parameter_name.endswith(".bias") or "norm" in normalized or
                "embedding" in normalized else decay)
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
        return torch.optim.AdamW(
            groups,
            lr=training_args.learning_rate,
            betas=(0.9, 0.96),
            eps=1e-8,
        )

    def create_scheduler(
        self,
        name,
        optimizer,
        num_training_steps,
        training_args,
    ):
        del name, num_training_steps, training_args
        torch = import_optional(
            "torch",
            model_type="xtts",
            install_extra="xtts",
        )
        milestones = tuple(
            int(value) for value in getattr(
                self.model.config,
                "training_lr_milestones",
                (900_000, 2_700_000, 5_400_000),
            ))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(getattr(self.model.config, "training_lr_gamma", 0.5)),
        )

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        safetensors = import_optional(
            "safetensors.torch",
            model_type="xtts",
            install_extra="xtts",
        )
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in self.primary_model.state_dict().items()
        }
        safetensors.save_file(
            state,
            str(destination / "gpt.safetensors"),
        )


__all__ = ["XTTSTrainingAdapter"]
