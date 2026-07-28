"""MOSS-TTS family integration backed entirely by vendored source."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


class MossTTSConfig(VoiceHubConfig):
    """Configuration shared by Delay, Local, Local v1.5, and Realtime."""

    model_type = "mosstts"

    def __init__(
        self,
        *,
        variant: str = "auto",
        codec_name_or_path: str | None = None,
        torch_dtype: str = "bfloat16",
        attention_implementation: str | None = None,
        training_channelwise_loss_weights: tuple[float, ...] | str = (1.0, 32.0),
        training_adam_beta1: float = 0.9,
        training_adam_beta2: float = 0.95,
        training_adam_epsilon: float = 1e-4,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.variant = variant
        self.codec_name_or_path = codec_name_or_path
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.training_channelwise_loss_weights = training_channelwise_loss_weights
        self.training_adam_beta1 = training_adam_beta1
        self.training_adam_beta2 = training_adam_beta2
        self.training_adam_epsilon = training_adam_epsilon


class MossTTSForTextToSpeech(PreTrainedTTSModel):
    """Unified interface for the source-released MOSS-TTS architectures."""

    config_class = MossTTSConfig
    default_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-v1.5"
    _SUPPORTED_VARIANTS = ("delay", "local", "local_v1_5", "realtime")
    _VARIANT_ALIASES = {
        "local_v15": "local_v1_5",
    }
    _DEFAULT_CODEC_BY_VARIANT = {
        "delay": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "local": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
        "local_v1_5": "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        "realtime": "OpenMOSS-Team/MOSS-Audio-Tokenizer",
    }
    _STANDARD_VARIANTS = {
        "delay": (
            "moss_tts_delay",
            "MossTTSDelayConfig",
            "MossTTSDelayModel",
            "MossTTSDelayProcessor",
            "moss_tts_delay",
        ),
        "local": (
            "moss_tts_local",
            "MossTTSDelayConfig",
            "MossTTSDelayModel",
            "MossTTSDelayProcessor",
            "moss_tts_delay",
        ),
        "local_v1_5": (
            "moss_tts_local_v1_5",
            "MossTTSLocalConfig",
            "MossTTSLocalModel",
            "MossTTSLocalProcessor",
            "moss_tts_local",
        ),
    }
    _REALTIME_GENERATION_OPTIONS = frozenset({
        "do_sample",
        "repetition_penalty",
        "repetition_window",
        "temperature",
        "top_k",
        "top_p",
    })

    def __init__(
        self,
        config: MossTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        self._processor = None
        self._torch = None
        self._variant = ""
        self._codec_name_or_path = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @classmethod
    def _normalize_variant(cls, variant: str) -> str:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("MOSS-TTS `variant` must be a non-empty string.")
        normalized = (variant.strip().lower().replace("-", "_").replace(".", "_"))
        return cls._VARIANT_ALIASES.get(normalized, normalized)

    def _resolve_variant(self) -> str:
        variant = self._normalize_variant(self.config.variant)
        if variant != "auto":
            if variant not in self._SUPPORTED_VARIANTS:
                supported = ", ".join(("auto", *self._SUPPORTED_VARIANTS))
                raise ValueError(
                    f"Unsupported MOSS-TTS variant {self.config.variant!r}. "
                    f"Choose one of: {supported}.")
            return variant
        model_id = self.config.name_or_path.lower()
        if "realtime" in model_id:
            return "realtime"
        if "local" in model_id and "v1.5" in model_id:
            return "local_v1_5"
        if "local" in model_id:
            return "local"
        return "delay"

    def _resolve_codec_name_or_path(self, variant: str) -> str:
        configured = self.config.codec_name_or_path
        if configured is None:
            return self._DEFAULT_CODEC_BY_VARIANT[variant]
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(
                "MOSS-TTS `codec_name_or_path` must be a non-empty string "
                "when explicitly configured.")
        return configured.strip()

    @staticmethod
    def _codec_sample_rate(codec) -> int:
        codec_config = getattr(codec, "config", None)
        candidates = (
            getattr(codec_config, "sampling_rate", None),
            getattr(codec_config, "sample_rate", None),
            getattr(codec, "sampling_rate", None),
            getattr(codec, "sample_rate", None),
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                sample_rate = int(candidate)
            except (TypeError, ValueError):
                continue
            if sample_rate > 0:
                return sample_rate
        raise RuntimeError("MOSS-TTS audio tokenizer does not expose a valid sample rate.")

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        variant = self._resolve_variant()
        self._resolve_codec_name_or_path(variant)
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty local path or "
                                 "None.")
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"MOSS-TTS reference audio was not found: {reference_path}.")
        max_new_tokens = model_inputs.get("max_new_tokens", 4096)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        duration_tokens = model_inputs.get("duration_tokens")
        if duration_tokens is not None and (isinstance(duration_tokens, bool) or
                                            not isinstance(duration_tokens, int) or duration_tokens <= 0):
            raise ValueError("`duration_tokens` must be a positive integer when provided.")
        for name in ("language", "instruction"):
            value = model_inputs.get(name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"`{name}` must be a non-empty string or None.")
        if variant == "realtime":
            ignored = [
                name for name in ("language", "instruction", "duration_tokens")
                if model_inputs.get(name) is not None
            ]
            if ignored:
                raise ValueError("MOSS-TTS Realtime does not support: "
                                 f"{', '.join(ignored)}.")
            common = {
                "text",
                "output_file",
                "speaker_audio_path",
                "max_new_tokens",
                "seed",
            }
            unsupported = sorted(set(model_inputs) - common - self._REALTIME_GENERATION_OPTIONS)
            if unsupported:
                raise ValueError(
                    "Unsupported MOSS-TTS Realtime generation option(s): "
                    f"{', '.join(unsupported)}.")

    def _validate_training_runtime(self) -> None:
        # Local v1.5 uses a channel-wise supervised objective rather than a
        # loss returned by its model forward. VoiceHub's built-in MOSS recipe
        # supplies that objective for safetensors checkpoints.
        identifier = self.config.name_or_path.lower()
        if ("gguf" in identifier or "llama.cpp" in identifier or "llama_cpp" in identifier):
            raise ValueError(
                "MOSS-TTS fine-tuning requires the unquantized Hugging Face "
                "safetensors graph, not its llama.cpp/GGUF serving artifact. "
                "Use OpenMOSS-Team/MOSS-TTS-v1.5 (or the matching Local/"
                "Realtime source checkpoint).")

    def _register_vendored_architectures(self, transformers) -> None:
        codec_config = import_optional(
            "voicehub.models.mosstts.source.moss_audio_tokenizer."
            "configuration_moss_audio_tokenizer",
            model_type="mosstts",
            install_extra=None,
        )
        codec_model = import_optional(
            "voicehub.models.mosstts.source.moss_audio_tokenizer."
            "modeling_moss_audio_tokenizer",
            model_type="mosstts",
            install_extra=None,
        )
        transformers.AutoConfig.register(
            "moss-audio-tokenizer",
            codec_config.MossAudioTokenizerConfig,
            exist_ok=True,
        )
        transformers.AutoModel.register(
            codec_config.MossAudioTokenizerConfig,
            codec_model.MossAudioTokenizerModel,
            exist_ok=True,
        )

    @staticmethod
    def _variant_module_path(package: str, module: str) -> str:
        return ("voicehub.models.mosstts.source."
                f"{package}.{module}")

    def _load_standard_variant(
        self,
        transformers,
        *,
        dtype,
    ) -> None:
        (
            package,
            config_class_name,
            model_class_name,
            processor_class_name,
            model_type,
        ) = self._STANDARD_VARIANTS[self._variant]
        configuration = import_optional(
            self._variant_module_path(package, "configuration_moss_tts"),
            model_type="mosstts",
            install_extra=None,
        )
        modeling = import_optional(
            self._variant_module_path(package, "modeling_moss_tts"),
            model_type="mosstts",
            install_extra=None,
        )
        processing = import_optional(
            self._variant_module_path(package, "processing_moss_tts"),
            model_type="mosstts",
            install_extra=None,
        )
        config_class = getattr(configuration, config_class_name)
        model_class = getattr(modeling, model_class_name)
        processor_class = getattr(processing, processor_class_name)

        transformers.AutoConfig.register(
            model_type,
            config_class,
            exist_ok=True,
        )
        load_kwargs = {"dtype": dtype}
        if self.config.attention_implementation:
            load_kwargs["attn_implementation"] = (self.config.attention_implementation)
        self.model = model_class.from_pretrained(
            self.config.name_or_path,
            **load_kwargs,
        ).to(self.device)
        self.model.eval()
        self._processor = processor_class.from_pretrained(
            self.config.name_or_path,
            codec_path=self._codec_name_or_path,
        )
        codec = self._processor.audio_tokenizer
        if callable(getattr(codec, "eval", None)):
            codec.eval()
        moved_codec = codec.to(self.device)
        if moved_codec is not None:
            codec = moved_codec
        self._processor.audio_tokenizer = codec
        self.config.sample_rate = self._codec_sample_rate(codec)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="mosstts",
            install_extra=None,
        )
        transformers = import_optional(
            "transformers",
            model_type="mosstts",
            install_extra=None,
        )
        self._register_vendored_architectures(transformers)
        self._variant = self._resolve_variant()
        self._codec_name_or_path = self._resolve_codec_name_or_path(self._variant)
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        if self._variant == "realtime":
            self._load_realtime(torch, dtype)
        else:
            self._load_standard_variant(
                transformers,
                dtype=dtype,
            )
        self._torch = torch

    def _load_realtime(self, torch, dtype) -> None:
        modeling = import_optional(
            "voicehub.models.mosstts.source.moss_tts_realtime."
            "mossttsrealtime.modeling_mossttsrealtime",
            model_type="mosstts",
            install_extra=None,
        )
        inferencer = import_optional(
            "voicehub.models.mosstts.source.moss_tts_realtime.inferencer",
            model_type="mosstts",
            install_extra=None,
        )
        tokenizer_module = import_optional(
            "transformers",
            model_type="mosstts",
            install_extra=None,
        )
        codec_module = import_optional(
            "voicehub.models.mosstts.source.moss_audio_tokenizer."
            "modeling_moss_audio_tokenizer",
            model_type="mosstts",
            install_extra=None,
        )
        tokenizer = tokenizer_module.AutoTokenizer.from_pretrained(self.config.name_or_path)
        codec = (
            codec_module.MossAudioTokenizerModel.from_pretrained(
                self._codec_name_or_path,
                dtype=dtype,
            ).eval().to(self.device))
        runtime_model = modeling.MossTTSRealtime.from_pretrained(
            self.config.name_or_path,
            dtype=dtype,
        ).to(self.device)
        runtime_model.eval()
        self.model = inferencer.MossTTSRealtimeInference(
            runtime_model,
            tokenizer,
            codec=codec,
            codec_sample_rate=self._codec_sample_rate(codec),
        )
        self._processor = codec
        self.config.sample_rate = self._codec_sample_rate(codec)
        self._torch = torch

    @staticmethod
    def _set_eval(module) -> None:
        evaluate = getattr(module, "eval", None)
        if callable(evaluate):
            evaluate()

    def _prepare_for_inference(self) -> None:
        """Restore serving mode without replacing trained parameter objects."""
        self._set_eval(self.model)
        self._set_eval(getattr(self.model, "model", None))
        self._set_eval(self._processor)
        self._set_eval(getattr(self._processor, "audio_tokenizer", None))
        self._set_eval(getattr(self.model, "codec", None))

    def _normalize_mono_audio(self, audio):
        """Return one detached float32 CPU waveform and its source channels."""
        with self._torch.inference_mode():
            waveform = self._torch.as_tensor(audio)
            if waveform.numel() == 0:
                raise RuntimeError("MOSS-TTS returned an empty audio waveform.")

            while waveform.ndim > 2 and int(waveform.shape[0]) == 1:
                waveform = waveform.squeeze(0)
            if waveform.ndim == 1:
                source_channels = 1
            elif waveform.ndim == 2:
                source_channels = int(waveform.shape[0])
                if source_channels <= 0:
                    raise RuntimeError("MOSS-TTS returned an empty audio channel dimension.")
                waveform = (waveform.squeeze(0) if source_channels == 1 else waveform.mean(dim=0))
            else:
                raise RuntimeError(
                    "MOSS-TTS must return one channel-first waveform; "
                    f"received shape {tuple(waveform.shape)}.")

            return waveform.detach().float().cpu(), source_channels

    def _generate_realtime(
        self,
        text: str,
        *,
        speaker_audio_path: str | None,
        max_new_tokens: int,
        generation_options: dict,
    ):
        with self._torch.inference_mode():
            token_batches = self.model.generate(
                text,
                reference_audio_path=speaker_audio_path,
                max_length=max_new_tokens,
                **generation_options,
            )
            if not token_batches:
                raise RuntimeError("MOSS-TTS Realtime returned no audio tokens.")
            codes = self._torch.as_tensor(
                token_batches[0],
                device=self.device,
                dtype=self._torch.long,
            )
            if codes.ndim != 2 or codes.numel() == 0:
                raise RuntimeError("MOSS-TTS Realtime returned an invalid audio-token matrix.")
            decoded = self._processor.decode(codes.transpose(0, 1))
            decoded_audio = getattr(decoded, "audio", None)
            if decoded_audio is None or len(decoded_audio) == 0:
                raise RuntimeError("MOSS-TTS Realtime returned no decoded audio.")
            return decoded_audio[0]

    def _generate_standard(
        self,
        text: str,
        *,
        speaker_audio_path: str | None,
        language: str | None,
        instruction: str | None,
        duration_tokens: int | None,
        max_new_tokens: int,
        generation_options: dict,
    ):
        reference = ([speaker_audio_path] if speaker_audio_path is not None else None)
        message = self._processor.build_user_message(
            text=text,
            reference=reference,
            instruction=instruction,
            tokens=duration_tokens,
            language=language,
        )
        batch = self._processor([[message]], mode="generation")
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with self._torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **generation_options,
            )
            messages = self._processor.decode(generated)
        audio_segments = [
            segment for message in messages if message is not None
            for segment in getattr(message, "audio_codes_list", ())
        ]
        if not audio_segments:
            raise RuntimeError("MOSS-TTS returned no decoded audio.")
        if len(audio_segments) == 1:
            return audio_segments[0]
        return self._torch.cat(
            tuple(audio_segments),
            dim=-1,
        )

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str | None = None,
        instruction: str | None = None,
        duration_tokens: int | None = None,
        max_new_tokens: int = 4096,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        with seeded_inference(
                seed,
                device=self.device,
                model_type="mosstts",
        ) as effective_seed:
            if self._variant == "realtime":
                audio = self._generate_realtime(
                    text,
                    speaker_audio_path=speaker_audio_path,
                    max_new_tokens=max_new_tokens,
                    generation_options=generation_options,
                )
            else:
                audio = self._generate_standard(
                    text,
                    speaker_audio_path=speaker_audio_path,
                    language=language,
                    instruction=instruction,
                    duration_tokens=duration_tokens,
                    max_new_tokens=max_new_tokens,
                    generation_options=generation_options,
                )

        audio, source_channels = self._normalize_mono_audio(audio)
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "variant": self._variant,
                "language": language,
                "codec_name_or_path": self._codec_name_or_path,
                "seed": effective_seed,
                "requested_seed": seed,
                "source_channels": source_channels,
                "downmixed_to_mono": source_channels > 1,
            },
        )


MossTTS = MossTTSForTextToSpeech
