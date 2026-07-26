"""MOSS-TTS family integration backed entirely by vendored source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype


class MossTTSConfig(VoiceHubConfig):
    """Configuration shared by Delay, Local, Local v1.5, and Realtime."""

    model_type = "mosstts"

    def __init__(
        self,
        *,
        variant: str = "auto",
        codec_name_or_path: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        torch_dtype: str = "bfloat16",
        attention_implementation: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.variant = variant
        self.codec_name_or_path = codec_name_or_path
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation


class MossTTSForTextToSpeech(PreTrainedTTSModel):
    """Unified interface for the source-released MOSS-TTS architectures."""

    config_class = MossTTSConfig
    default_model_name_or_path = "OpenMOSS-Team/MOSS-TTS-v1.5"

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
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _resolve_variant(self) -> str:
        variant = self.config.variant.lower().replace("-", "_").replace(".", "_")
        if variant != "auto":
            return variant
        model_id = self.config.name_or_path.lower()
        if "realtime" in model_id:
            return "realtime"
        if "local" in model_id and "v1.5" in model_id:
            return "local_v1_5"
        if "local" in model_id:
            return "local"
        return "delay"

    def _register_vendored_architectures(self, transformers) -> None:
        codec_config = import_optional(
            "voicehub.models.mosstts.source.moss_audio_tokenizer."
            "configuration_moss_audio_tokenizer",
            model_type="mosstts",
            install_extra="mosstts",
        )
        codec_model = import_optional(
            "voicehub.models.mosstts.source.moss_audio_tokenizer."
            "modeling_moss_audio_tokenizer",
            model_type="mosstts",
            install_extra="mosstts",
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

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="mosstts",
            install_extra="mosstts",
        )
        transformers = import_optional(
            "transformers",
            model_type="mosstts",
            install_extra="mosstts",
        )
        self._register_vendored_architectures(transformers)
        self._variant = self._resolve_variant()
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )

        if self._variant == "delay":
            configuration = import_optional(
                "voicehub.models.mosstts.source.moss_tts_delay."
                "configuration_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            modeling = import_optional(
                "voicehub.models.mosstts.source.moss_tts_delay.modeling_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            processing = import_optional(
                "voicehub.models.mosstts.source.moss_tts_delay.processing_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            model_class = modeling.MossTTSDelayModel
            processor_class = processing.MossTTSDelayProcessor
            config_class = configuration.MossTTSDelayConfig
            model_type = "moss_tts_delay"
        elif self._variant == "local":
            configuration = import_optional(
                "voicehub.models.mosstts.source.moss_tts_local."
                "configuration_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            modeling = import_optional(
                "voicehub.models.mosstts.source.moss_tts_local.modeling_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            processing = import_optional(
                "voicehub.models.mosstts.source.moss_tts_local.processing_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            model_class = modeling.MossTTSDelayModel
            processor_class = processing.MossTTSDelayProcessor
            config_class = configuration.MossTTSDelayConfig
            model_type = "moss_tts_delay"
        elif self._variant in {"local_v1_5", "local_v15"}:
            configuration = import_optional(
                "voicehub.models.mosstts.source.moss_tts_local_v1_5."
                "configuration_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            modeling = import_optional(
                "voicehub.models.mosstts.source.moss_tts_local_v1_5."
                "modeling_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            processing = import_optional(
                "voicehub.models.mosstts.source.moss_tts_local_v1_5."
                "processing_moss_tts",
                model_type="mosstts",
                install_extra="mosstts",
            )
            model_class = modeling.MossTTSLocalModel
            processor_class = processing.MossTTSLocalProcessor
            config_class = configuration.MossTTSLocalConfig
            model_type = "moss_tts_local"
            self._variant = "local_v1_5"
        elif self._variant == "realtime":
            self._load_realtime(torch, dtype)
            return
        else:
            raise ValueError("variant must be auto, delay, local, local_v1_5, or realtime.")

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
            codec_path=self.config.codec_name_or_path,
        )
        self._processor.audio_tokenizer = (self._processor.audio_tokenizer.to(self.device))
        self.config.sample_rate = int(self._processor.model_config.sampling_rate)
        self._torch = torch

    def _load_realtime(self, torch, dtype) -> None:
        modeling = import_optional(
            "voicehub.models.mosstts.source.moss_tts_realtime."
            "mossttsrealtime.modeling_mossttsrealtime",
            model_type="mosstts",
            install_extra="mosstts",
        )
        inferencer = import_optional(
            "voicehub.models.mosstts.source.moss_tts_realtime.inferencer",
            model_type="mosstts",
            install_extra="mosstts",
        )
        tokenizer_module = import_optional(
            "transformers",
            model_type="mosstts",
            install_extra="mosstts",
        )
        codec_module = import_optional(
            "voicehub.models.mosstts.source.moss_audio_tokenizer."
            "modeling_moss_audio_tokenizer",
            model_type="mosstts",
            install_extra="mosstts",
        )
        tokenizer = tokenizer_module.AutoTokenizer.from_pretrained(self.config.name_or_path)
        codec = codec_module.MossAudioTokenizerModel.from_pretrained(
            self.config.codec_name_or_path,
            dtype=dtype,
        ).to(self.device)
        runtime_model = modeling.MossTTSRealtime.from_pretrained(
            self.config.name_or_path,
            dtype=dtype,
        ).to(self.device)
        runtime_model.eval()
        self.model = inferencer.MossTTSRealtimeInference(
            runtime_model,
            tokenizer,
            codec=codec,
            codec_sample_rate=int(codec.config.sampling_rate),
        )
        self._processor = codec
        self.config.sample_rate = int(codec.config.sampling_rate)
        self._torch = torch

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
        **generation_options,
    ) -> TTSOutput:
        self.load()
        if self._variant == "realtime":
            token_batches = self.model.generate(
                text,
                reference_audio_path=speaker_audio_path,
                max_length=max_new_tokens,
                **generation_options,
            )
            codes = self._torch.as_tensor(
                token_batches[0],
                device=self.device,
                dtype=self._torch.long,
            ).transpose(0, 1)
            decoded = self._processor.decode(codes)
            audio = decoded.audio[0]
        else:
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
            message = next((item for item in messages if item is not None), None)
            if message is None or not message.audio_codes_list:
                raise RuntimeError("MOSS-TTS returned no decoded audio.")
            audio = message.audio_codes_list[0]

        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "variant": self._variant,
                "language": language
            },
        )


MossTTS = MossTTSForTextToSpeech
