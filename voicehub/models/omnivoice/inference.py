"""OmniVoice integration backed by vendored k2-fsa source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype


class OmniVoiceConfig(VoiceHubConfig):
    """Configuration for multilingual OmniVoice synthesis."""

    model_type = "omnivoice"

    def __init__(
        self,
        *,
        torch_dtype: str = "float16",
        training_torch_dtype: str = "float32",
        load_asr: bool = False,
        asr_model_name: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.training_torch_dtype = training_torch_dtype
        self.load_asr = load_asr
        self.asr_model_name = asr_model_name


class OmniVoiceForTextToSpeech(PreTrainedTTSModel):
    """Massively multilingual cloning, design, and automatic voices."""

    config_class = OmniVoiceConfig
    default_model_name_or_path = "k2-fsa/OmniVoice"

    def __init__(
        self,
        config: OmniVoiceConfig | str | None = None,
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
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="omnivoice",
            install_extra="omnivoice",
        )
        runtime = import_optional(
            "voicehub.models.omnivoice.source.omnivoice",
            model_type="omnivoice",
            install_extra="omnivoice",
        )
        self.model = runtime.OmniVoice.from_pretrained(
            self.config.name_or_path,
            device_map=self.device,
            dtype=resolve_torch_dtype(
                torch,
                (self.config.training_torch_dtype if self.is_training_load else self.config.torch_dtype),
                self.device,
            ),
            load_asr=self.config.load_asr,
            asr_model_name=self.config.asr_model_name,
            train=self.is_training_load,
        )
        if self.model.sampling_rate is not None:
            self.config.sample_rate = int(self.model.sampling_rate)
        self._loaded_for_training = self.is_training_load

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            return
        torch = import_optional(
            "torch",
            model_type="omnivoice",
            install_extra="omnivoice",
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.training_torch_dtype,
            self.device,
        )
        self.model.to(device=self.device, dtype=dtype)
        for name in (
                "text_tokenizer",
                "audio_tokenizer",
                "feature_extractor",
                "duration_estimator",
                "_asr_pipe",
        ):
            if hasattr(self.model, name):
                setattr(self.model, name, None)
        self._loaded_for_training = True

    def _prepare_for_inference(self) -> None:
        if not self._loaded_for_training:
            return
        # The source uses the same neural module for training and inference;
        # only tokenizers, feature extraction, duration estimation, and ASR are
        # omitted in train mode. Keep the exact optimizer-owned module and
        # borrow those serving auxiliaries from a temporary inference load.
        trained_model = self.model
        previous_mode = self._loading_for_training
        self.model = None
        self._loading_for_training = False
        try:
            self._load_pretrained_model()
            serving_model = self.model
            for name in (
                    "text_tokenizer",
                    "audio_tokenizer",
                    "feature_extractor",
                    "duration_estimator",
                    "_asr_pipe",
                    "_asr_model_name",
                    "_asr_device",
                    "sampling_rate",
            ):
                if hasattr(serving_model, name):
                    setattr(
                        trained_model,
                        name,
                        getattr(serving_model, name),
                    )
            self.model = trained_model
            self._loaded_for_training = False
            if hasattr(self.model, "eval"):
                self.model.eval()
        except BaseException:
            self.model = trained_model
            self._loaded_for_training = True
            raise
        finally:
            self._loading_for_training = previous_mode

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        language: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        instruct: str | None = None,
        speed: float | None = None,
        duration: float | None = None,
        normalize_text: bool = False,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        audios = self.model.generate(
            text=text,
            language=language,
            ref_audio=speaker_audio_path,
            ref_text=reference_text,
            instruct=instruct,
            speed=speed,
            duration=duration,
            normalize_text=normalize_text,
            **generation_options,
        )
        return finish_audio_output(
            audios[0],
            self.sample_rate,
            output_file=output_file,
            metadata={"language": language},
        )


OmniVoiceTTS = OmniVoiceForTextToSpeech
