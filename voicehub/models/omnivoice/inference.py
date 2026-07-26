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
        load_asr: bool = False,
        asr_model_name: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
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
                self.config.torch_dtype,
                self.device,
            ),
            load_asr=self.config.load_asr,
            asr_model_name=self.config.asr_model_name,
        )
        self.config.sample_rate = int(self.model.sampling_rate)

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
