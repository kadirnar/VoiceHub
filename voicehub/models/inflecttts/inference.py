"""Inflect Micro/Nano v2 inference backed by vendored architecture source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory


class InflectTTSConfig(VoiceHubConfig):
    """Configuration for the compact Inflect v2 checkpoint family."""

    model_type = "inflecttts"

    def __init__(self, *, sample_rate: int = 22050, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)


class InflectTTSForTextToSpeech(PreTrainedTTSModel):
    """Source-integrated Inflect Micro/Nano v2 speech synthesis."""

    config_class = InflectTTSConfig
    default_model_name_or_path = "owensong/Inflect-Micro-v2"

    def __init__(
        self,
        config: InflectTTSConfig | str | None = None,
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
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="inflecttts",
        )
        runtime = import_optional(
            "voicehub.models.inflecttts.source.inflect.inference",
            model_type="inflecttts",
            install_extra="inflecttts",
        )
        self.model = runtime.InflectTTS(
            model_directory,
            device=self.device,
        )
        self.config.sample_rate = int(self.model.sample_rate)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        sample_rate, audio = self.model.synthesize(
            text,
            speed=speed,
            variation=variation,
            seed=seed,
            **generation_options,
        )
        self.config.sample_rate = int(sample_rate)
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "speed": speed,
                "variation": variation,
                "seed": seed,
            },
        )


InflectTTSModel = InflectTTSForTextToSpeech
