"""XTTS v2 inference backed by the vendored Coqui architecture source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory


class XTTSConfig(VoiceHubConfig):
    """Configuration for multilingual XTTS v2 voice cloning."""

    model_type = "xtts"

    def __init__(
        self,
        *,
        language: str = "en",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.language = language


class XTTSForTextToSpeech(PreTrainedTTSModel):
    """Source-integrated XTTS v2 synthesis."""

    config_class = XTTSConfig
    default_model_name_or_path = "coqui/XTTS-v2"

    def __init__(
        self,
        config: XTTSConfig | str | None = None,
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
        self._xtts_config = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="xtts",
        )
        config_module = import_optional(
            "voicehub.models.xtts.source.TTS.tts.configs.xtts_config",
            model_type="xtts",
            install_extra="xtts",
        )
        model_module = import_optional(
            "voicehub.models.xtts.source.TTS.tts.models.xtts",
            model_type="xtts",
            install_extra="xtts",
        )
        xtts_config = config_module.XttsConfig()
        xtts_config.load_json(str(model_directory / "config.json"))
        model = model_module.Xtts.init_from_config(xtts_config)
        model.load_checkpoint(
            xtts_config,
            checkpoint_dir=str(model_directory),
            eval=True,
        )
        model.to(self.device)
        self._xtts_config = xtts_config
        self.model = model

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        if not speaker_audio_path:
            raise ValueError("XTTS requires speaker_audio_path for voice cloning.")
        language = language or self.config.language
        result = self.model.synthesize(
            text,
            self._xtts_config,
            speaker_wav=speaker_audio_path,
            language=language,
            **generation_options,
        )
        audio = result["wav"] if isinstance(result, dict) else result
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={"language": language},
        )


XTTS = XTTSForTextToSpeech
