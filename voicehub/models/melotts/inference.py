"""MeloTTS integration backed by vendored upstream source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class MeloTTSConfig(VoiceHubConfig):
    """Configuration for multilingual MeloTTS checkpoints."""

    model_type = "melotts"

    def __init__(
        self,
        *,
        language: str = "EN",
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        use_huggingface: bool = True,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.language = language
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.use_huggingface = use_huggingface


class MeloTTSForTextToSpeech(PreTrainedTTSModel):
    """Fast multilingual synthesis without the ``melotts`` package."""

    config_class = MeloTTSConfig
    default_model_name_or_path = "EN"

    def __init__(
        self,
        config: MeloTTSConfig | str | None = None,
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
        if config.name_or_path:
            config.language = config.name_or_path
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        api = import_optional(
            "voicehub.models.melotts.source.melo.api",
            model_type="melotts",
            install_extra="melotts",
        )
        self.model = api.TTS(
            language=self.config.language,
            device=self.device,
            use_hf=self.config.use_huggingface,
            config_path=self.config.config_path,
            ckpt_path=self.config.checkpoint_path,
        )
        self.config.sample_rate = self.model.hps.data.sampling_rate

    @property
    def speakers(self) -> tuple[str, ...]:
        """List the speakers available in the loaded checkpoint."""
        self.load()
        return tuple(self.model.hps.data.spk2id)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: str | int | None = None,
        speed: float = 1.0,
        sdp_ratio: float = 0.2,
        noise_scale: float = 0.6,
        noise_scale_w: float = 0.8,
    ) -> TTSOutput:
        self.load()
        speaker_ids = self.model.hps.data.spk2id
        if speaker is None:
            speaker_id = next(iter(speaker_ids.values()))
        elif isinstance(speaker, int):
            speaker_id = speaker
        else:
            try:
                speaker_id = speaker_ids[speaker]
            except KeyError as exc:
                available = ", ".join(speaker_ids)
                raise ValueError(
                    f"Unknown speaker {speaker!r}. Available: {available}."
                ) from exc

        audio = self.model.tts_to_file(
            text,
            speaker_id,
            output_path=None,
            speed=speed,
            sdp_ratio=sdp_ratio,
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            quiet=True,
        )
        output = TTSOutput(
            audio=audio,
            sample_rate=self.sample_rate,
            metadata={"speaker_id": speaker_id},
        )
        if output_file:
            output.save(output_file)
        return output


MeloTTS = MeloTTSForTextToSpeech
