"""F5-TTS integration backed by the vendored upstream implementation."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class F5TTSConfig(VoiceHubConfig):
    """Configuration for the source-integrated F5-TTS architecture."""

    model_type = "f5tts"

    def __init__(
        self,
        *,
        model_name: str = "F5TTS_v1_Base",
        checkpoint_path: str = "",
        vocabulary_path: str = "",
        ode_method: str = "euler",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_after_step: int = 0,
        ema_update_every: int = 1,
        vocoder_path: str | None = None,
        cache_dir: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.vocabulary_path = vocabulary_path
        self.ode_method = ode_method
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_after_step = ema_update_after_step
        self.ema_update_every = ema_update_every
        self.vocoder_path = vocoder_path
        self.cache_dir = cache_dir


class F5TTSForTextToSpeech(PreTrainedTTSModel):
    """F5-TTS voice cloning without an external ``f5-tts`` package."""

    config_class = F5TTSConfig
    default_model_name_or_path = "F5TTS_v1_Base"

    def __init__(
        self,
        config: F5TTSConfig | str | None = None,
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
            config.model_name = config.name_or_path
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        api = import_optional(
            "voicehub.models.f5tts.source.f5_tts.api",
            model_type="f5tts",
            install_extra="f5tts",
        )
        self.model = api.F5TTS(
            model=self.config.model_name,
            ckpt_file=self.config.checkpoint_path,
            vocab_file=self.config.vocabulary_path,
            ode_method=self.config.ode_method,
            use_ema=self.config.use_ema,
            vocoder_local_path=self.config.vocoder_path,
            device=self.device,
            hf_cache_dir=self.config.cache_dir,
        )
        self.config.sample_rate = self.model.target_sample_rate

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str,
        reference_text: str = "",
        output_file: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        cross_fade_duration: float = 0.15,
        remove_silence: bool = False,
    ) -> TTSOutput:
        self.load()
        waveform, _, spectrogram = self.model.infer(
            ref_file=speaker_audio_path,
            ref_text=reference_text,
            gen_text=text,
            file_wave=output_file,
            speed=speed,
            seed=seed,
            nfe_step=nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            cross_fade_duration=cross_fade_duration,
            remove_silence=remove_silence,
        )
        return TTSOutput(
            audio=waveform,
            sample_rate=self.sample_rate,
            file_path=output_file,
            metadata={
                "seed": self.model.seed,
                "spectrogram": spectrogram,
            },
        )


F5TTS = F5TTSForTextToSpeech
