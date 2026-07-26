"""Echo-TTS integration using source included in VoiceHub."""

from __future__ import annotations

from functools import partial

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class EchoTTSConfig(VoiceHubConfig):
    """Configuration for Echo-TTS flow matching inference."""

    model_type = "echo"

    def __init__(
        self,
        *,
        compile_model: bool = False,
        compile: bool | None = None,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.compile_model = (compile_model if compile is None else compile)


class EchoTTSForTextToSpeech(PreTrainedTTSModel):
    """Speaker-conditioned Echo-TTS with local architecture source."""

    config_class = EchoTTSConfig
    default_model_name_or_path = "jordand/echo-tts-base"

    def __init__(
        self,
        config: EchoTTSConfig | str | None = None,
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
        self.fish_ae = None
        self.pca_state = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        from voicehub.models.echo.sampling import load_fish_ae_from_hf, load_model_from_hf, load_pca_state_from_hf

        self.model = load_model_from_hf(
            repo_id=self.config.name_or_path,
            device=self.device,
            compile=self.config.compile_model,
            delete_blockwise_modules=True,
        )
        self.fish_ae = load_fish_ae_from_hf(device=self.device)
        self.pca_state = load_pca_state_from_hf(
            repo_id=self.config.name_or_path,
            device=self.device,
        )

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 8.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        sequence_length: int = 640,
        seed: int = 0,
        rng_seed: int | None = None,
        truncation_factor: float | None = None,
    ) -> TTSOutput:
        self.load()
        from voicehub.models.echo.sampling import load_audio, sample_euler_cfg_independent_guidances, sample_pipeline

        speaker_audio = (load_audio(speaker_audio_path).to(self.device) if speaker_audio_path else None)
        sample_function = partial(
            sample_euler_cfg_independent_guidances,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=sequence_length,
        )
        effective_seed = seed if rng_seed is None else rng_seed
        audio, _ = sample_pipeline(
            model=self.model,
            fish_ae=self.fish_ae,
            pca_state=self.pca_state,
            sample_fn=sample_function,
            text_prompt=text,
            speaker_audio=speaker_audio,
            rng_seed=effective_seed,
        )
        output = TTSOutput(
            audio=audio[0],
            sample_rate=self.sample_rate,
            metadata={"seed": effective_seed},
        )
        if output_file:
            output.save(output_file)
        return output


EchoTTS = EchoTTSForTextToSpeech
