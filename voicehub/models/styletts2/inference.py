"""StyleTTS 2 integration assembled from vendored official source."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class StyleTTS2Config(VoiceHubConfig):
    """Configuration for StyleTTS 2 checkpoints and auxiliary weights."""

    model_type = "styletts2"

    def __init__(
        self,
        *,
        config_path: str | None = None,
        assets_directory: str | None = None,
        language: str = "en-us",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.config_path = config_path
        self.assets_directory = assets_directory
        self.language = language


class StyleTTS2ForTextToSpeech(PreTrainedTTSModel):
    """Style diffusion and cloning without the ``styletts2`` package."""

    config_class = StyleTTS2Config

    def __init__(
        self,
        config: StyleTTS2Config | str | None = None,
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
        if not self.config.name_or_path:
            raise ValueError("StyleTTS 2 requires model_path pointing to an official "
                             "checkpoint.")
        config_path = self.config.config_path
        if config_path is None:
            config_path = (Path(__file__).parent / "source" / "styletts2" / "Configs" / "config_libritts.yml")

        from voicehub.models.styletts2.runtime import StyleTTS2Runtime

        self.model = StyleTTS2Runtime(
            checkpoint_path=self.config.name_or_path,
            config_path=str(config_path),
            assets_directory=self.config.assets_directory,
            device=self.device,
            language=self.config.language,
        )
        self.config.sample_rate = self.model.sample_rate

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0,
        seed: int | None = None,
    ) -> TTSOutput:
        self.load()
        audio = self.model.generate(
            text,
            speaker_audio_path=speaker_audio_path,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            seed=seed,
        )
        output = TTSOutput(
            audio=audio,
            sample_rate=self.sample_rate,
            metadata={"seed": seed},
        )
        if output_file:
            output.save(output_file)
        return output


StyleTTS2 = StyleTTS2ForTextToSpeech
