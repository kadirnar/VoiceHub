"""StyleTTS 2 integration assembled from vendored official source."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference


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
            raise ValueError("StyleTTS 2 requires `model_path` pointing to an official checkpoint.")
        checkpoint_path = Path(self.config.name_or_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"StyleTTS 2 checkpoint was not found: {checkpoint_path}.")
        config_path = (
            Path(self.config.config_path).expanduser() if self.config.config_path else Path(__file__).parent /
            "source" / "styletts2" / "Configs" / "config_libritts.yml")
        if not config_path.is_file():
            raise FileNotFoundError(f"StyleTTS 2 runtime configuration was not found: {config_path}.")

        from voicehub.models.styletts2.runtime import StyleTTS2Runtime

        self.model = StyleTTS2Runtime(
            checkpoint_path=str(checkpoint_path.resolve()),
            config_path=str(config_path),
            assets_directory=self.config.assets_directory,
            device=self.device,
            language=self.config.language,
        )
        self.config.sample_rate = self.model.sample_rate

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if speaker_audio_path is not None:
            if not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip():
                raise ValueError("`speaker_audio_path` must be a local audio path or None.")
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"StyleTTS 2 reference audio was not found: {reference_path}.")

        for name, default in (("alpha", 0.3), ("beta", 0.7)):
            value = model_inputs.get(name, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"`{name}` must be numeric.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be in the interval [0, 1].")

        diffusion_steps = model_inputs.get("diffusion_steps", 5)
        if (not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool) or
                diffusion_steps <= 0):
            raise ValueError("`diffusion_steps` must be a positive integer.")
        embedding_scale = model_inputs.get("embedding_scale", 1.0)
        if (not isinstance(embedding_scale, (int, float)) or isinstance(embedding_scale, bool) or
                not math.isfinite(embedding_scale) or embedding_scale <= 0):
            raise ValueError("`embedding_scale` must be a finite positive number.")
        seed = model_inputs.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("`seed` must be an integer or None.")

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
        with seeded_inference(
                seed,
                device=self.device,
                model_type="styletts2",
        ) as effective_seed:
            audio = self.model.generate(
                text,
                speaker_audio_path=speaker_audio_path,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                seed=effective_seed,
            )
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": effective_seed,
                "voice_cloned": speaker_audio_path is not None,
                "diffusion_steps": diffusion_steps,
            },
        )


StyleTTS2 = StyleTTS2ForTextToSpeech
