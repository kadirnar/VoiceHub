"""Supertonic 3 ONNX inference backed by vendored runtime source."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory


class SupertonicConfig(VoiceHubConfig):
    """Configuration for the lightweight multilingual Supertonic runtime."""

    model_type = "supertonic"

    def __init__(
        self,
        *,
        voice: str = "M1",
        language: str = "en",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.voice = voice
        self.language = language


class SupertonicForTextToSpeech(PreTrainedTTSModel):
    """CPU-friendly Supertonic synthesis using bundled inference code."""

    config_class = SupertonicConfig
    default_model_name_or_path = "Supertone/supertonic-3"

    def __init__(
        self,
        config: SupertonicConfig | str | None = None,
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
        self._runtime = None
        self._model_directory = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _validate_training_runtime(self) -> None:
        raise RuntimeError(
            "The published Supertonic runtime contains ONNX inference "
            "sessions only and cannot receive gradients. Register a custom "
            "PyTorch training adapter backed by a trainable checkpoint.")

    def _load_pretrained_model(self) -> None:
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="supertonic",
        )
        runtime = import_optional(
            "voicehub.models.supertonic.source.supertonic.helper",
            model_type="supertonic",
            install_extra="supertonic",
        )
        onnx_directory = (
            model_directory / "onnx" if (model_directory / "onnx").is_dir() else model_directory)
        self.model = runtime.load_text_to_speech(
            str(onnx_directory),
            use_gpu=False,
        )
        self.config.sample_rate = int(self.model.sample_rate)
        self._model_directory = model_directory
        self._runtime = runtime

    def _resolve_style(self, voice: str) -> Path:
        candidate = Path(voice).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        style_path = self._model_directory / "voice_styles" / f"{voice}.json"
        if not style_path.is_file():
            raise ValueError(f"Unknown Supertonic voice {voice!r}; pass a voice style JSON.")
        return style_path

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        voice: str | None = None,
        language: str | None = None,
        total_steps: int = 5,
        speed: float = 1.05,
        silence_duration: float = 0.3,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        voice = voice or self.config.voice
        language = language or self.config.language
        style = self._runtime.load_voice_style([str(self._resolve_style(voice))])
        audio, _duration = self.model(
            text,
            language,
            style,
            total_steps,
            speed=speed,
            silence_duration=silence_duration,
            **generation_options,
        )
        return finish_audio_output(
            audio[0],
            self.sample_rate,
            output_file=output_file,
            metadata={
                "voice": voice,
                "language": language
            },
        )


SupertonicTTS = SupertonicForTextToSpeech
