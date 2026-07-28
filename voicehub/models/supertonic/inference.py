"""Supertonic 3 ONNX inference backed by vendored runtime source."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, seeded_inference

SUPPORTED_LANGUAGES = frozenset({
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    "na",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
})


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
            "sessions only and cannot receive gradients. Fine-tuning requires "
            "the upstream PyTorch generator, text encoder, style encoder, and "
            "training checkpoint rather than the exported ONNX graphs.")

    def _load_pretrained_model(self) -> None:
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="supertonic",
        )
        runtime = import_optional(
            "voicehub.models.supertonic.source.supertonic.helper",
            model_type="supertonic",
            install_extra=None,
        )
        onnx_directory = (
            model_directory / "onnx" if (model_directory / "onnx").is_dir() else model_directory)
        # The published helper currently exposes CPUExecutionProvider only.
        self.device = "cpu"
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
            available = ", ".join(
                path.stem for path in sorted((self._model_directory / "voice_styles").glob("*.json")))
            suffix = f" Available bundled voices: {available}." if available else ""
            raise ValueError(f"Unknown Supertonic voice {voice!r}; pass a voice style JSON.{suffix}")
        return style_path

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        voice = model_inputs.get("voice") or self.config.voice
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("`voice` must be a non-empty voice ID or style JSON path.")

        language = model_inputs.get("language") or self.config.language
        normalized_language = (language.strip().lower() if isinstance(language, str) else language)
        if normalized_language not in SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES))
            raise ValueError(f"Unsupported Supertonic language {language!r}. Supported: {supported}.")

        total_steps = model_inputs.get("total_steps", 5)
        if (not isinstance(total_steps, int) or isinstance(total_steps, bool) or total_steps <= 0):
            raise ValueError("`total_steps` must be a positive integer.")
        for name, default, allow_zero in (
            ("speed", 1.05, False),
            ("silence_duration", 0.3, True),
        ):
            value = model_inputs.get(name, default)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or
                    value < 0 or (not allow_zero and value == 0)):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"`{name}` must be a finite {qualifier} number.")

    def _trim_waveform(self, audio: Any, duration: Any):
        if audio is None or duration is None:
            raise RuntimeError("Supertonic returned no audio waveform.")
        waveform = audio[0] if getattr(audio, "ndim", 1) > 1 else audio
        first_duration = duration[0] if hasattr(duration, "__getitem__") else duration
        seconds = (float(first_duration.item()) if hasattr(first_duration, "item") else float(first_duration))
        if not math.isfinite(seconds) or seconds <= 0:
            raise RuntimeError(f"Supertonic returned an invalid audio duration: {seconds}.")
        sample_count = min(
            len(waveform),
            max(0, round(self.sample_rate * seconds)),
        )
        if sample_count == 0:
            raise RuntimeError("Supertonic returned an empty audio waveform.")
        return waveform[:sample_count]

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
        seed: int | None = None,
    ) -> TTSOutput:
        voice = voice or self.config.voice
        language = (language or self.config.language).strip().lower()
        style = self._runtime.load_voice_style([str(self._resolve_style(voice))])
        with seeded_inference(
                seed,
                device=self.device,
                model_type="supertonic",
        ) as effective_seed:
            audio, duration = self.model(
                text,
                language,
                style,
                total_steps,
                speed=speed,
                silence_duration=silence_duration,
            )
        return finish_audio_output(
            self._trim_waveform(audio, duration),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "voice": voice,
                "language": language,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


SupertonicTTS = SupertonicForTextToSpeech
