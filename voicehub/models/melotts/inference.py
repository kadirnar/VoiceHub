"""MeloTTS integration backed by vendored upstream source."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, seeded_inference


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
        config: MeloTTSConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        explicit_model_source = (
            model_path is not None or isinstance(config, (str, Path)) or
            (isinstance(config, MeloTTSConfig) and bool(config.name_or_path)))
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        if (explicit_model_source and not self._looks_like_checkpoint_source(config.name_or_path, )):
            config.language = config.name_or_path
        elif not explicit_model_source:
            config.name_or_path = config.language
        if not isinstance(config.language, str) or not config.language.strip():
            raise ValueError("`language` must be a non-empty checkpoint code.")
        if not isinstance(config.use_huggingface, bool):
            raise TypeError("`use_huggingface` must be a boolean.")
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _looks_like_checkpoint_source(name_or_path: str) -> bool:
        source = Path(name_or_path).expanduser()
        return source.exists() or "/" in name_or_path or "\\" in name_or_path

    def _resolve_checkpoint_paths(self) -> tuple[str | None, str | None]:
        config_path = self.config.config_path
        checkpoint_path = self.config.checkpoint_path
        if config_path is not None and checkpoint_path is not None:
            return str(config_path), str(checkpoint_path)
        if not self._looks_like_checkpoint_source(self.config.name_or_path):
            return (
                None if config_path is None else str(config_path),
                None if checkpoint_path is None else str(checkpoint_path),
            )

        source = Path(self.config.name_or_path).expanduser()
        if source.is_file():
            model_directory = source.parent.resolve()
            if source.suffix.lower() == ".json":
                config_path = config_path or str(source.resolve())
            else:
                checkpoint_path = checkpoint_path or str(source.resolve())
        else:
            model_directory = resolve_model_directory(
                self.config.name_or_path,
                model_type="melotts",
            )
        if config_path is None:
            candidate = model_directory / "config.json"
            if not candidate.is_file():
                raise FileNotFoundError(f"MeloTTS configuration was not found: {candidate}.")
            config_path = str(candidate)
        if checkpoint_path is None:
            candidate = model_directory / "checkpoint.pth"
            if not candidate.is_file():
                raise FileNotFoundError(f"MeloTTS checkpoint was not found: {candidate}.")
            checkpoint_path = str(candidate)
        return str(config_path), str(checkpoint_path)

    def _load_pretrained_model(self) -> None:
        api = import_optional(
            "voicehub.models.melotts.source.melo.api",
            model_type="melotts",
            install_extra="melotts",
        )
        config_path, checkpoint_path = self._resolve_checkpoint_paths()
        self.model = api.TTS(
            language=self.config.language,
            device=self.device,
            use_hf=self.config.use_huggingface,
            config_path=config_path,
            ckpt_path=checkpoint_path,
        )
        self.config.sample_rate = int(self.model.hps.data.sampling_rate)

    @property
    def speakers(self) -> tuple[str, ...]:
        """List the speakers available in the loaded checkpoint."""
        self.load()
        return tuple(self.model.hps.data.spk2id)

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker = model_inputs.get("speaker")
        if speaker is not None and (isinstance(speaker, bool) or not isinstance(speaker, (str, int))):
            raise TypeError("`speaker` must be a speaker name, integer ID, or None.")
        if isinstance(speaker, str) and not speaker.strip():
            raise ValueError("`speaker` must not be an empty string.")

        for name, default in (
            ("speed", 1.0),
            ("sdp_ratio", 0.2),
            ("noise_scale", 0.6),
            ("noise_scale_w", 0.8),
        ):
            value = model_inputs.get(name, default)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"`{name}` must be numeric.")
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
            if name == "speed" and value <= 0:
                raise ValueError("`speed` must be greater than zero.")
            if name != "speed" and value < 0:
                raise ValueError(f"`{name}` must be non-negative.")
            if name == "sdp_ratio" and value > 1:
                raise ValueError("`sdp_ratio` must be in the interval [0, 1].")

    @staticmethod
    def _resolve_speaker_id(
        speaker_ids: Mapping[str, int],
        speaker: str | int | None,
    ) -> int:
        if not speaker_ids:
            raise RuntimeError("The loaded MeloTTS checkpoint defines no speakers.")
        if speaker is None:
            return int(next(iter(speaker_ids.values())))
        if isinstance(speaker, int) and not isinstance(speaker, bool):
            if speaker not in speaker_ids.values():
                available_ids = ", ".join(str(value) for value in speaker_ids.values())
                raise ValueError(f"Unknown speaker ID {speaker}. Available IDs: {available_ids}.")
            return speaker
        try:
            return int(speaker_ids[speaker])
        except KeyError as exc:
            available = ", ".join(speaker_ids)
            raise ValueError(f"Unknown speaker {speaker!r}. Available speakers: {available}.") from exc

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
        seed: int | None = None,
    ) -> TTSOutput:
        speaker_ids = self.model.hps.data.spk2id
        speaker_id = self._resolve_speaker_id(speaker_ids, speaker)

        with seeded_inference(
                seed,
                device=self.device,
                model_type="melotts",
        ) as effective_seed:
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
        if audio is None:
            raise RuntimeError("MeloTTS returned no audio waveform.")
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "speaker_id": speaker_id,
                "speed": speed,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


MeloTTS = MeloTTSForTextToSpeech
