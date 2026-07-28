"""OpenVoice V2 integration backed by vendored source implementations."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference


class OpenVoiceConfig(VoiceHubConfig):
    """Configuration for OpenVoice tone-color conversion."""

    model_type = "openvoice"

    def __init__(
        self,
        *,
        watermark: str = "@MyShell",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.watermark = watermark


class OpenVoiceForTextToSpeech(PreTrainedTTSModel):
    """Cross-lingual cloning using local OpenVoice and MeloTTS source."""

    config_class = OpenVoiceConfig
    default_model_name_or_path = "checkpoints_v2"

    def __init__(
        self,
        config: OpenVoiceConfig | str | None = None,
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
        self._base_models: dict[str, object] = {}
        self._se_extractor = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        checkpoint_root = Path(self.config.name_or_path).expanduser()
        converter_directory = checkpoint_root / "converter"
        required_files = (
            converter_directory / "config.json",
            converter_directory / "checkpoint.pth",
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing OpenVoice converter asset(s): " + ", ".join(missing))

        openvoice_api = import_optional(
            "voicehub.models.openvoice.source.openvoice.api",
            model_type="openvoice",
            install_extra=None,
        )
        self._se_extractor = import_optional(
            "voicehub.models.openvoice.source.openvoice.se_extractor",
            model_type="openvoice",
            install_extra=None,
        )
        converter = openvoice_api.ToneColorConverter(
            str(converter_directory / "config.json"),
            device=self.device,
        )
        converter.load_ckpt(str(converter_directory / "checkpoint.pth"))
        self.model = converter

    def _validate_training_runtime(self) -> None:
        raise RuntimeError(
            "The public OpenVoice V2 bundle exposes the tone-color converter "
            "for inference, but does not publish the discriminator, training "
            "data pipeline, or source objective used to optimize it. A "
            "converter-only checkpoint is not sufficient for faithful VITS "
            "fine-tuning; provide an upstream full training checkpoint and "
            "recipe before registering a custom adapter.")

    def _base_model(self, language: str):
        if language not in self._base_models:
            melo_api = import_optional(
                "voicehub.models.melotts.source.melo.api",
                model_type="openvoice",
                install_extra=None,
            )
            self._base_models[language] = melo_api.TTS(
                language=language,
                device=self.device,
            )
        return self._base_models[language]

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip():
            raise ValueError("`speaker_audio_path` must point to a local reference-audio file.")
        reference_path = Path(speaker_audio_path).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"OpenVoice reference audio was not found: {reference_path}.")

        language = model_inputs.get("language", "EN_NEWEST")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty MeloTTS language code.")
        speaker = model_inputs.get("speaker")
        if speaker is not None and (not isinstance(speaker, str) or not speaker.strip()):
            raise ValueError("`speaker` must be a non-empty speaker name or None.")
        speed = model_inputs.get("speed", 1.0)
        if not isinstance(speed, (int, float)) or isinstance(speed, bool):
            raise TypeError("`speed` must be numeric.")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("`speed` must be finite and greater than zero.")

    @staticmethod
    def _resolve_speaker(
        speaker_ids: Mapping[str, int],
        requested_speaker: str | None,
    ) -> tuple[str, int]:
        if not speaker_ids:
            raise RuntimeError("The OpenVoice base checkpoint defines no speakers.")
        speaker = requested_speaker or next(iter(speaker_ids))
        try:
            return speaker, int(speaker_ids[speaker])
        except KeyError as exc:
            available = ", ".join(speaker_ids)
            raise ValueError(
                f"Unknown base speaker {speaker!r}. "
                f"Available speakers: {available}.") from exc

    def _source_embedding_path(self, speaker: str) -> Path:
        speaker_key = speaker.lower().replace("_", "-")
        path = (Path(self.config.name_or_path).expanduser() / "base_speakers" / "ses" / f"{speaker_key}.pth")
        if not path.is_file():
            raise FileNotFoundError(f"OpenVoice source-speaker embedding was not found: {path}.")
        return path

    def _load_source_embedding(self, speaker: str):
        torch = import_optional(
            "torch",
            model_type="openvoice",
            install_extra=None,
        )
        return torch.load(
            str(self._source_embedding_path(speaker)),
            map_location=self.device,
            weights_only=True,
        )

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str,
        output_file: str | None = None,
        language: str = "EN_NEWEST",
        speaker: str | None = None,
        speed: float = 1.0,
        vad: bool = True,
        watermark: str | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        with (
                tempfile.TemporaryDirectory(prefix="voicehub-openvoice-") as directory,
                seeded_inference(
                    seed,
                    device=self.device,
                    model_type="openvoice",
                ) as effective_seed,
        ):
            base_model = self._base_model(language)
            speaker_ids = base_model.hps.data.spk2id
            speaker, speaker_id = self._resolve_speaker(
                speaker_ids,
                speaker,
            )
            source_embedding = self._load_source_embedding(speaker)
            temporary_directory = Path(directory)
            target_embedding, _ = self._se_extractor.get_se(
                str(Path(speaker_audio_path).expanduser()),
                self.model,
                target_dir=str(temporary_directory / "reference"),
                vad=vad,
            )
            base_audio_path = temporary_directory / "base.wav"
            base_model.tts_to_file(
                text,
                speaker_id,
                str(base_audio_path),
                speed=speed,
                quiet=True,
            )
            output_path = (
                Path(output_file).expanduser() if output_file else temporary_directory / "converted.wav")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.convert(
                audio_src_path=str(base_audio_path),
                src_se=source_embedding,
                tgt_se=target_embedding,
                output_path=str(output_path),
                message=(self.config.watermark if watermark is None else watermark),
            )
            soundfile = import_optional(
                "soundfile",
                model_type="openvoice",
                install_extra=None,
            )
            audio, sample_rate = soundfile.read(str(output_path))
            if getattr(audio, "size", 0) == 0:
                raise RuntimeError("OpenVoice returned an empty audio waveform.")
            self.config.sample_rate = int(sample_rate)
            output = finish_audio_output(
                audio,
                self.sample_rate,
                metadata={
                    "speaker": speaker,
                    "language": language,
                    "speed": speed,
                    "seed": effective_seed,
                    "requested_seed": seed,
                },
            )
            if output_file:
                output.file_path = str(output_path.resolve())
            return output


OpenVoiceTTS = OpenVoiceForTextToSpeech
