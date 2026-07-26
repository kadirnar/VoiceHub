"""OpenVoice V2 integration backed by vendored source implementations."""

from __future__ import annotations

import tempfile
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


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
        openvoice_api = import_optional(
            "voicehub.models.openvoice.source.openvoice.api",
            model_type="openvoice",
            install_extra="openvoice",
        )
        self._se_extractor = import_optional(
            "voicehub.models.openvoice.source.openvoice.se_extractor",
            model_type="openvoice",
            install_extra="openvoice",
        )
        converter_directory = (
            Path(self.config.name_or_path).expanduser() / "converter"
        )
        converter = openvoice_api.ToneColorConverter(
            str(converter_directory / "config.json"),
            device=self.device,
        )
        converter.load_ckpt(
            str(converter_directory / "checkpoint.pth")
        )
        self.model = converter

    def _base_model(self, language: str):
        if language not in self._base_models:
            melo_api = import_optional(
                "voicehub.models.melotts.source.melo.api",
                model_type="openvoice",
                install_extra="openvoice",
            )
            self._base_models[language] = melo_api.TTS(
                language=language,
                device=self.device,
            )
        return self._base_models[language]

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
    ) -> TTSOutput:
        self.load()
        base_model = self._base_model(language)
        speaker_ids = base_model.hps.data.spk2id
        speaker = speaker or next(iter(speaker_ids))
        try:
            speaker_id = speaker_ids[speaker]
        except KeyError as exc:
            available = ", ".join(speaker_ids)
            raise ValueError(
                f"Unknown base speaker {speaker!r}. Available: {available}."
            ) from exc

        target_embedding, _ = self._se_extractor.get_se(
            speaker_audio_path,
            self.model,
            vad=vad,
        )
        speaker_key = speaker.lower().replace("_", "-")
        source_embedding_path = (
            Path(self.config.name_or_path).expanduser()
            / "base_speakers"
            / "ses"
            / f"{speaker_key}.pth"
        )
        torch = import_optional(
            "torch",
            model_type="openvoice",
            install_extra="openvoice",
        )
        source_embedding = torch.load(
            str(source_embedding_path),
            map_location=self.device,
            weights_only=True,
        )

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            base_model.tts_to_file(
                text,
                speaker_id,
                str(temporary_path),
                speed=speed,
                quiet=True,
            )
            output_path = (
                Path(output_file).expanduser()
                if output_file
                else temporary_path.with_name(
                    f"{temporary_path.stem}-converted.wav"
                )
            )
            self.model.convert(
                audio_src_path=str(temporary_path),
                src_se=source_embedding,
                tgt_se=target_embedding,
                output_path=str(output_path),
                message=(
                    self.config.watermark
                    if watermark is None
                    else watermark
                ),
            )
            soundfile = import_optional(
                "soundfile",
                model_type="openvoice",
                install_extra="openvoice",
            )
            audio, sample_rate = soundfile.read(str(output_path))
            self.config.sample_rate = sample_rate
            return TTSOutput(
                audio=audio,
                sample_rate=sample_rate,
                file_path=str(output_path) if output_file else None,
                metadata={"speaker": speaker, "language": language},
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            if not output_file and "output_path" in locals():
                output_path.unlink(missing_ok=True)


OpenVoiceTTS = OpenVoiceForTextToSpeech
