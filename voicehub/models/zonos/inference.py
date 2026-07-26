"""Zonos v0.1 integration backed by vendored Zyphra source."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output


class ZonosConfig(VoiceHubConfig):
    """Configuration for Zonos transformer and hybrid checkpoints."""

    model_type = "zonos"

    def __init__(self, *, sample_rate: int = 44100, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)


class ZonosForTextToSpeech(PreTrainedTTSModel):
    """Expressive multilingual synthesis and zero-shot voice cloning."""

    config_class = ZonosConfig
    default_model_name_or_path = "Zyphra/Zonos-v0.1-transformer"

    def __init__(
        self,
        config: ZonosConfig | str | None = None,
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
        self._conditioning = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        modeling = import_optional(
            "voicehub.models.zonos.source.zonos.model",
            model_type="zonos",
            install_extra="zonos",
        )
        self._conditioning = import_optional(
            "voicehub.models.zonos.source.zonos.conditioning",
            model_type="zonos",
            install_extra="zonos",
        )
        source = Path(self.config.name_or_path).expanduser()
        if source.is_dir():
            self.model = modeling.Zonos.from_local(
                str(source / "config.json"),
                str(source / "model.safetensors"),
                device=self.device,
            )
        else:
            self.model = modeling.Zonos.from_pretrained(
                self.config.name_or_path,
                device=self.device,
            )
        self.config.sample_rate = int(self.model.autoencoder.sampling_rate)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str = "en-us",
        emotion: list[float] | None = None,
        speaking_rate: float = 15.0,
        pitch_std: float = 20.0,
        cfg_scale: float = 2.0,
        max_new_tokens: int = 2580,
        seed: int | None = None,
        **sampling_options,
    ) -> TTSOutput:
        self.load()
        torch = import_optional(
            "torch",
            model_type="zonos",
            install_extra="zonos",
        )
        speaker = None
        if speaker_audio_path:
            torchaudio = import_optional(
                "torchaudio",
                model_type="zonos",
                install_extra="zonos",
            )
            waveform, sample_rate = torchaudio.load(speaker_audio_path)
            speaker = self.model.make_speaker_embedding(waveform, sample_rate)
        if seed is not None:
            torch.manual_seed(seed)
        condition_kwargs = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "speaking_rate": speaking_rate,
            "pitch_std": pitch_std,
            "device": self.device,
        }
        if emotion is not None:
            condition_kwargs["emotion"] = emotion
        condition = self._conditioning.make_cond_dict(**condition_kwargs)
        prefix = self.model.prepare_conditioning(condition)
        codes = self.model.generate(
            prefix,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            sampling_params=sampling_options or {"min_p": 0.1},
        )
        audio = self.model.autoencoder.decode(codes).cpu()[0]
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "language": language,
                "seed": seed
            },
        )


ZonosTTS = ZonosForTextToSpeech
