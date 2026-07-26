"""Chatterbox integration using source included in VoiceHub."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class ChatterboxConfig(VoiceHubConfig):
    """Configuration for the original Chatterbox architecture."""

    model_type = "chatterbox"

    def __init__(self, *, sample_rate: int = 24000, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)


class ChatterboxForTextToSpeech(PreTrainedTTSModel):
    """Zero-shot voice cloning without the ``chatterbox-tts`` package."""

    config_class = ChatterboxConfig
    default_model_name_or_path = "ResembleAI/chatterbox"

    def __init__(
        self,
        config: ChatterboxConfig | str | None = None,
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
        from voicehub.models.chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(
            device=self.device,
            repo_id=self.config.name_or_path,
        )
        self.config.sample_rate = self.model.sr

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        audio_prompt_path: str | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        prompt_path = speaker_audio_path or audio_prompt_path
        waveform = self.model.generate(
            text,
            audio_prompt_path=prompt_path,
            **generation_options,
        )
        output = TTSOutput(
            audio=waveform,
            sample_rate=self.sample_rate,
        )
        if output_file:
            output.save(output_file)
        return output


ChatterboxInference = ChatterboxForTextToSpeech
