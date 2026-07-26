"""Vui integration using the architecture source included in VoiceHub."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class VuiConfig(VoiceHubConfig):
    """Configuration for Vui checkpoint loading."""

    model_type = "vui"

    def __init__(self, *, sample_rate: int = 22050, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)


class VuiForTextToSpeech(PreTrainedTTSModel):
    """Vui synthesis with locally maintained source."""

    config_class = VuiConfig

    def __init__(
        self,
        config: VuiConfig | str | None = None,
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
        from voicehub.models.vui.model import Vui

        self.model = Vui.from_pretrained(checkpoint_path=self.config.name_or_path).to(self.device)
        self.model.eval()
        self.config.sample_rate = self.model.codec.config.sample_rate

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        from voicehub.models.vui.tts import render

        waveform = render(
            self.model,
            text,
            **generation_options,
        )
        output = TTSOutput(
            audio=waveform[0],
            sample_rate=self.sample_rate,
        )
        if output_file:
            output.save(output_file)
        return output


VuiTTS = VuiForTextToSpeech
