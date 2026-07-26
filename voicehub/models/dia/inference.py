"""Dia integration using the architecture source included in VoiceHub."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class DiaConfig(VoiceHubConfig):
    """VoiceHub loading and generation configuration for Dia."""

    model_type = "dia"

    def __init__(
        self,
        *,
        compute_dtype: str = "bfloat16",
        use_torch_compile: bool = False,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.compute_dtype = compute_dtype
        self.use_torch_compile = use_torch_compile


class DiaForTextToSpeech(PreTrainedTTSModel):
    """Dialogue synthesis with the local Dia implementation."""

    config_class = DiaConfig
    default_model_name_or_path = "nari-labs/Dia-1.6B"

    def __init__(
        self,
        config: DiaConfig | str | None = None,
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
        from voicehub.models.dia.model import Dia

        self.model = Dia.from_pretrained(
            self.config.name_or_path,
            compute_dtype=self.config.compute_dtype,
            device=self.device,
        )

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        audio = self.model.generate(
            text,
            use_torch_compile=self.config.use_torch_compile,
            **generation_options,
        )
        output = TTSOutput(audio=audio, sample_rate=self.sample_rate)
        if output_file:
            output.save(output_file)
        return output


DiaVoiceHubConfig = DiaConfig
DiaTTS = DiaForTextToSpeech
