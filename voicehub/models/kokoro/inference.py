"""Kokoro integration backed by the source included in VoiceHub."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class KokoroConfig(VoiceHubConfig):
    """Configuration for Kokoro model and G2P pipeline."""

    model_type = "kokoro"

    def __init__(
        self,
        *,
        language_code: str = "a",
        lang_code: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.language_code = (language_code if lang_code is None else lang_code)


class KokoroForTextToSpeech(PreTrainedTTSModel):
    """Multilingual Kokoro synthesis without the ``kokoro`` package."""

    config_class = KokoroConfig
    default_model_name_or_path = "hexgrad/Kokoro-82M"

    def __init__(
        self,
        config: KokoroConfig | str | None = None,
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
        pipeline_module = import_optional(
            "voicehub.models.kokoro.pipeline",
            model_type="kokoro",
            install_extra="kokoro",
        )
        self.model = pipeline_module.KPipeline(
            lang_code=self.config.language_code,
            repo_id=self.config.name_or_path,
            device=self.device,
        )

    def _generate(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        split_pattern: str = r"\n+",
        output_file: str | None = None,
    ) -> TTSOutput:
        self.load()
        chunks = []
        segments = []
        for result in self.model(
                text,
                voice=voice,
                speed=speed,
                split_pattern=split_pattern,
        ):
            if result.audio is not None:
                chunks.append(result.audio.reshape(-1))
                segments.append(result.graphemes)
        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")

        torch = import_optional(
            "torch",
            model_type="kokoro",
            install_extra="kokoro",
        )
        output = TTSOutput(
            audio=torch.cat(chunks),
            sample_rate=self.sample_rate,
            metadata={
                "segments": segments,
                "voice": voice
            },
        )
        if output_file:
            output.save(output_file)
        return output


KokoroTTS = KokoroForTextToSpeech
