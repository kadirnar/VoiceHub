"""OuteTTS integration backed by the vendored upstream implementation."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class OuteTTSConfig(VoiceHubConfig):
    """Configuration for OuteTTS 1.0 and its selectable runtime."""

    model_type = "outetts"

    def __init__(
        self,
        *,
        tokenizer_path: str | None = None,
        backend: str = "HF",
        interface_version: str = "V3",
        additional_model_config: dict | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.tokenizer_path = tokenizer_path
        self.backend = backend
        self.interface_version = interface_version
        self.additional_model_config = additional_model_config or {}


class OuteTTSForTextToSpeech(PreTrainedTTSModel):
    """OuteTTS synthesis without the external ``outetts`` package."""

    config_class = OuteTTSConfig
    default_model_name_or_path = "OuteAI/Llama-OuteTTS-1.0-1B"

    def __init__(
        self,
        config: OuteTTSConfig | str | None = None,
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
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.outetts.source.outetts",
            model_type="outetts",
            install_extra="outetts",
        )
        backend = getattr(runtime.Backend, self.config.backend.upper())
        interface_version = getattr(
            runtime.InterfaceVersion,
            self.config.interface_version.upper(),
        )
        model_config = runtime.ModelConfig(
            model_path=self.config.name_or_path,
            tokenizer_path=(
                self.config.tokenizer_path or self.config.name_or_path
            ),
            interface_version=interface_version,
            backend=backend,
            device=self.device,
            additional_model_config=self.config.additional_model_config,
        )
        self.model = runtime.Interface(config=model_config)
        self._runtime = runtime

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: str = "EN-FEMALE-1-NEUTRAL",
        speaker_audio_path: str | None = None,
        speaker_profile_path: str | None = None,
        generation_type: str = "CHUNKED",
        max_length: int = 8192,
        sampler: dict | None = None,
    ) -> TTSOutput:
        self.load()
        if speaker_profile_path:
            profile = self.model.load_speaker(speaker_profile_path)
        elif speaker_audio_path:
            profile = self.model.create_speaker(speaker_audio_path)
        else:
            profile = self.model.load_default_speaker(speaker)

        generation_config = self._runtime.GenerationConfig(
            text=text,
            speaker=profile,
            generation_type=getattr(
                self._runtime.GenerationType,
                generation_type.upper(),
            ),
            sampler_config=self._runtime.SamplerConfig(**(sampler or {})),
            max_length=max_length,
        )
        generated = self.model.generate(config=generation_config)
        self.config.sample_rate = generated.sr
        if output_file:
            generated.save(output_file)

        audio = getattr(generated, "audio", generated)
        return TTSOutput(
            audio=audio,
            sample_rate=generated.sr,
            file_path=output_file,
            metadata={"speaker": speaker},
        )


OuteTTS = OuteTTSForTextToSpeech
