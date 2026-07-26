"""NeuTTS inference backed by vendored NeuTTS and NeuCodec source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output


class NeuTTSConfig(VoiceHubConfig):
    """Configuration for NeuTTS Air, Nano, multilingual, and 2E variants."""

    model_type = "neutts"

    def __init__(
        self,
        *,
        codec_name_or_path: str = "neuphonic/neucodec",
        language: str | None = None,
        seed: int | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.language = language
        self.seed = seed


class NeuTTSForTextToSpeech(PreTrainedTTSModel):
    """Source-integrated NeuTTS synthesis with an embedded NeuCodec runtime."""

    config_class = NeuTTSConfig
    default_model_name_or_path = "neuphonic/neutts-2e"

    def __init__(
        self,
        config: NeuTTSConfig | str | None = None,
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

    def _validate_training_runtime(self) -> None:
        if self.config.name_or_path.lower().endswith((".gguf", "-gguf")):
            raise ValueError(
                "NeuTTS fine-tuning requires a differentiable Transformers "
                "backbone; GGUF checkpoints are inference-only.")

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.neutts.source.neutts.neutts",
            model_type="neutts",
            install_extra="neutts",
        )
        self.model = runtime.NeuTTS(
            backbone_repo=self.config.name_or_path,
            backbone_device=self.device,
            codec_repo=self.config.codec_name_or_path,
            codec_device=self.device,
            language=self.config.language,
            seed=self.config.seed,
        )
        self.config.sample_rate = int(self.model.sample_rate)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        if not speaker_audio_path or not reference_text:
            raise ValueError(
                "NeuTTS requires speaker_audio_path and reference_text. "
                "This keeps fixed speaker embeddings outside the source tree.")
        reference_codes = self.model.encode_reference(speaker_audio_path)
        audio = self.model.infer(
            text,
            reference_codes,
            reference_text,
            emotion=emotion,
            temperature=temperature,
            top_k=top_k,
            **generation_options,
        )
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "emotion": emotion,
                "voice_cloned": True,
            },
        )


NeuTTSModel = NeuTTSForTextToSpeech
