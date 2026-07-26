"""Irodori-TTS integration backed by vendored runtime and DACVAE source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory


class IrodoriTTSConfig(VoiceHubConfig):
    """Configuration for Irodori-TTS v2/v3 and VoiceDesign checkpoints."""

    model_type = "irodoritts"

    def __init__(
        self,
        *,
        codec_name_or_path: str = "Aratako/Semantic-DACVAE-Japanese-32dim",
        model_precision: str = "fp32",
        codec_precision: str = "fp32",
        compile_model: bool = False,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.model_precision = model_precision
        self.codec_precision = codec_precision
        self.compile_model = compile_model


class IrodoriTTSForTextToSpeech(PreTrainedTTSModel):
    """Flow-matching speech synthesis with optional reference or caption."""

    config_class = IrodoriTTSConfig
    default_model_name_or_path = "Aratako/Irodori-TTS-500M-v3"

    def __init__(
        self,
        config: IrodoriTTSConfig | str | None = None,
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
        self._runtime_module = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.irodoritts.source.irodori_tts.inference_runtime",
            model_type="irodoritts",
            install_extra="irodoritts",
        )
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="irodoritts",
        )
        checkpoints = sorted(model_directory.glob("*.safetensors"))
        if not checkpoints:
            checkpoints = sorted(model_directory.glob("*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"No Irodori checkpoint found in {model_directory}.")
        key = runtime.RuntimeKey(
            checkpoint=str(checkpoints[0]),
            model_device=self.device,
            codec_repo=self.config.codec_name_or_path,
            model_precision=self.config.model_precision,
            codec_device=self.device,
            codec_precision=self.config.codec_precision,
            compile_model=(self.config.compile_model and not self.is_training_load),
        )
        self.model = runtime.InferenceRuntime.from_key(key)
        self._runtime_module = runtime
        self.config.sample_rate = int(self.model.codec.sample_rate)
        self._loaded_for_training = self.is_training_load

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            return
        self.model = None
        self._loading_for_training = True
        try:
            self.load()
        finally:
            self._loading_for_training = False

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        caption: str | None = None,
        no_reference: bool = False,
        seconds: float | None = None,
        duration_scale: float = 1.0,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_caption: float = 3.0,
        cfg_scale_speaker: float = 5.0,
        seed: int | None = None,
        **sampling_options,
    ) -> TTSOutput:
        self.load()
        request = self._runtime_module.SamplingRequest(
            text=text,
            caption=caption,
            ref_wav=speaker_audio_path,
            no_ref=no_reference or speaker_audio_path is None,
            seconds=seconds,
            duration_scale=duration_scale,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            seed=seed,
            **sampling_options,
        )
        result = self.model.synthesize(request)
        self.config.sample_rate = int(result.sample_rate)
        return finish_audio_output(
            result.audio,
            result.sample_rate,
            output_file=output_file,
            metadata={
                "caption": caption,
                "seed": result.used_seed,
                "stage_timings": result.stage_timings,
            },
        )


IrodoriTTS = IrodoriTTSForTextToSpeech
