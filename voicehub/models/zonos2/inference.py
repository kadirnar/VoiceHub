"""ZONOS2 offline inference backed by vendored source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype


class Zonos2Config(VoiceHubConfig):
    """Configuration for the ZONOS2 MoE TTS runtime."""

    model_type = "zonos2"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        decode_audio: bool = True,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.decode_audio = decode_audio


class Zonos2ForTextToSpeech(PreTrainedTTSModel):
    """Batched ZONOS2 synthesis with cloning and conditioning controls."""

    config_class = Zonos2Config
    default_model_name_or_path = "Zyphra/ZONOS2"

    def __init__(
        self,
        config: Zonos2Config | str | None = None,
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
        self._sampling_class = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _validate_training_runtime(self) -> None:
        raise RuntimeError(
            "ZONOS2's vendored TTSLLM is a fused inference engine rather than "
            "a differentiable nn.Module. Fine-tuning requires a custom "
            "training adapter built around the unfused training graph.")

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="zonos2",
            install_extra="zonos2",
        )
        runtime = import_optional(
            "voicehub.models.zonos2.source.zonos2.tts",
            model_type="zonos2",
            install_extra="zonos2",
        )
        message = import_optional(
            "voicehub.models.zonos2.source.zonos2.message",
            model_type="zonos2",
            install_extra="zonos2",
        )
        self.model = runtime.TTSLLM(
            model_path=self.config.name_or_path,
            dtype=resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            ),
            decode_audio=self.config.decode_audio,
        )
        self._sampling_class = message.TTSSamplingParams

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str = "en_us",
        speed: float | None = None,
        temperature: float = 1.15,
        top_k: int = 106,
        top_p: float = 0.0,
        min_p: float = 0.18,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
        accurate_mode: bool = True,
        text_normalization: bool = True,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        speaker_embedding = None
        if speaker_audio_path:
            speaker_embedding = self.model.embed_speaker_file(speaker_audio_path)
        params = self._sampling_class(
            temperature=temperature,
            topk=top_k,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        result = self.model.generate(
            [text],
            params,
            language=language,
            speed=speed,
            speaker_embedding=speaker_embedding,
            accurate_mode=accurate_mode,
            text_normalization=text_normalization,
            **generation_options,
        )[0]
        audio = result["audio"]
        sample_rate = int(result.get("sample_rate", self.sample_rate))
        if isinstance(audio, (bytes, bytearray, memoryview)):
            numpy = import_optional(
                "numpy",
                model_type="zonos2",
                install_extra="zonos2",
            )
            audio = numpy.frombuffer(audio, dtype=numpy.float32).copy()
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "language": language,
                "eos_frame": result.get("eos_frame"),
            },
        )


Zonos2TTS = Zonos2ForTextToSpeech
