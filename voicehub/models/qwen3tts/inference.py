"""Qwen3-TTS integration backed by the vendored Qwen source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype


class Qwen3TTSConfig(VoiceHubConfig):
    """Configuration for Qwen3-TTS Base, CustomVoice, and VoiceDesign."""

    model_type = "qwen3tts"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        attention_implementation: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation


class Qwen3TTSForTextToSpeech(PreTrainedTTSModel):
    """One API for every released Qwen3-TTS checkpoint role."""

    config_class = Qwen3TTSConfig
    default_model_name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    def __init__(
        self,
        config: Qwen3TTSConfig | str | None = None,
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
        torch = import_optional(
            "torch",
            model_type="qwen3tts",
            install_extra="qwen3tts",
        )
        runtime = import_optional(
            "voicehub.models.qwen3tts.source.qwen_tts",
            model_type="qwen3tts",
            install_extra="qwen3tts",
        )
        kwargs = {
            "device_map": self.device,
            "dtype": resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            ),
        }
        if self.config.attention_implementation:
            kwargs["attn_implementation"] = (self.config.attention_implementation)
        self.model = runtime.Qwen3TTSModel.from_pretrained(
            self.config.name_or_path,
            **kwargs,
        )

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        mode: str = "auto",
        language: str = "Auto",
        speaker: str = "Vivian",
        instruct: str = "",
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        x_vector_only_mode: bool = False,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        normalized_mode = mode.lower().replace("-", "_")
        model_id = self.config.name_or_path.lower()
        if normalized_mode == "auto":
            if speaker_audio_path or "base" in model_id:
                normalized_mode = "voice_clone"
            elif "voicedesign" in model_id or "voice_design" in model_id:
                normalized_mode = "voice_design"
            else:
                normalized_mode = "custom_voice"

        if normalized_mode == "voice_clone":
            if not speaker_audio_path:
                raise ValueError("voice_clone mode requires speaker_audio_path.")
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=speaker_audio_path,
                ref_text=reference_text,
                x_vector_only_mode=x_vector_only_mode,
                **generation_options,
            )
        elif normalized_mode == "voice_design":
            if not instruct:
                raise ValueError("voice_design mode requires instruct.")
            wavs, sample_rate = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
                **generation_options,
            )
        elif normalized_mode == "custom_voice":
            wavs, sample_rate = self.model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct or None,
                **generation_options,
            )
        else:
            raise ValueError("mode must be auto, custom_voice, voice_design, or voice_clone.")

        self.config.sample_rate = int(sample_rate)
        return finish_audio_output(
            wavs[0],
            sample_rate,
            output_file=output_file,
            metadata={
                "mode": normalized_mode,
                "language": language,
                "speaker": speaker,
            },
        )


Qwen3TTS = Qwen3TTSForTextToSpeech
