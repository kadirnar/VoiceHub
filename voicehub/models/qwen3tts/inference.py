"""Qwen3-TTS integration backed by the vendored Qwen source."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from urllib.parse import urlparse

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


class Qwen3TTSConfig(VoiceHubConfig):
    """Configuration for Qwen3-TTS Base, CustomVoice, and VoiceDesign."""

    model_type = "qwen3tts"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        attention_implementation: str | None = None,
        training_speaker_name: str = "voicehub",
        training_speaker_id: int = 3000,
        sub_talker_loss_weight: float = 0.3,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.training_speaker_name = training_speaker_name
        self.training_speaker_id = training_speaker_id
        self.sub_talker_loss_weight = sub_talker_loss_weight


class Qwen3TTSForTextToSpeech(PreTrainedTTSModel):
    """One API for every released Qwen3-TTS checkpoint role."""

    config_class = Qwen3TTSConfig
    default_model_name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    _SUPPORTED_MODES = (
        "auto",
        "custom_voice",
        "voice_design",
        "voice_clone",
    )
    _GENERATION_OPTIONS = frozenset({
        "do_sample",
        "max_new_tokens",
        "non_streaming_mode",
        "repetition_penalty",
        "subtalker_dosample",
        "subtalker_temperature",
        "subtalker_top_k",
        "subtalker_top_p",
        "temperature",
        "top_k",
        "top_p",
    })

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

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("Qwen3-TTS `mode` must be a non-empty string.")
        return mode.strip().lower().replace("-", "_")

    def _configured_role_hint(self) -> str | None:
        model_id = (self.config.name_or_path.lower().replace("_", "-").replace(" ", "-"))
        if "customvoice" in model_id or "custom-voice" in model_id:
            return "custom_voice"
        if "voicedesign" in model_id or "voice-design" in model_id:
            return "voice_design"
        if "-base" in model_id or model_id.endswith("base"):
            return "voice_clone"
        return None

    def _loaded_role(self) -> str | None:
        wrapped_model = getattr(self.model, "model", self.model)
        role = getattr(wrapped_model, "tts_model_type", None)
        if not isinstance(role, str):
            return None
        normalized = self._normalize_mode(role)
        resolved = "voice_clone" if normalized == "base" else normalized
        if resolved not in self._SUPPORTED_MODES[1:]:
            supported = "base, custom_voice, voice_design"
            raise ValueError(
                "The loaded Qwen3-TTS checkpoint reports unsupported "
                f"`tts_model_type` {role!r}. Expected one of: {supported}.")
        return resolved

    @staticmethod
    def _normalize_reference_audio(reference_audio: str | Path, ) -> str:
        """Normalize local paths while preserving URL/base64 inputs."""
        value = str(reference_audio).strip()
        parsed = urlparse(value)
        if (parsed.scheme in {"http", "https"} and
                parsed.netloc) or value.startswith("data:audio") or ("/" not in value and
                                                                     "\\" not in value and len(value) > 256):
            return value

        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Qwen3-TTS reference audio was not found: {path}.")
        return str(path)

    def _resolve_generation_mode(
        self,
        mode: str,
        *,
        speaker_audio_path: str | None,
    ) -> str:
        normalized = self._normalize_mode(mode)
        if normalized not in self._SUPPORTED_MODES:
            supported = ", ".join(self._SUPPORTED_MODES)
            raise ValueError(f"Unsupported Qwen3-TTS mode {mode!r}. "
                             f"Choose one of: {supported}.")
        if normalized != "auto":
            return normalized
        if speaker_audio_path:
            return "voice_clone"
        return (self._loaded_role() or self._configured_role_hint() or "custom_voice")

    @staticmethod
    def _validate_mode_requirements(
        mode: str,
        *,
        speaker: str,
        speaker_audio_path: str | Path | None,
        reference_text: str | None,
        x_vector_only_mode: bool,
    ) -> None:
        if mode == "voice_clone":
            if not speaker_audio_path:
                raise ValueError("Qwen3-TTS `voice_clone` mode requires "
                                 "`speaker_audio_path`.")
            if not x_vector_only_mode and (reference_text is None or not reference_text.strip()):
                raise ValueError(
                    "Qwen3-TTS ICL voice cloning requires a non-empty "
                    "`reference_text`. Set `x_vector_only_mode=True` to clone "
                    "from only the speaker embedding.")
            return

        if speaker_audio_path is not None:
            raise ValueError("`speaker_audio_path` is only valid in `voice_clone` mode.")
        if reference_text is not None:
            raise ValueError("`reference_text` is only valid in `voice_clone` mode.")
        if x_vector_only_mode:
            raise ValueError("`x_vector_only_mode` is only valid in `voice_clone` mode.")
        if mode == "custom_voice" and not speaker.strip():
            raise ValueError("Qwen3-TTS `custom_voice` mode requires `speaker`.")

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        mode = self._normalize_mode(model_inputs.get("mode", "auto"))
        if mode not in self._SUPPORTED_MODES:
            supported = ", ".join(self._SUPPORTED_MODES)
            raise ValueError(f"Unsupported Qwen3-TTS mode {mode!r}. "
                             f"Choose one of: {supported}.")
        x_vector_only_mode = model_inputs.get(
            "x_vector_only_mode",
            False,
        )
        if not isinstance(x_vector_only_mode, bool):
            raise TypeError("`x_vector_only_mode` must be a boolean.")

        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty audio reference "
                                 "or None.")
            self._normalize_reference_audio(speaker_audio_path)
        reference_text = model_inputs.get("reference_text")
        if reference_text is not None and not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string or None.")
        if reference_text is not None and not speaker_audio_path:
            raise ValueError("`reference_text` is only valid with "
                             "`speaker_audio_path`.")
        if x_vector_only_mode and not speaker_audio_path:
            raise ValueError("`x_vector_only_mode` requires `speaker_audio_path`.")
        for name, default in (
            ("language", "Auto"),
            ("speaker", "Vivian"),
            ("instruct", ""),
        ):
            value = model_inputs.get(name, default)
            if not isinstance(value, str):
                raise TypeError(f"`{name}` must be a string.")
        language = model_inputs.get("language", "Auto")
        if not language.strip():
            raise ValueError("`language` must be a non-empty string.")
        core_options = {
            "instruct",
            "language",
            "mode",
            "output_file",
            "reference_text",
            "seed",
            "speaker",
            "speaker_audio_path",
            "text",
            "x_vector_only_mode",
        }
        unsupported = sorted(set(model_inputs) - core_options - self._GENERATION_OPTIONS)
        if unsupported:
            raise ValueError("Unsupported Qwen3-TTS generation option(s): "
                             f"{', '.join(unsupported)}.")
        for name in ("do_sample", "subtalker_dosample", "non_streaming_mode"):
            value = model_inputs.get(name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")
        for name in ("top_k", "subtalker_top_k", "max_new_tokens"):
            value = model_inputs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"`{name}` must be a positive integer.")
        for name in ("top_p", "subtalker_top_p"):
            value = model_inputs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"`{name}` must be in the interval (0, 1].")
        for name in ("temperature", "subtalker_temperature"):
            value = model_inputs.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(f"`{name}` must be finite and positive.")
        repetition_penalty = model_inputs.get("repetition_penalty")
        if repetition_penalty is not None and (isinstance(repetition_penalty, bool) or
                                               not isinstance(repetition_penalty, Real) or
                                               not math.isfinite(repetition_penalty) or
                                               repetition_penalty <= 0):
            raise ValueError("`repetition_penalty` must be finite and positive.")
        if mode != "auto":
            self._validate_mode_requirements(
                mode,
                speaker=model_inputs.get("speaker", "Vivian"),
                speaker_audio_path=model_inputs.get("speaker_audio_path"),
                reference_text=model_inputs.get("reference_text"),
                x_vector_only_mode=x_vector_only_mode,
            )

    def _validate_training_runtime(self) -> None:
        if self._configured_role_hint() in {
                "custom_voice",
                "voice_design",
        }:
            raise ValueError(
                "The official Qwen3-TTS SFT recipe starts from a 12 Hz Base "
                "checkpoint. Select Qwen/Qwen3-TTS-12Hz-0.6B-Base or "
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base.")

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="qwen3tts",
            install_extra=None,
        )
        runtime = import_optional(
            "voicehub.models.qwen3tts.source.qwen_tts",
            model_type="qwen3tts",
            install_extra=None,
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

    def _prepare_for_inference(self) -> None:
        """Restore serving state on the optimizer-owned Qwen module."""
        wrapped_model = getattr(self.model, "model", None)
        if wrapped_model is None:
            return
        if hasattr(wrapped_model, "eval"):
            wrapped_model.eval()

        model_config = getattr(wrapped_model, "config", None)
        talker_config = getattr(model_config, "talker_config", None)
        code_predictor_config = getattr(
            talker_config,
            "code_predictor_config",
            None,
        )
        for cache_config in (
                model_config,
                talker_config,
                code_predictor_config,
        ):
            if cache_config is not None and hasattr(cache_config, "use_cache"):
                cache_config.use_cache = True

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
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        normalized_mode = self._resolve_generation_mode(
            mode,
            speaker_audio_path=speaker_audio_path,
        )
        self._validate_mode_requirements(
            normalized_mode,
            speaker=speaker,
            speaker_audio_path=speaker_audio_path,
            reference_text=reference_text,
            x_vector_only_mode=x_vector_only_mode,
        )
        reference_audio = (
            self._normalize_reference_audio(speaker_audio_path) if speaker_audio_path is not None else None)

        with seeded_inference(
                seed,
                device=self.device,
                model_type="qwen3tts",
        ) as effective_seed:
            if normalized_mode == "voice_clone":
                wavs, sample_rate = self.model.generate_voice_clone(
                    text=text,
                    language=language,
                    ref_audio=reference_audio,
                    ref_text=reference_text,
                    x_vector_only_mode=x_vector_only_mode,
                    **generation_options,
                )
            elif normalized_mode == "voice_design":
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
        if wavs is None or len(wavs) == 0:
            raise RuntimeError("Qwen3-TTS returned no generated waveform.")

        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError(f"Qwen3-TTS returned an invalid sample rate: {sample_rate}.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            wavs[0],
            sample_rate,
            output_file=output_file,
            metadata={
                "mode": normalized_mode,
                "language": language,
                "speaker": speaker,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


Qwen3TTS = Qwen3TTSForTextToSpeech
