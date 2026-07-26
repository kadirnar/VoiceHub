"""OuteTTS integration backed by the vendored upstream implementation."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference


class OuteTTSConfig(VoiceHubConfig):
    """Configuration for OuteTTS 1.0 and its selectable runtime."""

    model_type = "outetts"

    def __init__(
        self,
        *,
        tokenizer_path: str | None = None,
        backend: str = "HF",
        interface_version: str = "V3",
        max_seq_length: int | None = None,
        additional_model_config: dict | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.tokenizer_path = tokenizer_path
        self.backend = backend
        self.interface_version = interface_version
        self.max_seq_length = max_seq_length
        self.additional_model_config = dict(additional_model_config or {})


class OuteTTSForTextToSpeech(PreTrainedTTSModel):
    """OuteTTS synthesis without the external ``outetts`` package."""

    config_class = OuteTTSConfig
    default_model_name_or_path = "OuteAI/Llama-OuteTTS-1.0-1B"
    _BACKENDS = (
        "HF",
        "LLAMACPP",
        "EXL2",
        "EXL2ASYNC",
        "VLLM",
        "LLAMACPP_SERVER",
        "LLAMACPP_ASYNC_SERVER",
    )
    _INTERFACE_VERSIONS = ("V1", "V2", "V3")
    _GENERATION_TYPES = (
        "REGULAR",
        "CHUNKED",
        "GUIDED_WORDS",
        "STREAM",
        "BATCH",
    )
    _BATCH_BACKENDS = frozenset({
        "EXL2ASYNC",
        "VLLM",
        "LLAMACPP_ASYNC_SERVER",
    })
    _GUIDED_WORDS_BACKENDS = frozenset({
        "LLAMACPP",
        "LLAMACPP_SERVER",
    })
    _SAMPLER_OPTIONS = frozenset({
        "temperature",
        "repetition_penalty",
        "repetition_range",
        "top_k",
        "top_p",
        "min_p",
        "mirostat_tau",
        "mirostat_eta",
        "mirostat",
    })

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
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _enum_member(enum_type, value: str, *, option_name: str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{option_name}` must be a non-empty string.")
        normalized = value.strip().upper().replace("-", "_")
        compact = normalized.replace("_", "").replace(".", "")
        for member in enum_type:
            candidates = (member.name, str(member.value))
            if any(candidate.upper().replace("-", "").replace("_", "").replace(".", "") == compact
                   for candidate in candidates):
                return member
        supported = ", ".join(member.name.lower() for member in enum_type)
        raise ValueError(f"Unsupported OuteTTS {option_name} {value!r}. "
                         f"Choose one of: {supported}.")

    @staticmethod
    def _canonical_choice(
        value: str,
        choices: tuple[str, ...],
        *,
        option_name: str,
    ) -> str:
        """Resolve names and enum-style values without importing OuteTTS."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`{option_name}` must be a non-empty string.")
        compact = (value.strip().upper().replace("-", "").replace("_", "").replace(".", ""))
        for choice in choices:
            if choice.replace("_", "") == compact:
                return choice
            if choice.startswith("V") and choice[1:] == compact:
                return choice
        supported = ", ".join(choice.lower() for choice in choices)
        raise ValueError(f"Unsupported OuteTTS {option_name} {value!r}. "
                         f"Choose one of: {supported}.")

    def _interface_version(self) -> str:
        return self._canonical_choice(
            self.config.interface_version,
            self._INTERFACE_VERSIONS,
            option_name="interface_version",
        )

    def _backend(self) -> str:
        return self._canonical_choice(
            self.config.backend,
            self._BACKENDS,
            option_name="backend",
        )

    def _generation_type(self, value: str | None) -> str:
        if value is None:
            return "BATCH" if self._backend() in self._BATCH_BACKENDS else "CHUNKED"
        return self._canonical_choice(
            value,
            self._GENERATION_TYPES,
            option_name="generation_type",
        )

    def _configured_max_seq_length(self) -> int:
        value = self.config.max_seq_length
        if value is None:
            return 4_096 if self._interface_version() in {"V1", "V2"} else 8_192
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("OuteTTS `max_seq_length` must be a positive integer or None.")
        return value

    def _validate_backend_generation_pair(
        self,
        *,
        backend: str,
        interface_version: str,
        generation_type: str,
    ) -> None:
        if generation_type == "STREAM":
            raise ValueError(
                "OuteTTS streaming returns an iterator and is not supported by "
                "the waveform-returning VoiceHub `generate()` API.")
        if backend in self._BATCH_BACKENDS and generation_type != "BATCH":
            raise ValueError(
                f"OuteTTS backend {backend.lower()!r} only supports batch "
                "generation. Set `generation_type='batch'`.")
        if generation_type == "BATCH" and backend not in self._BATCH_BACKENDS:
            supported = ", ".join(name.lower() for name in sorted(self._BATCH_BACKENDS))
            raise ValueError("OuteTTS batch generation requires an asynchronous backend: "
                             f"{supported}.")
        if generation_type == "GUIDED_WORDS":
            if interface_version != "V3":
                raise ValueError("OuteTTS guided-words generation requires "
                                 "`interface_version='V3'`.")
            if backend not in self._GUIDED_WORDS_BACKENDS:
                supported = ", ".join(name.lower() for name in sorted(self._GUIDED_WORDS_BACKENDS))
                raise ValueError(
                    "OuteTTS guided-words generation requires a llama.cpp "
                    f"backend: {supported}.")

    def _validate_sampler(self, sampler: dict | None) -> None:
        if sampler is None:
            return
        if not isinstance(sampler, dict):
            raise TypeError("`sampler` must be a dictionary or None.")
        unsupported = sorted(set(sampler) - self._SAMPLER_OPTIONS)
        if unsupported:
            raise ValueError("Unsupported OuteTTS sampler option(s): "
                             f"{', '.join(unsupported)}.")

        top_k = sampler.get("top_k")
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0):
            raise ValueError("OuteTTS sampler `top_k` must be non-negative.")
        repetition_range = sampler.get("repetition_range")
        if repetition_range is not None and (isinstance(repetition_range, bool) or
                                             not isinstance(repetition_range, int) or repetition_range <= 0):
            raise ValueError("OuteTTS sampler `repetition_range` must be positive.")
        mirostat = sampler.get("mirostat")
        if mirostat is not None and not isinstance(mirostat, bool):
            raise TypeError("OuteTTS sampler `mirostat` must be a boolean.")
        for name in ("top_p", "min_p"):
            value = sampler.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"OuteTTS sampler `{name}` must be finite and in [0, 1].")
        temperature = sampler.get("temperature")
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                                        not math.isfinite(temperature) or temperature < 0):
            raise ValueError("OuteTTS sampler `temperature` must be finite and "
                             "non-negative.")
        for name in (
                "repetition_penalty",
                "mirostat_tau",
                "mirostat_eta",
        ):
            value = sampler.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real) or
                                      not math.isfinite(value) or value <= 0):
                raise ValueError(f"OuteTTS sampler `{name}` must be finite and positive.")

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        interface_version = self._interface_version()
        backend = self._backend()
        generation_type = self._generation_type(model_inputs.get("generation_type"))
        self._validate_backend_generation_pair(
            backend=backend,
            interface_version=interface_version,
            generation_type=generation_type,
        )

        speaker_audio_path = model_inputs.get("speaker_audio_path")
        speaker_profile_path = model_inputs.get("speaker_profile_path")
        if speaker_audio_path is not None and speaker_profile_path is not None:
            raise ValueError("Pass only one of `speaker_audio_path` or "
                             "`speaker_profile_path`.")
        if (interface_version in {"V1", "V2"} and speaker_audio_path is None and
                speaker_profile_path is None):
            raise ValueError(
                f"OuteTTS {interface_version} has no bundled default speaker. "
                "Provide `speaker_audio_path` or `speaker_profile_path`.")
        for name, value in (
            ("speaker_audio_path", speaker_audio_path),
            ("speaker_profile_path", speaker_profile_path),
        ):
            if value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"`{name}` must be a non-empty local path or None.")
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"OuteTTS {name.replace('_', ' ')} was not found: {path}.")
        speaker = model_inputs.get(
            "speaker",
            "EN-FEMALE-1-NEUTRAL",
        )
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError("`speaker` must be a non-empty speaker name.")
        max_seq_length = self._configured_max_seq_length()
        max_length = model_inputs.get("max_length")
        if max_length is None:
            max_length = max_seq_length
        if (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0):
            raise ValueError("`max_length` must be a positive integer.")
        if max_length > max_seq_length:
            raise ValueError(
                f"OuteTTS `max_length` ({max_length}) exceeds the configured "
                f"`max_seq_length` ({max_seq_length}).")
        self._validate_sampler(model_inputs.get("sampler"))

    def _validate_training_runtime(self) -> None:
        backend = self._backend()
        if backend != "HF":
            raise ValueError(
                "OuteTTS fine-tuning requires the differentiable HF backend; "
                f"{self.config.backend!r} is an inference-only backend.")
        options = self.config.additional_model_config
        quantized = (
            bool(options.get("load_in_4bit")) or bool(options.get("load_in_8bit")) or
            options.get("quantization_config") is not None)
        if quantized:
            raise ValueError(
                "OuteTTS's generic training adapter cannot optimize a "
                "quantized HF runtime. Use an unquantized checkpoint or "
                "register a PEFT-aware specialized adapter.")

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="outetts",
            install_extra="outetts",
        )
        runtime = import_optional(
            "voicehub.models.outetts.source.outetts",
            model_type="outetts",
            install_extra="outetts",
        )
        backend = self._enum_member(
            runtime.Backend,
            self.config.backend,
            option_name="backend",
        )
        interface_version = self._enum_member(
            runtime.InterfaceVersion,
            self.config.interface_version,
            option_name="interface_version",
        )
        model_config = runtime.ModelConfig(
            model_path=self.config.name_or_path,
            tokenizer_path=(self.config.tokenizer_path or self.config.name_or_path),
            interface_version=interface_version,
            backend=backend,
            device=self.device,
            additional_model_config=self.config.additional_model_config,
            max_seq_length=self._configured_max_seq_length(),
        )
        self.model = runtime.Interface(config=model_config)
        self._runtime = runtime
        self._torch = torch

    def _prepare_for_inference(self) -> None:
        """Restore the nested HF model without invalidating optimizer
        identity."""
        backend = getattr(self.model, "model", None)
        language_model = getattr(backend, "model", backend)
        if language_model is not None and hasattr(language_model, "eval"):
            language_model.eval()
        model_config = getattr(language_model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True

    @staticmethod
    def _normalize_output_file(output_file: str | None) -> str | None:
        """Retain OuteTTS's historical default to WAV for suffixless paths."""
        if output_file is None:
            return None
        output_path = Path(output_file)
        if output_path.suffix:
            return str(output_path)
        return f"{output_path}.wav"

    def _resolve_speaker(
        self,
        *,
        speaker: str,
        speaker_audio_path: str | None,
        speaker_profile_path: str | None,
    ):
        if speaker_profile_path:
            return self.model.load_speaker(str(Path(speaker_profile_path).expanduser()))
        if speaker_audio_path:
            return self.model.create_speaker(str(Path(speaker_audio_path).expanduser()))
        return self.model.load_default_speaker(speaker)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: str = "EN-FEMALE-1-NEUTRAL",
        speaker_audio_path: str | None = None,
        speaker_profile_path: str | None = None,
        generation_type: str | None = None,
        max_length: int | None = None,
        sampler: dict | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        backend = self._backend()
        interface_version = self._interface_version()
        resolved_generation_type = self._generation_type(generation_type)
        self._validate_backend_generation_pair(
            backend=backend,
            interface_version=interface_version,
            generation_type=resolved_generation_type,
        )
        if max_length is None:
            max_length = self._configured_max_seq_length()

        with seeded_inference(
                seed,
                device=self.device,
                model_type="outetts",
        ) as effective_seed:
            profile = self._resolve_speaker(
                speaker=speaker,
                speaker_audio_path=speaker_audio_path,
                speaker_profile_path=speaker_profile_path,
            )
            generation_config = self._runtime.GenerationConfig(
                text=text,
                speaker=profile,
                generation_type=self._enum_member(
                    self._runtime.GenerationType,
                    resolved_generation_type,
                    option_name="generation_type",
                ),
                sampler_config=self._runtime.SamplerConfig(**(sampler or {})),
                max_length=max_length,
            )
            with self._torch.inference_mode():
                generated = self.model.generate(config=generation_config)
        sample_rate = int(generated.sr)
        if sample_rate <= 0:
            raise RuntimeError("OuteTTS returned an invalid sample rate: "
                               f"{generated.sr!r}.")
        self.config.sample_rate = sample_rate
        audio = getattr(generated, "audio", generated)
        if audio is None:
            raise RuntimeError("OuteTTS returned no generated audio.")
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=self._normalize_output_file(output_file),
            metadata={
                "speaker": speaker,
                "backend": backend.lower(),
                "interface_version": interface_version.lower(),
                "generation_type": resolved_generation_type.lower(),
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


OuteTTS = OuteTTSForTextToSpeech
