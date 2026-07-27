"""Universal Hugging Face Transformers provider for speech recognition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import Parameter, signature
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.errors import OptionalDependencyError
from voicehub.modeling_outputs import ASROutput, ASRSegment, ASRWord
from voicehub.models.asr_transformers.configuration_asr_transformers import TransformersASRConfig

_AUTO_MODEL_CLASS_NAMES = {
    "ctc": "AutoModelForCTC",
    "speech-seq2seq": "AutoModelForSpeechSeq2Seq",
    "rnnt": "AutoModelForRNNT",
    "tdt": "AutoModelForTDT",
}
_AUTO_PROBE_ORDER = ("ctc", "speech-seq2seq", "rnnt", "tdt")

# These identifiers are hints, not an exhaustive compatibility table. Unknown
# future architectures fall through to public AutoModel probing.
_KNOWN_MODEL_TYPES = {
    "ctc":
    frozenset({
        "data2vec-audio",
        "hubert",
        "lasr",
        "parakeet-ctc",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wav2vec2",
        "wav2vec2-bert",
        "wav2vec2-conformer",
        "wavlm",
    }),
    "rnnt":
    frozenset({
        "nemotron-asr",
        "nemotron-asr-streaming",
        "parakeet-rnnt",
    }),
    "tdt":
    frozenset({"parakeet-tdt"}),
    "speech-seq2seq":
    frozenset({
        "cohere-asr",
        "granite-speech",
        "granite-speech-plus",
        "kyutai-speech-to-text",
        "moonshine",
        "moonshine-streaming",
        "qwen3-asr",
        "seamless_m4t",
        "seamless_m4t_v2",
        "speech-encoder-decoder",
        "speech_to_text",
        "speecht5",
        "vibevoice-asr",
        "voxtral",
        "voxtral-realtime",
        "whisper",
    }),
}

_SERVING_ONLY_MARKERS = (
    ".gguf",
    "-gguf",
    "/gguf",
    "llama.cpp",
    "llama_cpp",
    ".onnx",
    ".ort",
    ".engine",
    ".plan",
    ".tflite",
    ".mlmodel",
)
_QUANTIZATION_KEYS = (
    "gguf_file",
    "hf_quantizer",
    "is_loaded_in_4bit",
    "is_loaded_in_8bit",
    "load_in_4bit",
    "load_in_8bit",
    "quantization_config",
    "quantization_method",
)
_AUTO_CLASS_MISMATCH_MARKERS = (
    "unrecognized configuration class",
    "is not supported for this kind of automodel",
)
_AUDIO_TEXT_MODEL_TYPE_MARKERS = (
    "audioflamingo",
    "audio_flamingo",
    "qwen2_audio",
)
_NON_ASR_ARCHITECTURE_MARKERS = (
    "foraudioframeclassification",
    "foraudioclassification",
    "fortexttoaudio",
    "fortexttospeech",
)


class TransformersASRForSpeechRecognition(PreTrainedASRModel):
    """Load inference- and training-capable Transformers ASR checkpoints.

    ``self.model`` always remains the native differentiable Transformers
    module. The high-level ASR pipeline is a disposable inference view,
    so a transition to :meth:`load_for_training` never leaves a pipeline
    or an optimized serving wrapper in the trainable graph.
    """

    config_class = TransformersASRConfig
    default_model_name_or_path = "openai/whisper-small"

    def __init__(
        self,
        config: TransformersASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        self.native_config = None
        self.transformers_processor = None
        self.architecture_family: str | None = None
        self._pipeline = None
        self._pipeline_model = None
        self._token = token
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def training_processor(self):
        """Return the native processor paired with the trainable model."""
        return self.transformers_processor

    @staticmethod
    def _local_weight_file(name_or_path: str | Path) -> Path | None:
        path = Path(name_or_path).expanduser()
        if path.is_file() and path.suffix.lower() in {".safetensors", ".bin"}:
            return path.resolve()
        return None

    def _transformers_model_source(self) -> str:
        source = self.config.name_or_path or self.default_model_name_or_path
        weight_file = self._local_weight_file(source)
        return str(weight_file.parent) if weight_file is not None else str(source)

    def _transformers_config_source(self) -> str:
        configured = self.config.config_name_or_path
        if configured is not None:
            return str(configured)
        return self._transformers_model_source()

    def _transformers_processor_source(self) -> str:
        configured = self.config.processor_name_or_path
        if configured is not None:
            return str(configured)
        return self._transformers_config_source()

    def _hub_kwargs(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "token": self._token,
            }.items() if value is not None
        }

    @staticmethod
    def _family_from_architecture_name(name: str) -> str | None:
        normalized = name.replace("_", "").replace("-", "").lower()
        if "fortdt" in normalized or normalized.endswith("tdtmodel"):
            return "tdt"
        if "forrnnt" in normalized or "transducer" in normalized:
            return "rnnt"
        if "forctc" in normalized:
            return "ctc"
        if ("forspeechseq2seq" in normalized or "speechencoderdecoder" in normalized):
            return "speech-seq2seq"
        return None

    @classmethod
    def _infer_architecture_family(cls, native_config: Any) -> str | None:
        architectures = getattr(native_config, "architectures", ()) or ()
        if isinstance(architectures, str):
            architectures = (architectures, )
        for architecture in architectures:
            architecture_name = str(architecture)
            normalized = (architecture_name.replace("_", "").replace("-", "").lower())
            if any(marker in normalized for marker in _NON_ASR_ARCHITECTURE_MARKERS):
                raise ValueError(
                    f"Checkpoint architecture {architecture_name!r} is not an "
                    "ASR head. Select a task-compatible checkpoint or use "
                    "AutoModelForVoiceActivityDetection for audio/frame "
                    "classification.")
            family = cls._family_from_architecture_name(architecture_name)
            if family is not None:
                return family

        auto_map = getattr(native_config, "auto_map", {}) or {}
        if isinstance(auto_map, Mapping):
            for auto_class_name in auto_map:
                normalized_auto_class = (str(auto_class_name).replace("_", "").lower())
                if normalized_auto_class.endswith("automodelforaudiotexttotext"):
                    raise ValueError(
                        "Audio-text-to-text checkpoints require prompt/chat-"
                        "template preprocessing and causal labels, so they "
                        "cannot be dispatched through the Transformers ASR "
                        "provider. Register a dedicated provider.")
                if normalized_auto_class.endswith((
                        "automodelforaudioclassification",
                        "automodelforaudioframeclassification",
                )):
                    raise ValueError(
                        "This checkpoint advertises an audio-classification "
                        "head, not an ASR head. Use "
                        "AutoModelForVoiceActivityDetection when appropriate.")
                for family, expected_name in _AUTO_MODEL_CLASS_NAMES.items():
                    if str(auto_class_name).endswith(expected_name):
                        return family

        model_type = str(getattr(native_config, "model_type", "")).lower()
        normalized_model_type = model_type.replace("-", "_")
        if any(marker in normalized_model_type for marker in _AUDIO_TEXT_MODEL_TYPE_MARKERS):
            raise ValueError(
                f"Checkpoint model type {model_type!r} uses the audio-text-to-"
                "text contract. Its chat prompts and causal training labels "
                "require a dedicated VoiceHub provider.")
        model_type_aliases = {
            model_type,
            model_type.replace("_", "-"),
            model_type.replace("-", "_"),
        }
        for family, model_types in _KNOWN_MODEL_TYPES.items():
            if model_type_aliases.intersection(model_types):
                return family
        if bool(getattr(native_config, "is_encoder_decoder", False)):
            return "speech-seq2seq"
        return None

    @staticmethod
    def _auto_model_class(transformers: Any, family: str):
        class_name = _AUTO_MODEL_CLASS_NAMES[family]
        model_class = getattr(transformers, class_name, None)
        if model_class is None:
            raise OptionalDependencyError(
                f"'asr_transformers' architecture family {family!r} requires "
                f"a Transformers release exposing `{class_name}`. Upgrade "
                '`transformers` in `voicehub[asr-transformers]` and retry.')
        return model_class

    def _direct_state_dict(self) -> Mapping[str, Any] | None:
        weight_file = self._local_weight_file(self.config.name_or_path)
        if weight_file is None:
            return None
        if weight_file.suffix.lower() == ".safetensors":
            safetensors = import_optional(
                "safetensors.torch",
                model_type=self.config.model_type,
                install_extra="asr-transformers",
            )
            state_dict = safetensors.load_file(
                str(weight_file),
                device="cpu",
            )
        else:
            torch = import_optional(
                "torch",
                model_type=self.config.model_type,
                install_extra="asr-transformers",
            )
            state_dict = torch.load(
                str(weight_file),
                map_location="cpu",
                weights_only=True,
            )
        if not isinstance(state_dict, Mapping):
            raise TypeError("The direct checkpoint loader did not return a state-dict mapping.")
        return state_dict

    def _model_load_kwargs(self) -> dict[str, Any]:
        options = {
            **self._hub_kwargs(),
            **self.config.model_kwargs,
            "config": self.native_config,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.use_safetensors is not None:
            options["use_safetensors"] = self.config.use_safetensors
        state_dict = self._direct_state_dict()
        if state_dict is not None:
            if "state_dict" in options:
                raise ValueError(
                    "`model_kwargs['state_dict']` cannot be combined with a "
                    "direct checkpoint file.")
            options["state_dict"] = state_dict
        return options

    @staticmethod
    def _is_auto_class_mismatch(error: ValueError) -> bool:
        message = str(error).lower()
        return any(marker in message for marker in _AUTO_CLASS_MISMATCH_MARKERS)

    def _load_native_model(self, transformers: Any) -> tuple[Any, str]:
        requested_family = self.config.architecture_family
        inferred_family = self._infer_architecture_family(self.native_config)
        model_load_kwargs = self._model_load_kwargs()
        if requested_family != "auto":
            family = requested_family
            model_class = self._auto_model_class(transformers, family)
            return (
                model_class.from_pretrained(
                    self._transformers_model_source(),
                    **model_load_kwargs,
                ),
                family,
            )
        if inferred_family is not None:
            model_class = self._auto_model_class(transformers, inferred_family)
            return (
                model_class.from_pretrained(
                    self._transformers_model_source(),
                    **model_load_kwargs,
                ),
                inferred_family,
            )

        available_families = []
        mismatch_errors = []
        for family in _AUTO_PROBE_ORDER:
            class_name = _AUTO_MODEL_CLASS_NAMES[family]
            model_class = getattr(transformers, class_name, None)
            if model_class is None:
                continue
            available_families.append(family)
            try:
                model = model_class.from_pretrained(
                    self._transformers_model_source(),
                    **model_load_kwargs,
                )
            except ValueError as error:
                if not self._is_auto_class_mismatch(error):
                    raise
                mismatch_errors.append(f"{class_name}: {error}")
                continue
            return model, family

        families = ", ".join(available_families) or "none"
        details = f" Last dispatch error: {mismatch_errors[-1]}" if mismatch_errors else ""
        raise ValueError(
            "Transformers could not map this checkpoint to a supported ASR "
            "architecture. Set `architecture_family` explicitly to one of "
            "'ctc', 'speech-seq2seq', 'rnnt', or 'tdt'. Auto classes "
            f"available in this Transformers installation: {families}.{details}")

    def _load_pretrained_model(self) -> None:
        transformers = import_optional(
            "transformers",
            model_type=self.config.model_type,
            install_extra="asr-transformers",
        )
        self.native_config = transformers.AutoConfig.from_pretrained(
            self._transformers_config_source(),
            trust_remote_code=self.config.trust_remote_code,
            **self._hub_kwargs(),
        )
        processor_class = getattr(transformers, "AutoProcessor", None)
        if processor_class is None:
            raise OptionalDependencyError(
                "'asr_transformers' requires a Transformers release exposing "
                "`AutoProcessor`. Upgrade `voicehub[asr-transformers]` and retry.")
        processor_options = {
            **self._hub_kwargs(),
            **self.config.processor_kwargs,
            "trust_remote_code": self.config.trust_remote_code,
        }
        self.transformers_processor = processor_class.from_pretrained(
            self._transformers_processor_source(),
            **processor_options,
        )
        self.model, self.architecture_family = self._load_native_model(transformers)
        if self.model is None:
            raise RuntimeError("The Transformers ASR loader returned no model.")

        has_device_map = (
            "device_map" in self.config.model_kwargs or bool(getattr(self.model, "hf_device_map", None)))
        if not has_device_map:
            move = getattr(self.model, "to", None)
            if callable(move):
                moved = move(self.device)
                if moved is not None:
                    self.model = moved

        sample_rate = self._processor_sample_rate()
        self.config.sample_rate = sample_rate

    def _processor_sample_rate(self) -> int:
        processor = self.transformers_processor
        feature_extractor = getattr(processor, "feature_extractor", processor)
        sample_rate = getattr(
            feature_extractor,
            "sampling_rate",
            self.config.sample_rate,
        )
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Real) or
                not isfinite(float(sample_rate)) or float(sample_rate) <= 0 or
                not float(sample_rate).is_integer()):
            raise ValueError("The Transformers ASR processor reported an invalid sampling rate.")
        return int(sample_rate)

    def _ensure_pipeline(self):
        if self._pipeline is not None and self._pipeline_model is self.model:
            return self._pipeline
        transformers = import_optional(
            "transformers",
            model_type=self.config.model_type,
            install_extra="asr-transformers",
        )
        processor = self.transformers_processor
        if processor is None:
            raise RuntimeError("Transformers ASR inference requires a processor.")

        pipeline_options = dict(self.config.pipeline_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        feature_extractor = getattr(processor, "feature_extractor", None)
        if tokenizer is not None and feature_extractor is not None:
            pipeline_options["tokenizer"] = tokenizer
            pipeline_options["feature_extractor"] = feature_extractor
        elif self._accepts_keyword(transformers.pipeline, "processor"):
            pipeline_options["processor"] = processor
        else:
            if tokenizer is not None:
                pipeline_options["tokenizer"] = tokenizer
            if feature_extractor is not None:
                pipeline_options["feature_extractor"] = feature_extractor
        has_device_map = (
            "device_map" in self.config.model_kwargs or bool(getattr(self.model, "hf_device_map", None)))
        if not has_device_map:
            pipeline_options["device"] = self.device
        self._pipeline = transformers.pipeline(
            task="automatic-speech-recognition",
            model=self.model,
            **pipeline_options,
        )
        self._pipeline_model = self.model
        return self._pipeline

    @staticmethod
    def _accepts_keyword(callable_object: Any, name: str) -> bool:
        """Return whether a callable advertises a keyword parameter.

        Some extension-backed callables deliberately do not expose a
        Python signature. Treat those as unknown instead of failing
        model loading.
        """
        try:
            parameters = signature(callable_object).parameters
        except (TypeError, ValueError):
            return False
        return (
            name in parameters or
            any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()))

    @staticmethod
    def _merge_generation_option(
        generation_options: dict[str, Any],
        name: str,
        value: Any,
    ) -> None:
        if value is None:
            return
        if name in generation_options and generation_options[name] != value:
            raise ValueError(f"Conflicting values for generation option {name!r}.")
        generation_options[name] = value

    def _pipeline_call_options(
        self,
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        generation_options = dict(options.pop("generate_kwargs", {}) or {})
        family = self.architecture_family
        if family == "ctc":
            if task == "translate":
                raise ValueError("CTC checkpoints cannot perform speech translation.")
            if num_beams is not None or max_new_tokens is not None:
                raise ValueError(
                    "`num_beams` and `max_new_tokens` apply to generative "
                    "ASR checkpoints, not CTC models.")
            if language is not None:
                tokenizer = getattr(
                    self.transformers_processor,
                    "tokenizer",
                    self.transformers_processor,
                )
                set_target_lang = getattr(tokenizer, "set_target_lang", None)
                if not callable(set_target_lang):
                    raise ValueError(
                        "This CTC processor cannot switch languages at runtime. "
                        "Load a language-specific checkpoint or omit `language`.")
                set_target_lang(language)
        else:
            self._merge_generation_option(
                generation_options,
                "language",
                language,
            )
            if task != "transcribe":
                self._merge_generation_option(
                    generation_options,
                    "task",
                    task,
                )
            self._merge_generation_option(
                generation_options,
                "num_beams",
                num_beams,
            )
            self._merge_generation_option(
                generation_options,
                "max_new_tokens",
                max_new_tokens,
            )

        call_options = dict(options)
        if generation_options:
            call_options["generate_kwargs"] = generation_options
        if return_timestamps:
            call_options["return_timestamps"] = return_timestamps
        if chunk_length_s is not None:
            call_options["chunk_length_s"] = chunk_length_s
        if stride_length_s is not None:
            call_options["stride_length_s"] = stride_length_s
        if batch_size is not None:
            call_options["batch_size"] = batch_size
        if hotwords is not None:
            call_options["hotwords"] = hotwords
        return call_options

    @staticmethod
    def _confidence(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        value = float(value)
        return value if isfinite(value) and 0.0 <= value <= 1.0 else None

    @classmethod
    def _mapping_confidence(cls, value: Mapping[str, Any]) -> float | None:
        for name in ("confidence", "score", "probability"):
            if name in value:
                return cls._confidence(value[name])
        return None

    @staticmethod
    def _timestamp(value: Any) -> tuple[float | None, float | None]:
        if isinstance(value, Mapping):
            values = (value.get("start"), value.get("end"))
        elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2):
            values = (value[0], value[1])
        else:
            return None, None

        output = []
        for item in values:
            if (item is None or isinstance(item, bool) or not isinstance(item, Real) or
                    not isfinite(float(item)) or float(item) < 0):
                output.append(None)
            else:
                output.append(float(item))
        start, end = output
        if start is not None and end is not None and end < start:
            return None, None
        return start, end

    @staticmethod
    def _speaker(value: Any) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @classmethod
    def _word(cls, value: Any) -> ASRWord | None:
        if isinstance(value, str):
            text = value.strip()
            return ASRWord(text=text) if text else None
        if not isinstance(value, Mapping):
            return None
        text = str(value.get("word", value.get("text", ""))).strip()
        if not text:
            return None
        start, end = cls._timestamp(value.get("timestamp", value.get("timestamps")))
        return ASRWord(
            text=text,
            start=start,
            end=end,
            confidence=cls._mapping_confidence(value),
            speaker=cls._speaker(value.get("speaker")),
        )

    @classmethod
    def _segment(
        cls,
        value: Any,
        *,
        fallback_language: str | None,
    ) -> ASRSegment | None:
        if isinstance(value, str):
            return ASRSegment(text=value)
        if not isinstance(value, Mapping):
            return None
        text = str(value.get("text", value.get("word", ""))).strip()
        nested_words = value.get("words", ())
        words = tuple(word for word in (cls._word(item) for item in nested_words or ()) if word is not None)
        start, end = cls._timestamp(value.get("timestamp", value.get("timestamps")))
        language = value.get("language", fallback_language)
        if not isinstance(language, str) or not language.strip():
            language = fallback_language
        metadata = {
            key: item
            for key, item in value.items() if key not in {
                "text",
                "word",
                "timestamp",
                "timestamps",
                "confidence",
                "score",
                "probability",
                "language",
                "speaker",
                "words",
            }
        }
        return ASRSegment(
            text=text,
            start=start,
            end=end,
            confidence=cls._mapping_confidence(value),
            language=language,
            speaker=cls._speaker(value.get("speaker")),
            words=words,
            metadata=metadata,
        )

    def _normalize_pipeline_output(
        self,
        result: Any,
        *,
        duration: float,
        timestamp_mode: bool | str,
        fallback_language: str | None = None,
    ) -> ASROutput:
        if isinstance(result, list):
            if len(result) != 1:
                raise TypeError("A single-audio transcription must return one pipeline result.")
            result = result[0]
        if isinstance(result, str):
            result = {"text": result}
        if not isinstance(result, Mapping):
            raise TypeError("The Transformers ASR pipeline must return text or a mapping.")

        text = str(result.get("text", "")).strip()
        language_value = result.get("language", fallback_language)
        language = (
            language_value.strip()
            if isinstance(language_value, str) and language_value.strip() else fallback_language)
        chunks = result.get("chunks", ()) or ()
        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise TypeError("Transformers ASR pipeline `chunks` must be a sequence.")

        segments: tuple[ASRSegment, ...]
        if timestamp_mode == "word" and chunks:
            words = tuple(word for word in (self._word(chunk) for chunk in chunks) if word is not None)
            if not text:
                text = " ".join(word.text for word in words).strip()
            starts = [word.start for word in words if word.start is not None]
            ends = [word.end for word in words if word.end is not None]
            confidences = [word.confidence for word in words if word.confidence is not None]
            segments = (
                ASRSegment(
                    text=text,
                    start=min(starts) if starts else None,
                    end=max(ends) if ends else None,
                    confidence=(sum(confidences) / len(confidences) if confidences else None),
                    language=language,
                    words=words,
                ), ) if text or words else ()
        else:
            segments = tuple(
                segment for segment in (self._segment(chunk, fallback_language=language) for chunk in chunks)
                if segment is not None)
            if not text:
                text = " ".join(segment.text for segment in segments if segment.text).strip()

        metadata = {key: value for key, value in result.items() if key not in {"text", "chunks", "language"}}
        metadata.update({
            "backend": "transformers",
            "architecture_family": self.architecture_family,
        })
        return ASROutput(
            text=text,
            segments=segments,
            language=language,
            duration=duration,
            metadata=metadata,
        )

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
        **kwargs,
    ) -> ASROutput:
        target_rate = self._processor_sample_rate()
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_rate,
        )
        options = self._pipeline_call_options(
            language=language,
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            options=dict(kwargs),
        )
        result = self._ensure_pipeline()(
            {
                "array": materialized.waveform,
                "sampling_rate": materialized.sampling_rate,
            },
            **options,
        )
        return self._normalize_pipeline_output(
            result,
            duration=materialized.duration,
            timestamp_mode=return_timestamps,
            fallback_language=language,
        )

    @staticmethod
    def _array_rank(value: Any) -> int | None:
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            return len(shape)
        except TypeError:
            ndim = getattr(value, "ndim", None)
            return int(ndim) if isinstance(ndim, Integral) else None

    @classmethod
    def _as_batch(
        cls,
        value: Any,
        *,
        expected_size: int | None = None,
    ) -> list[Any]:
        rank = cls._array_rank(value)
        if rank == 2:
            values = [value[index] for index in range(int(value.shape[0]))]
        elif rank == 1 or rank is None and not isinstance(value, (list, tuple)):
            values = [value]
        elif rank is not None:
            raise ValueError(
                "Raw ASR training audio must be rank 1, or rank 2 with "
                "shape (batch, samples).")
        elif value and all(isinstance(item, Real) for item in value):
            values = [value]
        else:
            values = list(value)

        if not values:
            raise ValueError("Raw ASR training audio batches cannot be empty.")
        if expected_size is not None and len(values) != expected_size:
            raise ValueError("Batched audio and transcription fields must have equal lengths.")
        return values

    @staticmethod
    def _plain_value(value: Any) -> Any:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                return tolist()
            except (RuntimeError, TypeError, ValueError):
                pass
        item = getattr(value, "item", None)
        if callable(item):
            try:
                return item()
            except (RuntimeError, TypeError, ValueError):
                pass
        return value

    @classmethod
    def _batch_scalar_values(
        cls,
        value: Any,
        *,
        batch_size: int,
        name: str,
        broadcast: bool,
    ) -> list[Any]:
        if value is None:
            return [None] * batch_size
        plain = cls._plain_value(value)
        if isinstance(plain, (list, tuple)):
            values = list(plain)
            if broadcast and len(values) == 1 and batch_size > 1:
                values *= batch_size
            if len(values) != batch_size:
                raise ValueError(f"Batched audio and `{name}` fields must have equal lengths.")
        else:
            values = [plain] * batch_size
        return [cls._plain_value(item) for item in values]

    @classmethod
    def _trim_audio_batch(
        cls,
        audio_values: list[Any],
        audio_lengths: Any,
    ) -> list[Any]:
        if audio_lengths is None:
            return audio_values
        lengths = cls._batch_scalar_values(
            audio_lengths,
            batch_size=len(audio_values),
            name="audio_lengths",
            broadcast=False,
        )
        trimmed = []
        for audio, length in zip(audio_values, lengths):
            if isinstance(length, bool) or not isinstance(length, Integral):
                raise TypeError("`audio_lengths` must contain positive integer sample counts.")
            length = int(length)
            shape = getattr(audio, "shape", None)
            sample_count = int(shape[-1]) if shape is not None else len(audio)
            if length <= 0 or length > sample_count:
                raise ValueError(
                    "Each `audio_lengths` value must be between 1 and the "
                    "corresponding padded waveform length.")
            trimmed.append(audio[:length])
        return trimmed

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Convert raw audio/text batches to native Transformers tensors."""
        del phase
        if "input_values" in inputs or "input_features" in inputs:
            return dict(inputs)
        audio = inputs.get("audio")
        if audio is None:
            return dict(inputs)
        if self.transformers_processor is None:
            raise RuntimeError("Training input preparation requires load_for_training().")

        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if text is not None and not isinstance(text, (str, list, tuple)):
            raise TypeError(
                "ASR training transcription must be a string, a sequence of "
                "strings, or None.")
        text_values = (list(text) if isinstance(text, (list, tuple)) and not isinstance(text, str) else None)
        candidate_texts = text_values if text_values is not None else ([] if text is None else [text])
        if any(not isinstance(value, str) or not value.strip() for value in candidate_texts):
            raise ValueError("ASR training transcriptions must contain non-empty strings.")
        expected_size = len(text_values) if text_values is not None else None
        audio_values = self._as_batch(audio, expected_size=expected_size)
        if text_values is None and text is not None and len(audio_values) != 1:
            raise ValueError("Batched ASR audio requires one transcription per waveform.")
        audio_values = self._trim_audio_batch(
            audio_values,
            inputs.get("audio_lengths"),
        )
        rate_values = self._batch_scalar_values(
            inputs.get("sampling_rate"),
            batch_size=len(audio_values),
            name="sampling_rate",
            broadcast=True,
        )
        waveforms = [
            load_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=self._processor_sample_rate(),
            ).waveform for value, rate in zip(audio_values, rate_values)
        ]
        processor_audio = waveforms if len(waveforms) > 1 else waveforms[0]
        batch = dict(
            self.transformers_processor(
                processor_audio,
                sampling_rate=self._processor_sample_rate(),
                padding=True,
                return_tensors="pt",
            ))

        if "labels" in inputs:
            batch["labels"] = inputs["labels"]
            return batch
        if text is None:
            return batch

        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            self.transformers_processor,
        )
        label_text = text_values if text_values is not None else text
        encoded_labels = tokenizer(
            label_text,
            padding=True,
            return_tensors="pt",
        )
        if not isinstance(encoded_labels, Mapping) or "input_ids" not in encoded_labels:
            raise TypeError("The Transformers ASR tokenizer did not return `input_ids`.")
        labels = encoded_labels["input_ids"]
        attention_mask = encoded_labels.get("attention_mask")
        if attention_mask is not None and hasattr(labels, "masked_fill"):
            not_attended = (attention_mask.ne(1) if hasattr(attention_mask, "ne") else attention_mask == 0)
            labels = labels.masked_fill(not_attended, -100)
        batch["labels"] = labels
        return batch

    def _validate_training_runtime(self) -> None:
        identifier = str(self.config.name_or_path).lower()
        if any(marker in identifier for marker in _SERVING_ONLY_MARKERS):
            raise ValueError(
                "Transformers ASR fine-tuning requires a differentiable "
                "PyTorch/safetensors checkpoint; optimized serving artifacts "
                "such as GGUF, ONNX, TensorRT, and Core ML are inference-only.")
        for name in _QUANTIZATION_KEYS:
            value = self.config.model_kwargs.get(name)
            enabled = value not in (None, False, "", {}, ())
            if enabled:
                raise ValueError(
                    "Transformers ASR fine-tuning requires an unquantized "
                    "native model. Remove "
                    f"`model_kwargs[{name!r}]` or register a specialized "
                    "quantization-aware training adapter.")

    def _prepare_for_training(self) -> None:
        self._pipeline = None
        self._pipeline_model = None
        super()._prepare_for_training()

    def _prepare_for_inference(self) -> None:
        super()._prepare_for_inference()
        self._pipeline = None
        self._pipeline_model = None

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        save_model = getattr(self.model, "save_pretrained", None)
        if callable(save_model):
            save_model(save_directory, safe_serialization=True)
        save_processor = getattr(
            self.transformers_processor,
            "save_pretrained",
            None,
        )
        if callable(save_processor):
            save_processor(save_directory)


__all__ = ["TransformersASRForSpeechRecognition"]
