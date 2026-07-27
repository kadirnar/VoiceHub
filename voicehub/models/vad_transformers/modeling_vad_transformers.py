"""Universal Transformers checkpoint provider for neural VAD."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from re import findall
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.errors import OptionalDependencyError
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.modeling_outputs import VADOutput
from voicehub.models.vad_transformers.configuration_vad_transformers import TransformersVADConfig
from voicehub.vad_utils import frame_probabilities_to_segments

_SERVING_ONLY_MARKERS = (
    ".gguf",
    "-gguf",
    "/gguf",
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
_NON_VAD_ARCHITECTURE_MARKERS = (
    "forctc",
    "forrnnt",
    "fortdt",
    "forspeechseq2seq",
    "speechencoderdecoder",
)
_NEGATIVE_SPEECH_LABEL_TOKENS = frozenset({
    "background",
    "inactive",
    "music",
    "no",
    "noise",
    "non",
    "not",
    "silence",
    "silent",
})


class TransformersVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run and fine-tune compatible Transformers VAD checkpoints."""

    config_class = TransformersVADConfig
    default_model_name_or_path = ""

    def __init__(
        self,
        config: TransformersVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        config.validate()
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        super().__init__(config, device=device, lazy_load=lazy_load)
        self.native_config = None
        self.feature_extractor = None
        self.architecture_family = config.architecture_family
        self._token = token

    @staticmethod
    def _infer_architecture_family(native_config: Any) -> str:
        architectures = getattr(native_config, "architectures", ()) or ()
        if isinstance(architectures, str):
            architectures = (architectures, )
        for architecture in architectures:
            architecture_name = str(architecture)
            normalized = (architecture_name.replace("_", "").replace("-", "").lower())
            if "foraudioframeclassification" in normalized:
                return "frame-classification"
            if "foraudioclassification" in normalized:
                return "audio-classification"
            if any(marker in normalized for marker in _NON_VAD_ARCHITECTURE_MARKERS):
                raise ValueError(
                    f"Checkpoint architecture {architecture_name!r} is an ASR "
                    "head, not a VAD classifier. Use "
                    "AutoModelForSpeechRecognition instead.")

        auto_map = getattr(native_config, "auto_map", {}) or {}
        if isinstance(auto_map, Mapping):
            for auto_class_name in auto_map:
                normalized = str(auto_class_name).replace("_", "").lower()
                if normalized.endswith("automodelforaudioframeclassification"):
                    return "frame-classification"
                if normalized.endswith("automodelforaudioclassification"):
                    return "audio-classification"
                if normalized.endswith((
                        "automodelforctc",
                        "automodelforrnnt",
                        "automodelfortdt",
                        "automodelforspeechseq2seq",
                )):
                    raise ValueError(
                        "This checkpoint advertises an ASR auto class, not a "
                        "VAD classifier. Use AutoModelForSpeechRecognition.")

        model_type = getattr(native_config, "model_type", None)
        raise ValueError(
            "Transformers could not determine whether this checkpoint uses "
            "audio- or frame-classification" +
            (f" from model_type {model_type!r}" if model_type is not None else "") +
            ". Shared base model types such as 'wav2vec2' are task-"
            "ambiguous; set `architecture_family` explicitly or publish an "
            "ASR/VAD-specific `architectures` entry.")

    @staticmethod
    def _local_weight_file(name_or_path: str | Path) -> Path | None:
        path = Path(name_or_path).expanduser()
        if path.is_file() and path.suffix.lower() == ".safetensors":
            return path.resolve()
        return None

    def _model_source(self) -> str:
        source = self.config.name_or_path or self.default_model_name_or_path
        weight_file = self._local_weight_file(source)
        return str(weight_file.parent) if weight_file is not None else str(source)

    def _config_source(self) -> str:
        return str(self.config.config_name_or_path or self._model_source())

    def _processor_source(self) -> str:
        return str(self.config.processor_name_or_path or self._config_source())

    def _hub_kwargs(self) -> dict[str, Any]:
        return {
            name: value
            for name, value in {
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "token": self._token,
            }.items() if value is not None
        }

    def _direct_state_dict(self) -> Mapping[str, Any] | None:
        weight_file = self._local_weight_file(self.config.name_or_path)
        if weight_file is None:
            return None
        safetensors = import_optional(
            "safetensors.torch",
            model_type=self.config.model_type,
            install_extra="vad-transformers",
        )
        state_dict = safetensors.load_file(str(weight_file), device="cpu")
        if not isinstance(state_dict, Mapping):
            raise TypeError("The safetensors loader did not return a state-dict mapping.")
        return state_dict

    def _load_pretrained_model(self) -> None:
        source = self._model_source()
        if not source:
            raise ValueError(
                "`vad_transformers` is a checkpoint-family provider. Pass a "
                "compatible binary audio- or frame-classification checkpoint "
                "to from_pretrained().")
        transformers = import_optional(
            "transformers",
            model_type=self.config.model_type,
            install_extra="vad-transformers",
        )
        self.native_config = transformers.AutoConfig.from_pretrained(
            self._config_source(),
            trust_remote_code=self.config.trust_remote_code,
            **self._hub_kwargs(),
        )
        family = self.config.architecture_family
        if family == "auto":
            family = self._infer_architecture_family(self.native_config)
        if family == "frame-classification":
            model_class = getattr(
                transformers,
                "AutoModelForAudioFrameClassification",
                None,
            )
            if model_class is None:
                raise OptionalDependencyError(
                    "This frame-classification checkpoint requires a newer "
                    "Transformers release exposing "
                    "AutoModelForAudioFrameClassification.")
        else:
            model_class = getattr(
                transformers,
                "AutoModelForAudioClassification",
                None,
            )
            if model_class is None:
                raise OptionalDependencyError(
                    "'vad_transformers' requires a Transformers release "
                    "exposing `AutoModelForAudioClassification`. Upgrade "
                    "`voicehub[vad-transformers]` and retry.")

        model_options = {
            **self._hub_kwargs(),
            **self.config.model_kwargs,
            "config": self.native_config,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.use_safetensors is not None:
            model_options["use_safetensors"] = self.config.use_safetensors
        state_dict = self._direct_state_dict()
        if state_dict is not None:
            model_options["state_dict"] = state_dict
        self.model = model_class.from_pretrained(source, **model_options)
        processor_class = getattr(transformers, "AutoFeatureExtractor", None)
        if processor_class is None:
            processor_class = getattr(transformers, "AutoProcessor", None)
        if processor_class is None:
            raise OptionalDependencyError(
                "'vad_transformers' requires a Transformers release exposing "
                "`AutoFeatureExtractor` or `AutoProcessor`. Upgrade "
                "`voicehub[vad-transformers]` and retry.")
        self.feature_extractor = processor_class.from_pretrained(
            self._processor_source(),
            trust_remote_code=self.config.trust_remote_code,
            **self._hub_kwargs(),
            **self.config.processor_kwargs,
        )
        self.architecture_family = family
        has_device_map = (
            "device_map" in self.config.model_kwargs or bool(getattr(self.model, "hf_device_map", None)))
        if not has_device_map:
            move = getattr(self.model, "to", None)
            if callable(move):
                moved = move(self.device)
                if moved is not None:
                    self.model = moved
        self.config.sample_rate = self._processor_sample_rate()

    def _speech_class_id(self, class_count: int) -> int:
        if (isinstance(class_count, bool) or not isinstance(class_count, Integral) or class_count <= 0):
            raise ValueError("VAD logits must expose at least one class.")
        class_count = int(class_count)
        configured = self.config.speech_class_id
        if configured is not None:
            if configured >= class_count:
                raise ValueError(f"`speech_class_id` {configured} is outside {class_count} classes.")
            return configured
        if class_count == 1:
            # A single sigmoid logit represents the positive class.
            return 0
        id2label = getattr(self.native_config, "id2label", {}) or {}
        labels = self.config.speech_labels
        for class_id, label in id2label.items():
            normalized = str(label).strip().lower()
            label_tokens = frozenset(findall(r"[a-z0-9]+", normalized))
            is_exact_match = normalized in labels
            is_positive_token_match = (
                not label_tokens.intersection(_NEGATIVE_SPEECH_LABEL_TOKENS) and
                any(token in label_tokens for token in labels))
            if is_exact_match or is_positive_token_match:
                class_id = int(class_id)
                if not 0 <= class_id < class_count:
                    raise ValueError(
                        "Checkpoint `id2label` contains a class index outside "
                        "the logits dimension.")
                return class_id
        if class_count == 2:
            return 1
        raise ValueError(
            "Could not identify the speech class from checkpoint labels. "
            "Set `speech_class_id` in TransformersVADConfig.")

    def _processor_sample_rate(self) -> int:
        value = getattr(
            self.feature_extractor,
            "sampling_rate",
            self.config.sample_rate,
        )
        if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
                float(value) <= 0 or not float(value).is_integer()):
            raise ValueError("The Transformers VAD processor reported an invalid sampling rate.")
        return int(value)

    def _frame_geometry(
        self,
        *,
        frame_count: int,
        waveform_samples: int,
    ) -> tuple[int, int]:
        if (isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count <= 0):
            raise ValueError("Frame-classification logits must contain at least one frame.")
        ratio = getattr(self.native_config, "inputs_to_logits_ratio", None)
        if ratio is not None:
            if (isinstance(ratio, bool) or not isinstance(ratio, Real) or not isfinite(float(ratio)) or
                    ratio <= 0):
                raise ValueError(
                    "Checkpoint `inputs_to_logits_ratio` must be finite and "
                    "greater than zero.")
            frame_hop = max(1, round(float(ratio)))
            return frame_hop, frame_hop
        frame_hop = max(1, round(waveform_samples / int(frame_count)))
        return frame_hop, frame_hop

    @staticmethod
    def _model_inputs(batch, device: str) -> dict[str, Any]:
        values = dict(batch)
        for key, value in values.items():
            if hasattr(value, "to"):
                values[key] = value.to(device)
        return values

    def _probabilities(self, logits):
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra="vad-transformers",
        )
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ) -> VADOutput:
        target_rate = self._processor_sample_rate()
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_rate,
        )
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra="vad-transformers",
        )
        context = torch.inference_mode() if hasattr(torch, "inference_mode") else nullcontext()
        if self.architecture_family == "frame-classification":
            batch = self.feature_extractor(
                materialized.waveform,
                sampling_rate=target_rate,
                return_tensors="pt",
            )
            with context:
                outputs = self.model(**self._model_inputs(batch, self.device))
            probabilities = self._probabilities(outputs.logits)[0]
            class_count = probabilities.shape[-1]
            speech_id = self._speech_class_id(class_count)
            frame_scores = probabilities[..., speech_id]
            frame_count = int(frame_scores.shape[0])
            frame_hop, frame_length = self._frame_geometry(
                frame_count=frame_count,
                waveform_samples=len(materialized.waveform),
            )
        else:
            frame_length = (
                window_size_samples if window_size_samples is not None else round(
                    self.config.window_duration_s * target_rate))
            frame_hop = round(self.config.hop_duration_s * target_rate)
            if frame_length <= 0 or frame_hop <= 0:
                raise ValueError(
                    "VAD window and hop durations must each resolve to at "
                    "least one audio sample.")
            windows = []
            np = import_optional(
                "numpy",
                model_type=self.config.model_type,
                install_extra="vad-transformers",
            )
            for start in range(0, len(materialized.waveform), frame_hop):
                window = materialized.waveform[start:start + frame_length]
                if len(window) < frame_length:
                    window = np.pad(window, (0, frame_length - len(window)))
                windows.append(window)
                if start + frame_length >= len(materialized.waveform):
                    break
            batch = self.feature_extractor(
                windows,
                sampling_rate=target_rate,
                padding=True,
                return_tensors="pt",
            )
            with context:
                outputs = self.model(**self._model_inputs(batch, self.device))
            probabilities = self._probabilities(outputs.logits)
            speech_id = self._speech_class_id(probabilities.shape[-1])
            frame_scores = probabilities[..., speech_id]

        cpu_scores = frame_scores.detach().float().cpu()
        score_values = cpu_scores.tolist()
        postprocessing = VADInferenceConfig(
            threshold=threshold,
            onset=onset,
            offset=offset,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        segments = frame_probabilities_to_segments(
            score_values,
            sampling_rate=target_rate,
            frame_hop_samples=frame_hop,
            frame_length_samples=frame_length,
            duration_samples=len(materialized.waveform),
            config=postprocessing,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=target_rate,
            probabilities=cpu_scores if return_frames else None,
            metadata={
                "backend": "transformers",
                "architecture_family": self.architecture_family,
                "speech_class_id": speech_id,
                "frame_hop_samples": frame_hop,
                "frame_length_samples": frame_length,
            },
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
    def _audio_batch(cls, value: Any) -> list[Any]:
        rank = cls._array_rank(value)
        if rank == 2:
            values = [value[index] for index in range(int(value.shape[0]))]
        elif rank == 1 or (rank is None and not isinstance(value, (list, tuple))):
            values = [value]
        elif rank is not None:
            raise ValueError(
                "Raw VAD training audio must be rank 1, or rank 2 with "
                "shape (batch, samples).")
        elif value and all(isinstance(item, Real) for item in value):
            values = [value]
        else:
            values = list(value)
        if not values:
            raise ValueError("Raw VAD training audio batches cannot be empty.")
        return values

    @staticmethod
    def _plain_value(value: Any) -> Any:
        for method_name in ("detach", "cpu"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
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
            sample_count = int(audio.shape[-1]) if hasattr(audio, "shape") else len(audio)
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
        del phase
        if "input_values" in inputs or "input_features" in inputs:
            return dict(inputs)
        audio = inputs.get("audio")
        if audio is None:
            return dict(inputs)
        if self.feature_extractor is None:
            raise RuntimeError("Training input preparation requires load_for_training().")
        values = self._trim_audio_batch(
            self._audio_batch(audio),
            inputs.get("audio_lengths"),
        )
        sampling_rates = self._batch_scalar_values(
            inputs.get("sampling_rate"),
            batch_size=len(values),
            name="sampling_rate",
            broadcast=True,
        )
        materialized = [
            load_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=self._processor_sample_rate(),
            ).waveform for value, rate in zip(values, sampling_rates)
        ]
        batch = dict(
            self.feature_extractor(
                materialized,
                sampling_rate=self._processor_sample_rate(),
                padding=True,
                return_tensors="pt",
            ))
        for name in (
                "labels",
                "loss_mask",
                "label_mask",
                "labels_mask",
                "frame_mask",
                "valid_frames",
        ):
            if name in inputs:
                batch[name] = inputs[name]
        return batch

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_directory, safe_serialization=True)
        if hasattr(self.feature_extractor, "save_pretrained"):
            self.feature_extractor.save_pretrained(save_directory)

    def _validate_training_runtime(self) -> None:
        identifier = str(self.config.name_or_path).lower()
        if any(marker in identifier for marker in _SERVING_ONLY_MARKERS):
            raise ValueError(
                "Transformers VAD fine-tuning requires a differentiable "
                "PyTorch/safetensors checkpoint; optimized serving artifacts "
                "such as GGUF, ONNX, TensorRT, and Core ML are inference-only.")
        for name in _QUANTIZATION_KEYS:
            value = self.config.model_kwargs.get(name)
            if value not in (None, False, "", {}, ()):
                raise ValueError(
                    "Transformers VAD fine-tuning requires an unquantized "
                    f"native model. Remove `model_kwargs[{name!r}]` or "
                    "register a quantization-aware training adapter.")
