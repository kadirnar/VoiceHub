"""Lazy, streaming-capable Sherpa-ONNX VAD wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.hub import resolve_pretrained_file
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.native_utils import resolve_native_device
from voicehub.models.vad_sherpa_onnx.configuration_vad_sherpa_onnx import SherpaONNXVADConfig
from voicehub.vad_utils import merge_speech_segments


@dataclass(frozen=True)
class _SherpaONNXRuntime:
    module: Any
    model_path: Path


def _drain_segments(detector, *, sample_rate: int) -> tuple[SpeechSegment, ...]:
    segments = []
    while not detector.empty():
        native = detector.front
        start = round(float(native.start) / sample_rate, 12)
        samples = native.samples
        end = round(start + len(samples) / sample_rate, 12)
        if end > start:
            segments.append(
                SpeechSegment(
                    start=max(0.0, start),
                    end=end,
                    score=None,
                    metadata={
                        "decision": "native",
                        "start_sample": int(native.start),
                        "num_samples": len(samples),
                    },
                ))
        detector.pop()
    return tuple(segments)


def _finalize_segments(
    values,
    *,
    duration: float,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    padding = speech_pad_ms / 1000
    padded = []
    for segment in values:
        start = max(0.0, segment.start - padding)
        end = min(duration, segment.end + padding)
        if end <= start:
            continue
        padded.append(
            SpeechSegment(
                start=start,
                end=end,
                score=segment.score,
                label=segment.label,
                channel=segment.channel,
                metadata=dict(segment.metadata),
            ))
    merged = merge_speech_segments(padded)
    if max_speech_duration_s is None:
        return merged

    split = []
    for segment in merged:
        cursor = segment.start
        tolerance = 1e-12
        while segment.end - cursor > max_speech_duration_s + tolerance:
            split_end = round(cursor + max_speech_duration_s, 12)
            split.append(
                SpeechSegment(
                    start=cursor,
                    end=split_end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
            cursor = split_end
        if segment.end - cursor > tolerance:
            split.append(
                SpeechSegment(
                    start=cursor,
                    end=segment.end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
    return tuple(split)


class SherpaONNXVADSession:
    """One request-local Sherpa detector with incremental push/flush state."""

    def __init__(
        self,
        model: SherpaONNXVADForVoiceActivityDetection,
        *,
        sampling_rate: int,
        inference_kwargs: dict[str, Any] | None = None,
    ):
        if sampling_rate != model.sample_rate:
            raise ValueError(f"Sherpa-ONNX streaming requires {model.sample_rate} Hz chunks.")
        self.model = model.load()
        self.sampling_rate = sampling_rate
        self.inference_kwargs = dict(inference_kwargs or {})
        self._validate_options()
        defaults = VADInferenceConfig().to_dict()
        defaults.update(self.inference_kwargs)
        self.inference_kwargs = VADInferenceConfig.from_dict(defaults).to_dict()
        self._detector = self.model._create_detector(**self.inference_kwargs)
        self._window_size = self.model._window_size(self.inference_kwargs)
        self._pending = None
        self._segments: list[SpeechSegment] = []
        self._sample_count = 0
        self._result = None
        self._closed = False
        self._lock = RLock()

    def _validate_options(self) -> None:
        allowed = {
            "threshold",
            "onset",
            "offset",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "speech_pad_ms",
            "max_speech_duration_s",
            "window_size_samples",
            "return_frames",
        }
        unknown = sorted(set(self.inference_kwargs) - allowed)
        if unknown:
            raise ValueError("Unsupported Sherpa-ONNX streaming option(s): "
                             f"{', '.join(unknown)}.")
        if self.inference_kwargs.get("return_frames", False):
            raise ValueError(
                "Sherpa-ONNX does not expose calibrated frame scores; "
                "use `return_frames=False`.")

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _feed(self, waveform) -> tuple[SpeechSegment, ...]:
        np = import_optional(
            "numpy",
            model_type=self.model.config.model_type,
            install_extra=None,
        )
        if self._pending is None:
            buffered = waveform
        else:
            buffered = np.concatenate((self._pending, waveform))
        cursor = 0
        while len(buffered) - cursor >= self._window_size:
            self._detector.accept_waveform(buffered[cursor:cursor + self._window_size])
            cursor += self._window_size
        self._pending = buffered[cursor:].copy()
        emitted = _drain_segments(
            self._detector,
            sample_rate=self.sampling_rate,
        )
        self._segments.extend(emitted)
        return emitted

    def push(self, audio_chunk: Any) -> tuple[SpeechSegment, ...]:
        """Feed one chunk and return any newly completed native segments."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push audio to a closed stream.")
            if self._result is not None:
                raise RuntimeError("Reset the stream before pushing after flush().")
            chunk = load_audio(
                audio_chunk,
                sampling_rate=self.sampling_rate,
                target_sampling_rate=self.sampling_rate,
            )
            self._sample_count += len(chunk.waveform)
            return self._feed(chunk.waveform)

    def flush(self) -> VADOutput:
        """Finalize the detector exactly once and return all speech regions."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot flush a closed stream.")
            if self._result is not None:
                return self._result
            if self._sample_count == 0:
                raise ValueError("Cannot flush a stream that has no audio.")
            if self._pending is not None and len(self._pending):
                np = import_optional(
                    "numpy",
                    model_type=self.model.config.model_type,
                    install_extra=None,
                )
                padded = np.pad(
                    self._pending,
                    (0, self._window_size - len(self._pending)),
                )
                self._detector.accept_waveform(padded)
                self._pending = None
            self._detector.flush()
            self._segments.extend(_drain_segments(
                self._detector,
                sample_rate=self.sampling_rate,
            ))
            duration = self._sample_count / self.sampling_rate
            segments = _finalize_segments(
                self._segments,
                duration=duration,
                speech_pad_ms=self.inference_kwargs.get("speech_pad_ms", 30),
                max_speech_duration_s=self.inference_kwargs.get("max_speech_duration_s"),
            )
            self._result = VADOutput(
                segments=segments,
                duration=duration,
                sample_rate=self.sampling_rate,
                probabilities=None,
                metadata=self.model._output_metadata(window_size_samples=self._window_size),
            )
            return self._result

    def reset(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed stream.")
            reset = getattr(self._detector, "reset", None)
            if callable(reset):
                reset()
            else:
                self._detector = self.model._create_detector(**self.inference_kwargs)
            self._pending = None
            self._segments.clear()
            self._sample_count = 0
            self._result = None

    def close(self) -> None:
        with self._lock:
            self._pending = None
            self._segments.clear()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed stream.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class SherpaONNXVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run Silero or TEN VAD through Sherpa's optimized ONNX runtime."""

    config_class = SherpaONNXVADConfig
    default_model_name_or_path = "csukuangfj/vad"
    training_support = "inference-only"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: SherpaONNXVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs,
    ):
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        lifecycle_device = "cuda" if config.provider == "cuda" else "cpu"
        if device == "auto":
            device = lifecycle_device
        requested_device = device.partition(":")[0].lower()
        if requested_device != lifecycle_device:
            raise ValueError(
                "`device` is incompatible with the configured Sherpa provider; "
                f"provider={config.provider!r} uses device={lifecycle_device!r}, "
                f"but received {device!r}.")
        super().__init__(config, device=device, lazy_load=lazy_load)
        self._auth_token = token

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_native_device(
            device,
            provider="Sherpa-ONNX VAD",
            supported_types=("cpu", "cuda"),
            allow_device_index=False,
        )

    def _load_pretrained_model(self) -> None:
        sherpa_onnx = import_optional(
            "sherpa_onnx",
            model_type=self.config.model_type,
            install_extra=None,
        )
        for name in ("VadModelConfig", "VoiceActivityDetector"):
            if not callable(getattr(sherpa_onnx, name, None)):
                raise RuntimeError("The installed sherpa-onnx package does not expose "
                                   f"{name}.")
        source = self.config.name_or_path or self.default_model_name_or_path
        local_source = Path(source).expanduser()
        if local_source.is_file():
            if local_source.suffix.lower() != ".onnx":
                raise ValueError("A direct Sherpa VAD checkpoint must be an .onnx file.")
            if self.config.subfolder:
                raise ValueError("`subfolder` cannot be used with a direct local ONNX file.")
            if self.config.revision is not None:
                raise ValueError("`revision` cannot be used with a direct local ONNX file.")
            model_path = local_source.resolve()
        else:
            model_path = resolve_pretrained_file(
                source,
                self.config.model_filename,
                subfolder=self.config.subfolder,
                cache_dir=self.config.cache_dir,
                revision=self.config.revision,
                token=self._auth_token,
                local_files_only=self.config.local_files_only,
            )
        self.model = _SherpaONNXRuntime(
            module=sherpa_onnx,
            model_path=model_path,
        )

    def _window_size(self, options: dict[str, Any]) -> int:
        window_size = options.get("window_size_samples")
        if window_size is None:
            window_size = self.config.window_size_samples
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("`window_size_samples` must be a positive integer.")
        return window_size

    def _create_detector(
        self,
        *,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ):
        del speech_pad_ms
        if return_frames:
            raise ValueError(
                "Sherpa-ONNX does not expose calibrated frame scores; "
                "use `return_frames=False`.")
        if self.config.model_family == "ten" and offset is not None:
            raise ValueError("Sherpa TEN VAD does not expose a separate offset threshold.")
        config = self.model.module.VadModelConfig()
        family = getattr(config, f"{self.config.model_family}_vad")
        family.model = str(self.model.model_path)
        family.threshold = threshold if onset is None else onset
        if self.config.model_family == "silero" and offset is not None:
            family.neg_threshold = offset
        family.min_silence_duration = min_silence_duration_ms / 1000
        family.min_speech_duration = min_speech_duration_ms / 1000
        family.max_speech_duration = (
            self.config.buffer_size_s if max_speech_duration_s is None else max_speech_duration_s)
        family.window_size = self._window_size({
            "window_size_samples": window_size_samples,
        } if window_size_samples is not None else {})
        config.sample_rate = self.sample_rate
        config.num_threads = self.config.num_threads
        config.provider = self.config.provider
        config.debug = self.config.debug
        validate = getattr(config, "validate", None)
        if callable(validate) and not validate():
            raise RuntimeError("Sherpa-ONNX rejected the configured VAD artifact.")
        return self.model.module.VoiceActivityDetector(
            config,
            buffer_size_in_seconds=self.config.buffer_size_s,
        )

    def _output_metadata(self, *, window_size_samples: int) -> dict[str, Any]:
        return {
            "backend": "sherpa-onnx",
            "model_family": self.config.model_family,
            "model_path": str(self.model.model_path),
            "provider": self.config.provider,
            "window_size_samples": window_size_samples,
            "frame_scores_available": False,
        }

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
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        options = {
            "threshold": threshold,
            "onset": onset,
            "offset": offset,
            "min_speech_duration_ms": min_speech_duration_ms,
            "min_silence_duration_ms": min_silence_duration_ms,
            "speech_pad_ms": speech_pad_ms,
            "max_speech_duration_s": max_speech_duration_s,
            "window_size_samples": window_size_samples,
            "return_frames": return_frames,
        }
        session = SherpaONNXVADSession(
            self,
            sampling_rate=materialized.sampling_rate,
            inference_kwargs=options,
        )
        with session:
            session.push(materialized.waveform)
            return session.flush()

    def stream(
        self,
        *,
        sampling_rate: int,
        **inference_kwargs,
    ) -> SherpaONNXVADSession:
        """Create an isolated incremental Sherpa detector."""
        return SherpaONNXVADSession(
            self,
            sampling_rate=sampling_rate,
            inference_kwargs=inference_kwargs,
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "Sherpa-ONNX artifacts are optimized inference graphs. Fine-tune "
            "the corresponding source model before exporting a new ONNX artifact.")
