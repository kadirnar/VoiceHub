"""Lazy FunASR FSMN voice activity detection wrapper."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.native_utils import resolve_native_device
from voicehub.models.vad_funasr.configuration_vad_funasr import FunASRVADConfig
from voicehub.vad_utils import merge_speech_segments

_MILLISECONDS_TO_SECONDS = 0.001


def _clean_time(value: float) -> float:
    """Remove insignificant floating-point noise from millisecond
    arithmetic."""
    return round(value, 12)


def _is_boundary(value: Any) -> bool:
    if isinstance(value, Mapping):
        return ("start" in value and ("end" in value or "stop" in value))
    if isinstance(value, (str, bytes)):
        return False
    try:
        return len(value) >= 2
    except TypeError:
        return False


def _result_boundaries(result: Any) -> tuple[Any, tuple[str, ...]]:
    """Extract the single-utterance boundary list from FunASR output."""
    if (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping)):
        result = result[0]

    if isinstance(result, Mapping):
        raw_keys = tuple(sorted(str(key) for key in result))
        if "value" in result:
            values = result["value"]
            return (() if values is None else values), raw_keys
        if "segments" in result:
            values = result["segments"]
            return (() if values is None else values), raw_keys
        raise ValueError("FunASR VAD output must contain a `value` or `segments` field.")

    if isinstance(result, (tuple, list)):
        if not result:
            return (), ()
        if all(_is_boundary(value) for value in result):
            return result, ()
        if len(result) != 1:
            raise ValueError("FunASR VAD returned multiple utterances for one audio input.")
        return _result_boundaries(result[0])

    raise TypeError("FunASR VAD output must be a result mapping or boundary sequence.")


def _boundary_values(value: Any) -> tuple[float, float]:
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end", value.get("stop"))
    else:
        try:
            start, end = value[0], value[1]
        except (IndexError, KeyError, TypeError) as exc:
            raise TypeError("FunASR VAD boundaries must contain start/end pairs.") from exc
    for name, timestamp in (("start", start), ("end", end)):
        if hasattr(timestamp, "item"):
            timestamp = timestamp.item()
        if (isinstance(timestamp, bool) or not isinstance(timestamp, Real) or not isfinite(float(timestamp))):
            raise TypeError(f"FunASR VAD boundary `{name}` must be a finite number.")
        if name == "start":
            start = float(timestamp)
        else:
            end = float(timestamp)
    return start, end


def _normalize_boundaries(
    values,
    *,
    duration: float,
) -> tuple[SpeechSegment, ...]:
    segments = []
    for value in values:
        start_ms, end_ms = _boundary_values(value)
        if start_ms < 0 or end_ms < 0:
            raise RuntimeError(
                "FunASR returned an incomplete streaming boundary during "
                "offline inference.")
        if end_ms < start_ms:
            raise ValueError("FunASR returned a VAD boundary whose end precedes its start.")
        start = _clean_time(min(duration, start_ms * _MILLISECONDS_TO_SECONDS))
        end = _clean_time(min(duration, end_ms * _MILLISECONDS_TO_SECONDS))
        if end <= start:
            continue
        segments.append(SpeechSegment(start=start, end=end))
    return merge_speech_segments(segments)


def _postprocess_segments(
    segments: tuple[SpeechSegment, ...],
    *,
    duration: float,
    min_speech_duration_ms: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    minimum_speech = min_speech_duration_ms / 1000
    retained = tuple(segment for segment in segments if segment.end - segment.start >= minimum_speech)
    joined = merge_speech_segments(
        retained,
        max_gap=min_silence_duration_ms / 1000,
    )
    padding = speech_pad_ms / 1000
    padded = tuple(
        SpeechSegment(
            start=_clean_time(max(0.0, segment.start - padding)),
            end=_clean_time(min(duration, segment.end + padding)),
            score=segment.score,
            label=segment.label,
            channel=segment.channel,
            metadata=dict(segment.metadata),
        ) for segment in joined if min(duration, segment.end + padding) > max(0.0, segment.start - padding))
    padded = merge_speech_segments(padded)
    if max_speech_duration_s is None:
        return padded

    split = []
    for segment in padded:
        cursor = segment.start
        tolerance = 1e-12
        while segment.end - cursor > max_speech_duration_s + tolerance:
            split_end = _clean_time(cursor + max_speech_duration_s)
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


class FunASRVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run FunASR FSMN VAD and expose normalized second-based boundaries."""

    config_class = FunASRVADConfig
    default_model_name_or_path = "fsmn-vad"
    training_support = "upstream-custom"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: FunASRVADConfig | str | Path | None = None,
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
        super().__init__(config, device=device, lazy_load=lazy_load)
        self._token = token

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_native_device(
            device,
            provider="FunASR VAD",
            supported_types=("cpu", "cuda", "xpu", "npu"),
        )

    def _load_pretrained_model(self) -> None:
        funasr = import_optional(
            "funasr",
            model_type=self.config.model_type,
            install_extra=None,
        )
        auto_model = getattr(funasr, "AutoModel", None)
        if not callable(auto_model):
            raise RuntimeError("The installed FunASR package does not expose AutoModel.")
        source = self.config.name_or_path or self.default_model_name_or_path
        options = dict(self.config.model_kwargs)
        options.update({
            "model": source,
            "device": self.device,
            "hub": self.config.hub,
            "trust_remote_code": self.config.trust_remote_code,
            "disable_update": self.config.disable_update,
            "disable_pbar": self.config.disable_pbar,
        })
        if self.config.revision is not None:
            options["model_revision"] = self.config.revision
        if self.config.ncpu is not None:
            options["ncpu"] = self.config.ncpu
        if self._token is not None:
            options["token"] = self._token
        model = auto_model(**options)
        if model is None or not callable(getattr(model, "generate", None)):
            raise RuntimeError(f"FunASR could not load a VAD runtime from {source!r}.")
        self.model = model

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
        if return_frames:
            raise ValueError(
                "FunASR's public FSMN result does not expose frame scores; "
                "use `return_frames=False`.")
        if window_size_samples is not None:
            raise ValueError(
                "FunASR controls analysis geometry in the model artifact; "
                "`window_size_samples` is not supported.")
        effective_threshold = threshold if onset is None else onset
        if offset is not None and not isclose(
                offset,
                effective_threshold,
                rel_tol=0.0,
                abs_tol=1e-12,
        ):
            raise ValueError(
                "FunASR FSMN exposes one speech/noise threshold and cannot "
                "apply independent `onset` and `offset` values.")

        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        options = dict(self.config.generate_kwargs)
        options.update({
            "cache": {},
            "fs": materialized.sampling_rate,
            "is_final": True,
            "max_end_silence_time": min_silence_duration_ms,
            "speech_noise_thres": effective_threshold,
        })
        if max_speech_duration_s is not None:
            options["max_single_segment_time"] = round(max_speech_duration_s * 1000)
        result = self.model.generate(
            input=materialized.waveform,
            **options,
        )
        boundaries, raw_keys = _result_boundaries(result)
        segments = _normalize_boundaries(
            boundaries,
            duration=materialized.duration,
        )
        segments = _postprocess_segments(
            segments,
            duration=materialized.duration,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            metadata={
                "backend": "funasr",
                "architecture_family": "fsmn",
                "source": (self.config.name_or_path or self.default_model_name_or_path),
                "hub": self.config.hub,
                "native_timestamp_unit": "milliseconds",
                "frame_scores_available": False,
                "raw_keys": raw_keys,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "FunASR FSMN VAD training is upstream-custom and requires its "
            "configuration-driven training runner and data manifests. "
            "VoiceHub's generic fine-tuning adapter is intentionally "
            "unavailable.")
