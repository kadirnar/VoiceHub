"""Lazy native pyannote.audio VAD pipeline wrapper."""

from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_pyannote.configuration_vad_pyannote import PyannoteVADConfig
from voicehub.vad_utils import merge_speech_segments, normalize_backend_segments


def _finalize_segments(
    values,
    *,
    duration: float,
    sample_rate: int,
    speech_pad_ms: int,
    max_speech_duration_s: float | None,
) -> tuple[SpeechSegment, ...]:
    normalized = normalize_backend_segments(
        values,
        sampling_rate=sample_rate,
    )
    padding = speech_pad_ms / 1000
    padded = []
    for segment in normalized:
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
        start = segment.start
        tolerance = 1e-12
        while segment.end - start > max_speech_duration_s + tolerance:
            split_end = round(start + max_speech_duration_s, 12)
            split.append(
                SpeechSegment(
                    start=start,
                    end=split_end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
            start = split_end
        if segment.end - start > tolerance:
            split.append(
                SpeechSegment(
                    start=start,
                    end=segment.end,
                    score=segment.score,
                    label=segment.label,
                    channel=segment.channel,
                    metadata=dict(segment.metadata),
                ))
    return tuple(split)


def _timeline_values(output) -> tuple[Any, ...]:
    """Extract timeline segments across pyannote 2.x and 4.x outputs."""
    for attribute in (
            "voice_activity_detection",
            "speech_activity_detection",
            "annotation",
    ):
        candidate = getattr(output, attribute, None)
        if candidate is not None:
            output = candidate
            break
    get_timeline = getattr(output, "get_timeline", None)
    if callable(get_timeline):
        output = get_timeline()
    support = getattr(output, "support", None)
    if callable(support):
        output = support()
    try:
        return tuple(output)
    except TypeError as exc:
        raise TypeError(
            "pyannote VAD must return an Annotation, Timeline, or iterable "
            "of segments.") from exc


class PyannoteVADForVoiceActivityDetection(PreTrainedVADModel):
    """Normalize native pyannote VAD pipeline output into VoiceHub segments."""

    config_class = PyannoteVADConfig
    default_model_name_or_path = "pyannote/voice-activity-detection"
    training_support = "upstream-custom"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: PyannoteVADConfig | str | Path | None = None,
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
        self._auth_token = token

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="pyannote.audio VAD")

    def _load_pretrained_model(self) -> None:
        pyannote_audio = import_optional(
            "pyannote.audio",
            model_type=self.config.model_type,
            install_extra="asr-vad",
        )
        pipeline_class = getattr(pyannote_audio, "Pipeline", None)
        loader = getattr(pipeline_class, "from_pretrained", None)
        if not callable(loader):
            raise RuntimeError(
                "The installed pyannote.audio package does not expose "
                "Pipeline.from_pretrained().")

        try:
            parameters = signature(loader).parameters
        except (TypeError, ValueError):
            parameters = {}
            accepts_kwargs = True
        else:
            accepts_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
        options = dict(self.config.pipeline_kwargs)
        configured = {
            "revision": self.config.revision,
            "subfolder": self.config.subfolder,
            "cache_dir": self.config.cache_dir,
        }
        for name, value in configured.items():
            if value is None:
                continue
            if name not in parameters and not accepts_kwargs:
                raise RuntimeError(
                    f"The installed pyannote.audio version does not support "
                    f"the `{name}` loading option.")
            options.setdefault(name, value)
        if self._auth_token is not None:
            if "token" in parameters:
                options["token"] = self._auth_token
            elif "use_auth_token" in parameters:
                options["use_auth_token"] = self._auth_token
            elif accepts_kwargs:
                options["token"] = self._auth_token
            else:
                raise RuntimeError(
                    "The installed pyannote.audio version cannot accept an "
                    "authentication token.")

        source = self.config.name_or_path or self.default_model_name_or_path
        pipeline = loader(source, **options)
        if pipeline is None:
            raise RuntimeError(f"pyannote.audio could not load the VAD pipeline from {source!r}.")
        if self.device != "cpu":
            move = getattr(pipeline, "to", None)
            if callable(move):
                torch = import_optional(
                    "torch",
                    model_type=self.config.model_type,
                    install_extra="asr-vad",
                )
                move(torch.device(self.device))
        self.model = pipeline

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
                "The native pyannote pipeline does not expose calibrated "
                "frame scores; use `return_frames=False`.")
        if window_size_samples is not None:
            raise ValueError(
                "pyannote controls its analysis window in the pipeline "
                "artifact; `window_size_samples` is not supported.")
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra="asr-vad",
        )
        waveform = torch.as_tensor(materialized.waveform)
        if getattr(waveform, "ndim", len(getattr(waveform, "shape", ()))) == 1:
            waveform = waveform.unsqueeze(0)
        instantiate = getattr(self.model, "instantiate", None)
        parameters = {
            "onset": threshold if onset is None else onset,
            "offset": threshold if offset is None else offset,
            "min_duration_on": min_speech_duration_ms / 1000,
            "min_duration_off": min_silence_duration_ms / 1000,
        }
        if callable(instantiate):
            instantiate(parameters)
        native_output = self.model({
            "waveform": waveform,
            "sample_rate": materialized.sampling_rate,
        })
        segments = _finalize_segments(
            _timeline_values(native_output),
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            metadata={
                "backend": "pyannote",
                "source": self.config.name_or_path or self.default_model_name_or_path,
                "frame_scores_available": False,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "pyannote VAD training is upstream-custom and requires a "
            "pyannote.audio task/protocol recipe. VoiceHub's generic "
            "fine-tuning adapter is intentionally unavailable.")
