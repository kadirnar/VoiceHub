"""Pyannote Brouhaha multi-task frame inference adapter."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any

from voicehub.audio import load_audio
from voicehub.dependencies import import_optional
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.vad_pyannote.modeling_vad_pyannote import PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_brouhaha.configuration_vad_pyannote_brouhaha import PyannoteBrouhahaVADConfig
from voicehub.vad_utils import frame_probabilities_to_segments, merge_speech_segments


@dataclass(frozen=True)
class _BrouhahaRuntime:
    model: Any
    inference: Any


_ARCHITECTURE_IMPORT_LOCK = RLock()
_MISSING = object()


@contextmanager
def _brouhaha_architecture_import():
    """Make the checkpoint's ``brouhaha.models`` import available while
    loading.

    The original ``brouhaha`` distribution is not published on PyPI and
    pins an older pyannote runtime. Prefer it when explicitly installed;
    otherwise, expose VoiceHub's compatible, MIT-licensed architecture
    module just for the duration of checkpoint deserialization.
    """
    with _ARCHITECTURE_IMPORT_LOCK:
        try:
            import_module("brouhaha.models")
        except ModuleNotFoundError as exc:
            if exc.name not in {"brouhaha", "brouhaha.models"}:
                raise
        else:
            yield
            return

        compatibility_module = import_module("voicehub.models.vad_pyannote_brouhaha._architecture")
        package = sys.modules.get("brouhaha")
        created_package = package is None
        if created_package:
            package = ModuleType("brouhaha")
            package.__package__ = "brouhaha"
            package.__path__ = []
            sys.modules["brouhaha"] = package

        previous_attribute = getattr(package, "models", _MISSING)
        previous_module = sys.modules.get("brouhaha.models", _MISSING)
        package.models = compatibility_module
        sys.modules["brouhaha.models"] = compatibility_module
        try:
            yield
        finally:
            if previous_module is _MISSING:
                sys.modules.pop("brouhaha.models", None)
            else:
                sys.modules["brouhaha.models"] = previous_module
            if previous_attribute is _MISSING:
                try:
                    del package.models
                except AttributeError:
                    pass
            else:
                package.models = previous_attribute
            if created_package and sys.modules.get("brouhaha") is package:
                sys.modules.pop("brouhaha", None)


def _loader_options(loader, values: dict, *, required: tuple[str, ...]) -> dict:
    try:
        parameters = signature(loader).parameters
    except (TypeError, ValueError):
        return {name: value for name, value in values.items() if value is not None}
    accepts_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
    options = {
        name: value
        for name, value in values.items() if value is not None and (accepts_kwargs or name in parameters)
    }
    missing = sorted(name for name in required if name not in options)
    if missing:
        raise RuntimeError(
            "The installed pyannote.audio version does not support requested "
            f"model loading option(s): {', '.join(missing)}.")
    return options


def _cpu_array(value):
    for method_name in ("detach", "float", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return value


def _shift_and_clip(
    segments,
    *,
    offset: float,
    duration: float,
) -> tuple[SpeechSegment, ...]:
    shifted = []
    for segment in segments:
        start = max(0.0, segment.start + offset)
        end = min(duration, segment.end + offset)
        if end <= start:
            continue
        shifted.append(
            SpeechSegment(
                start=start,
                end=end,
                score=segment.score,
                label=segment.label,
                channel=segment.channel,
                metadata=dict(segment.metadata),
            ))
    return merge_speech_segments(shifted)


class PyannoteBrouhahaVADForVoiceActivityDetection(PyannoteVADForVoiceActivityDetection):
    """Expose Brouhaha VAD probabilities with SNR/C50 summary metadata."""

    config_class = PyannoteBrouhahaVADConfig
    default_model_name_or_path = "pyannote/brouhaha"
    training_support = "upstream-custom"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: PyannoteBrouhahaVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs,
    ):
        super().__init__(
            config,
            model_path=model_path,
            device=device,
            lazy_load=lazy_load,
            token=token,
            **kwargs,
        )

    def _load_pretrained_model(self) -> None:
        pyannote_audio = import_optional(
            "pyannote.audio",
            model_type=self.config.model_type,
            install_extra=None,
        )
        model_class = getattr(pyannote_audio, "Model", None)
        loader = getattr(model_class, "from_pretrained", None)
        inference_class = getattr(pyannote_audio, "Inference", None)
        if not callable(loader) or not callable(inference_class):
            raise RuntimeError(
                "The installed pyannote.audio package must expose "
                "Model.from_pretrained() and Inference().")

        configured = {
            "revision": self.config.revision,
            "subfolder": self.config.subfolder,
            "cache_dir": self.config.cache_dir,
            "token": self._auth_token,
        }
        required = tuple(name for name, value in configured.items() if value is not None)
        options = _loader_options(
            loader,
            configured,
            required=required,
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        with _brouhaha_architecture_import():
            model = loader(source, **options)
        if model is None:
            raise RuntimeError(f"pyannote.audio could not load Brouhaha model {source!r}.")

        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        inference_options = {
            "batch_size": self.config.batch_size,
            "device": torch.device(self.device),
        }
        if self.config.inference_duration_s is not None:
            inference_options["duration"] = self.config.inference_duration_s
        if self.config.inference_step_s is not None:
            inference_options["step"] = self.config.inference_step_s
        inference = inference_class(model, **inference_options)
        self.model = _BrouhahaRuntime(
            model=model,
            inference=inference,
        )

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
        if window_size_samples is not None:
            raise ValueError(
                "Brouhaha frame geometry is fixed by its checkpoint; "
                "`window_size_samples` is not supported.")
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        waveform = torch.as_tensor(materialized.waveform)
        if getattr(waveform, "ndim", len(getattr(waveform, "shape", ()))) == 1:
            waveform = waveform.unsqueeze(0)
        output = self.model.inference({
            "waveform": waveform,
            "sample_rate": materialized.sampling_rate,
        })

        np = import_optional(
            "numpy",
            model_type=self.config.model_type,
            install_extra=None,
        )
        data = np.asarray(_cpu_array(getattr(output, "data", output)))
        if data.ndim != 2 or data.shape[1] < 3:
            raise RuntimeError(
                "Brouhaha inference must return frame values shaped "
                "(num_frames, >=3) for VAD, SNR, and C50.")
        vad_scores = data[:, 0].astype("float32", copy=False)
        if (not np.isfinite(vad_scores).all() or ((vad_scores < 0.0) | (vad_scores > 1.0)).any()):
            raise ValueError("Brouhaha VAD frame scores must be finite probabilities "
                             "between 0 and 1.")

        sliding_window = getattr(output, "sliding_window", None)
        frame_step_s = getattr(sliding_window, "step", None)
        frame_duration_s = getattr(sliding_window, "duration", None)
        frame_start_s = getattr(sliding_window, "start", 0.0)
        if frame_step_s is None or frame_duration_s is None:
            raise RuntimeError("Brouhaha output must expose sliding-window step and duration.")
        frame_hop_samples = max(1, round(float(frame_step_s) * self.sample_rate))
        frame_length_samples = max(
            1,
            round(float(frame_duration_s) * self.sample_rate),
        )
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
            vad_scores.tolist(),
            sampling_rate=self.sample_rate,
            frame_hop_samples=frame_hop_samples,
            frame_length_samples=frame_length_samples,
            config=postprocessing,
        )
        segments = _shift_and_clip(
            segments,
            offset=float(frame_start_s),
            duration=materialized.duration,
        )
        snr_values = data[:, 1]
        c50_values = data[:, 2]
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            probabilities=vad_scores if return_frames else None,
            metadata={
                "backend": "pyannote-brouhaha",
                "source": self.config.name_or_path or self.default_model_name_or_path,
                "frame_hop_samples": frame_hop_samples,
                "frame_length_samples": frame_length_samples,
                "mean_snr_db": (float(np.mean(snr_values)) if len(snr_values) else None),
                "mean_c50_db": (float(np.mean(c50_values)) if len(c50_values) else None),
                "auxiliary_outputs": ("snr_db", "c50_db"),
                "frame_scores_available": True,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "Brouhaha fine-tuning is upstream-custom and requires its "
            "multi-task pyannote protocol, contamination pipeline, and "
            "joint VAD/SNR/C50 loss.")
