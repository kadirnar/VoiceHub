"""Lazy Silero VAD wrapper."""

from __future__ import annotations

from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import VADOutput
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_silero.configuration_vad_silero import SileroVADConfig
from voicehub.vad_utils import normalize_backend_segments


def _timestamp_options(callable_object, values, *, required=()):
    try:
        parameters = signature(callable_object).parameters
    except (TypeError, ValueError):
        return {name: value for name, value in values.items() if value is not None}
    accepts_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
    options = {
        name: value
        for name, value in values.items() if value is not None and (accepts_kwargs or name in parameters)
    }
    missing = sorted(name for name in required if name not in options)
    if missing:
        formatted = ", ".join(f"`{name}`" for name in missing)
        raise ValueError(
            "The installed silero-vad version does not support the requested "
            f"inference option(s): {formatted}.")
    return options


class SileroVADForVoiceActivityDetection(PreTrainedVADModel):
    """Speech detection for official Silero JIT and ONNX checkpoints."""

    config_class = SileroVADConfig
    default_model_name_or_path = "silero_vad"

    def __init__(
        self,
        config: SileroVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)
        self._get_speech_timestamps = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="silero-vad")

    def _load_pretrained_model(self) -> None:
        silero = import_optional(
            "silero_vad",
            model_type=self.config.model_type,
            install_extra="asr-vad",
        )
        loader = getattr(silero, "load_silero_vad", None)
        timestamps = getattr(silero, "get_speech_timestamps", None)
        if not callable(loader) or not callable(timestamps):
            raise RuntimeError(
                "The installed silero-vad package does not expose "
                "load_silero_vad/get_speech_timestamps.")
        try:
            available = signature(loader).parameters
        except (TypeError, ValueError):
            available = {}
        load_options = {"onnx": self.config.use_onnx}
        if "force_reload" in available:
            load_options["force_reload"] = self.config.force_reload
        if "force_onnx_cpu" in available:
            load_options["force_onnx_cpu"] = self.device == "cpu"
        self.model = loader(**load_options)
        if self.model is None:
            raise RuntimeError("silero-vad returned no model runtime.")
        if not self.config.use_onnx and self.device != "cpu":
            move = getattr(self.model, "to", None)
            if not callable(move):
                raise RuntimeError(
                    "This silero-vad JIT runtime cannot move to the requested "
                    f"device {self.device!r}.")
            moved = move(self.device)
            if moved is not None:
                self.model = moved
        self._get_speech_timestamps = timestamps

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
                "Silero's public timestamp API does not return calibrated frame "
                "scores; use `return_frames=False`.")
        if onset is not None:
            threshold = onset
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
        if not self.config.use_onnx and hasattr(waveform, "to"):
            waveform = waveform.to(self.device)
        options = {
            "sampling_rate": materialized.sampling_rate,
            "threshold": threshold,
            "min_speech_duration_ms": min_speech_duration_ms,
            "min_silence_duration_ms": min_silence_duration_ms,
            "speech_pad_ms": speech_pad_ms,
            "return_seconds": False,
        }
        if offset is not None:
            options["neg_threshold"] = offset
        if max_speech_duration_s is not None:
            options["max_speech_duration_s"] = max_speech_duration_s
        if window_size_samples is not None:
            options["window_size_samples"] = window_size_samples
        required = []
        if materialized.sampling_rate != 16_000:
            required.append("sampling_rate")
        if threshold != 0.5 or onset is not None:
            required.append("threshold")
        if min_speech_duration_ms != 250:
            required.append("min_speech_duration_ms")
        if min_silence_duration_ms != 100:
            required.append("min_silence_duration_ms")
        if speech_pad_ms != 30:
            required.append("speech_pad_ms")
        if offset is not None:
            required.append("neg_threshold")
        if max_speech_duration_s is not None:
            required.append("max_speech_duration_s")
        if window_size_samples is not None:
            required.append("window_size_samples")
        options = _timestamp_options(
            self._get_speech_timestamps,
            options,
            required=tuple(required),
        )
        timestamps = self._get_speech_timestamps(
            waveform,
            self.model,
            **options,
        )
        segments = normalize_backend_segments(
            timestamps,
            sampling_rate=materialized.sampling_rate,
            timestamps_are_samples=True,
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            metadata={
                "backend": "silero",
                "runtime": "onnx" if self.config.use_onnx else "jit",
                "frame_scores_available": False,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "Silero publishes inference artifacts but no supported training "
            "recipe. Fine-tuning is unavailable for this backend.")
