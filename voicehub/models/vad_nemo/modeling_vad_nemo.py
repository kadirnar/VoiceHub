"""Lazy NVIDIA NeMo MarbleNet VAD wrapper."""

from __future__ import annotations

import math
from contextlib import nullcontext
from numbers import Real
from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.inference_configuration import VADInferenceConfig
from voicehub.modeling_outputs import VADOutput
from voicehub.models.native_utils import resolve_cpu_cuda_device
from voicehub.models.vad_nemo.configuration_vad_nemo import NeMoVADConfig
from voicehub.vad_utils import frame_probabilities_to_segments


def _nested_values(value):
    if isinstance(value, dict):
        value = value.get("logits", value.get("outputs"))
    elif hasattr(value, "logits"):
        value = value.logits
    if (isinstance(value, tuple) and value and not isinstance(value[0], (int, float))):
        # Native NeMo methods may return ``(logits, length)``.
        first = value[0]
        if hasattr(first, "shape") or isinstance(first, (tuple, list)):
            value = first
    for method_name in ("detach", "float", "cpu"):
        method = getattr(value, method_name, None)
        if callable(method):
            value = method()
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _speech_probability(logits, speech_class_id: int) -> float:
    if hasattr(logits, "item"):
        logits = logits.item()
    if isinstance(logits, Real) and not isinstance(logits, bool):
        logits = (logits, )
    try:
        values = [float(value.item() if hasattr(value, "item") else value) for value in logits]
    except TypeError as exc:
        raise TypeError("NeMo VAD logits must be a scalar or a class-logit sequence.") from exc
    if not values:
        raise ValueError("NeMo VAD returned an empty class-logit vector.")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("NeMo VAD returned non-finite class logits.")
    if len(values) == 1:
        value = values[0]
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)
    if speech_class_id >= len(values):
        raise ValueError(
            f"`speech_class_id` {speech_class_id} is outside "
            f"{len(values)} NeMo output classes.")
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    return exponentials[speech_class_id] / sum(exponentials)


class NeMoVADForVoiceActivityDetection(PreTrainedVADModel):
    """Run native NeMo window or frame VAD checkpoints."""

    config_class = NeMoVADConfig
    default_model_name_or_path = "vad_multilingual_marblenet"
    training_support = "upstream-custom"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: NeMoVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)
        self.architecture_family = config.architecture_family

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="NeMo VAD")

    def _resolve_native_source(self) -> tuple[str, Path | None]:
        source = self.config.name_or_path or self.default_model_name_or_path
        local_source = Path(source).expanduser()
        if local_source.is_dir():
            checkpoints = sorted(
                path for path in local_source.iterdir()
                if path.is_file() and path.suffix.lower() in (".ckpt", ".nemo"))
            if len(checkpoints) != 1:
                raise ValueError(
                    "A local NeMo VAD directory must contain exactly one "
                    "top-level .nemo or .ckpt checkpoint.")
            local_source = checkpoints[0]
            source = str(local_source)
        return source, local_source if local_source.is_file() else None

    def _load_pretrained_model(self) -> None:
        nemo_models = import_optional(
            "nemo.collections.asr.models",
            model_type=self.config.model_type,
            install_extra=None,
        )
        source, local_source = self._resolve_native_source()
        family = self.config.architecture_family
        if family == "auto":
            family = ("frame" if "frame" in Path(source).name.lower() else "window")
        class_name = ("EncDecFrameClassificationModel" if family == "frame" else "EncDecClassificationModel")
        model_class = getattr(nemo_models, class_name, None)
        if model_class is None:
            raise RuntimeError(f"The installed NeMo package does not expose {class_name}.")
        options = dict(self.config.model_kwargs)
        suffix = local_source.suffix.lower() if local_source is not None else ""
        if suffix == ".nemo":
            loader = getattr(model_class, "restore_from", None)
            if not callable(loader):
                raise RuntimeError(f"{class_name} cannot restore .nemo checkpoints.")
            model = loader(
                restore_path=str(local_source),
                map_location=self.device,
                **options,
            )
        elif suffix == ".ckpt":
            loader = getattr(model_class, "load_from_checkpoint", None)
            if not callable(loader):
                raise RuntimeError(f"{class_name} cannot load .ckpt checkpoints.")
            model = loader(
                checkpoint_path=str(local_source),
                map_location=self.device,
                **options,
            )
        elif local_source is not None:
            raise ValueError("Local NeMo VAD artifacts must be .nemo or .ckpt files.")
        else:
            loader = getattr(model_class, "from_pretrained", None)
            if not callable(loader):
                raise RuntimeError(f"{class_name} does not expose from_pretrained().")
            model = loader(
                model_name=source,
                map_location=self.device,
                **options,
            )
        if model is None:
            raise RuntimeError(f"NeMo could not load the VAD runtime from {source!r}.")
        move = getattr(model, "to", None)
        if callable(move):
            move(self.device)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        self.architecture_family = family
        self.model = model

    def _call_model(self, waveforms, lengths):
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        input_signal = torch.as_tensor(waveforms)
        input_length = torch.as_tensor(lengths)
        if hasattr(input_signal, "to"):
            input_signal = input_signal.to(self.device)
        if hasattr(input_length, "to"):
            input_length = input_length.to(self.device)
        inference_mode = getattr(torch, "inference_mode", None)
        context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            output = self.model(
                input_signal=input_signal,
                input_signal_length=input_length,
            )
        return _nested_values(output)

    def _window_probabilities(
        self,
        waveform,
        *,
        frame_length: int,
        frame_hop: int,
    ) -> list[float]:
        np = import_optional(
            "numpy",
            model_type=self.config.model_type,
            install_extra=None,
        )
        windows = []
        for start in range(0, len(waveform), frame_hop):
            window = waveform[start:start + frame_length]
            if len(window) < frame_length:
                window = np.pad(window, (0, frame_length - len(window)))
            windows.append(window)
            if start + frame_length >= len(waveform):
                break
        probabilities = []
        for start in range(0, len(windows), self.config.batch_size):
            batch = np.stack(
                windows[start:start + self.config.batch_size],
                axis=0,
            )
            values = self._call_model(
                batch,
                [frame_length] * len(batch),
            )
            if not isinstance(values, (tuple, list)):
                values = [values]
            if len(batch) == 1 and values and isinstance(values[0], (int, float)):
                values = [values]
            for logits in values:
                while (isinstance(logits, (tuple, list)) and len(logits) == 1 and isinstance(logits[0],
                                                                                             (tuple, list))):
                    logits = logits[0]
                probabilities.append(_speech_probability(
                    logits,
                    self.config.speech_class_id,
                ))
        return probabilities

    def _frame_probabilities(self, waveform) -> list[float]:
        values = self._call_model(
            [waveform],
            [len(waveform)],
        )
        if not isinstance(values, (tuple, list)):
            values = [values]
        if len(values) == 1 and isinstance(values[0], (tuple, list)):
            values = values[0]
        return [_speech_probability(logits, self.config.speech_class_id) for logits in values]

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
        if self.architecture_family == "frame":
            if window_size_samples is not None:
                raise ValueError(
                    "NeMo frame VAD derives its frame geometry from the "
                    "checkpoint; `window_size_samples` is not supported.")
            probabilities = self._frame_probabilities(materialized.waveform)
            frame_count = len(probabilities)
            if frame_count == 0:
                raise ValueError("NeMo frame VAD returned no predictions.")
            frame_hop = max(1, round(len(materialized.waveform) / frame_count))
            frame_length = frame_hop
        else:
            frame_length = max(
                1,
                window_size_samples if window_size_samples is not None else round(
                    self.config.window_duration_s * self.sample_rate),
            )
            frame_hop = max(
                1,
                round(self.config.hop_duration_s * self.sample_rate),
            )
            probabilities = self._window_probabilities(
                materialized.waveform,
                frame_length=frame_length,
                frame_hop=frame_hop,
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
            probabilities,
            sampling_rate=materialized.sampling_rate,
            frame_hop_samples=frame_hop,
            frame_length_samples=frame_length,
            config=postprocessing,
            duration_samples=len(materialized.waveform),
        )
        return VADOutput(
            segments=segments,
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            probabilities=tuple(probabilities) if return_frames else None,
            metadata={
                "backend": "nemo",
                "architecture_family": self.architecture_family,
                "speech_class_id": self.config.speech_class_id,
                "frame_hop_samples": frame_hop,
                "frame_length_samples": frame_length,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        save_to = getattr(self.model, "save_to", None)
        if callable(save_to):
            save_directory.mkdir(parents=True, exist_ok=True)
            save_to(str(save_directory / "model.nemo"))

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "NeMo VAD training is upstream-custom and requires a NeMo "
            "Trainer/Hydra recipe. VoiceHub's generic fine-tuning adapter is "
            "intentionally unavailable.")
