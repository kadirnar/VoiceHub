from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from importlib import import_module
from math import isfinite
from numbers import Integral, Number, Real
from pathlib import Path
from typing import Any


class BaseSpeechModel(ABC):
    """Framework-independent base for every VoiceHub speech model.

    The class owns task-neutral waveform validation, persistence, and
    explicit optimization-plan state. Task semantics such as synthesis,
    transcription, and speech detection belong to the corresponding
    pretrained model subclasses.
    """

    def __init__(self, model_path: str = "", device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._optimization_results_by_mode: dict[str, Any] = {}

    def _optimization_lifecycle_lock(self):
        lock = getattr(self, "_lifecycle_lock", None)
        return lock if lock is not None else nullcontext()

    @staticmethod
    def _optimization_mode(mode):
        from voicehub.optimization import OptimizationMode

        return OptimizationMode.coerce(mode)

    def _validate_optimization_transition(self, target_mode) -> None:
        """Reject implicit transitions across active optimization plans."""
        normalized = self._optimization_mode(target_mode)
        active_modes = tuple(
            mode for mode, result in self._optimization_results_by_mode.items()
            if result is not None and mode != normalized.value)
        if active_modes:
            raise RuntimeError(
                f"Cannot enter {normalized.value!r} mode while an explicit "
                f"{active_modes[0]!r} optimization plan is active. Restore "
                "that plan explicitly with restore_optimization_plan() first.")

    def _default_optimization_context(self, mode):
        """Describe the loaded runtime without importing optimization tools."""
        from voicehub.optimization import OptimizationContext

        runtime = getattr(self, "model", None)
        device = str(getattr(self, "device", "cpu"))
        dtype = "float32"
        parameters = getattr(runtime, "parameters", None)
        if callable(parameters):
            try:
                parameter = next(iter(parameters()))
            except (StopIteration, TypeError):
                parameter = None
            if parameter is not None:
                parameter_device = getattr(parameter, "device", None)
                parameter_dtype = getattr(parameter, "dtype", None)
                if parameter_device is not None:
                    device = str(parameter_device)
                if parameter_dtype is not None:
                    dtype = str(parameter_dtype).removeprefix("torch.")
        return OptimizationContext(
            mode=mode,
            device=device,
            dtype=dtype,
        )

    def apply_optimization_plan(
        self,
        passes,
        *,
        mode,
        context=None,
        registry=None,
    ):
        """Explicitly apply named or configured passes to the model runtime.

        No plan is selected automatically. The returned
        :class:`~voicehub.optimization.OptimizationResult` owns rollback
        state and a deterministic manifest for the exact application.
        """
        from voicehub.optimization import OptimizationContext, OptimizationPassManager, bind_registered_architecture

        normalized_mode = self._optimization_mode(mode)
        if context is not None:
            if not isinstance(context, OptimizationContext):
                raise TypeError("`context` must be an OptimizationContext.")
            if context.mode is not normalized_mode:
                raise ValueError(
                    "Optimization context mode does not match the requested "
                    f"mode ({context.mode.value!r} != "
                    f"{normalized_mode.value!r}).")

        manager = OptimizationPassManager()
        resolved_passes = manager.resolve(passes, registry=registry)
        if not resolved_passes:
            raise ValueError("An optimization plan must contain at least one pass.")

        with self._optimization_lifecycle_lock():
            self._validate_optimization_transition(normalized_mode)
            if normalized_mode.value in self._optimization_results_by_mode:
                raise RuntimeError(
                    f"An explicit {normalized_mode.value!r} optimization plan "
                    "is already active. Restore it before applying another.")

            loader_name = ("load_for_training" if normalized_mode.value == "training" else "load")
            loader = getattr(self, loader_name, None)
            if not callable(loader):
                raise TypeError(
                    f"{type(self).__name__} does not implement "
                    f"{loader_name}(), which is required for "
                    f"{normalized_mode.value} optimization.")
            loader()
            runtime = getattr(self, "model", None)
            if runtime is None:
                raise RuntimeError(
                    f"{type(self).__name__} did not expose a loaded model "
                    "runtime for optimization.")
            resolved_context = (
                context if context is not None else self._default_optimization_context(normalized_mode))
            resolved_context = bind_registered_architecture(
                resolved_context,
                self,
            )
            result = manager.apply(
                runtime,
                resolved_passes,
                resolved_context,
            )
            # Validate manifest safety before attaching the transformed graph.
            result.manifest()
            self.model = result.model
            self._optimization_results_by_mode[normalized_mode.value] = result
            return result

    def optimization_result(self, *, mode):
        """Return the active result for one mode, if a plan was applied."""
        normalized = self._optimization_mode(mode)
        return self._optimization_results_by_mode.get(normalized.value)

    def optimization_manifest(self, *, mode=None):
        """Return checkpoint-safe metadata for active explicit plans."""
        if mode is not None:
            result = self.optimization_result(mode=mode)
            return None if result is None else result.manifest()
        return {
            active_mode: result.manifest()
            for active_mode, result in sorted(self._optimization_results_by_mode.items())
        }

    def restore_optimization_plan(self, *, mode):
        """Restore one reversible plan and return the original runtime."""
        normalized = self._optimization_mode(mode)
        with self._optimization_lifecycle_lock():
            result = self._optimization_results_by_mode.get(normalized.value)
            if result is None:
                raise RuntimeError(f"No {normalized.value!r} optimization plan is active.")
            restored = result.restore()
            self.model = restored
            del self._optimization_results_by_mode[normalized.value]
            return restored

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @staticmethod
    def validate_audio(audio_data: Any) -> None:
        """Validate a materialized waveform without changing its public
        type."""
        if audio_data is None:
            raise ValueError("`audio_data` cannot be None.")

        numel = getattr(audio_data, "numel", None)
        tensor_is_finite = getattr(audio_data, "isfinite", None)
        if callable(numel) and callable(tensor_is_finite):
            if int(numel()) == 0:
                raise ValueError("`audio_data` cannot be empty.")
            if str(getattr(audio_data, "dtype", "")) in {
                    "bool",
                    "torch.bool",
            }:
                raise TypeError("`audio_data` must contain real numeric samples.", )
            is_complex = getattr(audio_data, "is_complex", None)
            if callable(is_complex) and is_complex():
                raise TypeError("`audio_data` must contain real numeric samples.", )
            finite = tensor_is_finite().all()
            if hasattr(finite, "item"):
                finite = finite.item()
            if not bool(finite):
                raise ValueError("`audio_data` contains NaN or infinite samples.", )
            return

        def validate_sequence(value: Any) -> int | None:
            if isinstance(value, Number):
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise TypeError("`audio_data` must contain real numeric samples.", )
                if not isfinite(value):
                    raise ValueError("`audio_data` contains NaN or infinite samples.", )
                return 1
            if isinstance(value, (str, bytes, bytearray, dict)):
                raise TypeError("`audio_data` must contain real numeric samples.", )
            if not isinstance(value, Sequence):
                return None
            sample_count = 0
            for sample in value:
                nested_count = validate_sequence(sample)
                if nested_count is None:
                    return None
                sample_count += nested_count
            return sample_count

        sample_count = validate_sequence(audio_data)
        if sample_count is not None:
            if sample_count == 0:
                raise ValueError("`audio_data` cannot be empty.")
            return

        # NumPy-style arrays are intentionally supported through their public
        # array protocol without importing NumPy. This keeps waveform
        # validation independent from both NumPy and whichever tensor module a
        # provider happens to expose in a test or adapter process.
        array_size = getattr(audio_data, "size", None)
        flat_samples = getattr(audio_data, "flat", None)
        if (isinstance(array_size, Integral) and flat_samples is not None and not callable(flat_samples)):
            if array_size == 0:
                raise ValueError("`audio_data` cannot be empty.")
            dtype_kind = getattr(
                getattr(audio_data, "dtype", None),
                "kind",
                None,
            )
            if dtype_kind is not None and dtype_kind not in {"i", "u", "f"}:
                raise TypeError("`audio_data` must contain real numeric samples.")
            validated_count = 0
            for sample in flat_samples:
                item = getattr(sample, "item", None)
                if callable(item):
                    sample = item()
                nested_count = validate_sequence(sample)
                if nested_count is None:
                    raise TypeError("`audio_data` must contain numeric samples.")
                validated_count += nested_count
            if validated_count != int(array_size):
                raise ValueError("`audio_data` reported an inconsistent array size.")
            return

        torch = import_module("torch")
        try:
            audio_tensor = torch.as_tensor(audio_data)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("`audio_data` must contain numeric samples.", ) from exc
        if audio_tensor.numel() == 0:
            raise ValueError("`audio_data` cannot be empty.")
        if audio_tensor.dtype == torch.bool or audio_tensor.is_complex():
            raise TypeError("`audio_data` must contain real numeric samples.", )
        if not torch.isfinite(audio_tensor).all():
            raise ValueError("`audio_data` contains NaN or infinite samples.")

    @staticmethod
    def save_audio(
        file_path: str | Path,
        audio_data: Any,
        sample_rate: int,
    ) -> str:
        """Write one mono waveform as portable 16-bit PCM WAVE audio."""
        if not isinstance(file_path, (str, Path)) or not str(file_path).strip():
            raise ValueError("`file_path` must be a non-empty path.")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")

        BaseSpeechModel.validate_audio(audio_data)
        torch = import_module("torch")
        try:
            waveform = torch.as_tensor(audio_data).detach()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TypeError("`audio_data` must contain numeric samples.") from exc
        waveform = waveform.squeeze()
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim != 1:
            raise ValueError(
                "`audio_data` must contain one mono waveform; "
                f"received shape {tuple(waveform.shape)}.")

        output_path = Path(file_path).expanduser()
        if output_path.exists() and output_path.is_dir():
            raise IsADirectoryError(f"Audio output path is a directory: {output_path}.")
        if output_path.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError(
                "VoiceHub's dependency-free audio writer supports PCM WAVE "
                "output. Use a `.wav` or `.wave` file name.")

        from voicehub.processing.waveform import normalize_waveform, save_pcm_wave

        waveform = normalize_waveform(waveform)
        save_pcm_wave(output_path, waveform, int(sample_rate))
        return str(output_path)


class BaseTTSModel(BaseSpeechModel):
    """Backward-compatible base class for text-to-speech models."""

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError
