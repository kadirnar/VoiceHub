from abc import ABC, abstractmethod
from collections.abc import Sequence
from importlib import import_module
from math import isfinite
from numbers import Integral, Number, Real
from pathlib import Path
from typing import Any


class BaseTTSModel(ABC):
    """Abstract base class for all VoiceHub TTS inference models."""

    def __init__(self, model_path: str = "", device: str = "cuda"):
        self.model_path = model_path
        self.device = device

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @staticmethod
    def validate_audio(audio_data: Any) -> None:
        """Validate a materialized waveform without changing its public type.

        Python sequences and tensor-like values are checked without
        importing NumPy. Other array types fall back to a lazy NumPy
        validation path.
        """
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

        np = import_module("numpy")
        try:
            audio_array = np.asarray(audio_data)
        except (TypeError, ValueError) as exc:
            raise TypeError("`audio_data` must contain numeric samples.", ) from exc
        if audio_array.size == 0:
            raise ValueError("`audio_data` cannot be empty.")
        if (not np.issubdtype(audio_array.dtype, np.number) or
                np.issubdtype(audio_array.dtype, np.complexfloating)):
            raise TypeError("`audio_data` must contain real numeric samples.", )
        if not np.isfinite(audio_array).all():
            raise ValueError("`audio_data` contains NaN or infinite samples.")

    @staticmethod
    def save_audio(
        file_path: str | Path,
        audio_data: Any,
        sample_rate: int,
    ) -> str:
        """Write a mono waveform while keeping NumPy and SoundFile lazy.

        Importing VoiceHub should be cheap. Audio dependencies are
        therefore imported only when a backend actually writes a file.
        """
        if not isinstance(file_path, (str, Path)) or not str(file_path).strip():
            raise ValueError("`file_path` must be a non-empty path.")
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral) or sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")

        BaseTTSModel.validate_audio(audio_data)
        np = import_module("numpy")
        if hasattr(audio_data, "detach"):
            audio_data = audio_data.detach().cpu()
            if str(getattr(audio_data, "dtype", "")) == "torch.bfloat16":
                audio_data = audio_data.float()
            audio_data = audio_data.numpy()
        audio_data = np.asarray(audio_data)
        audio_data = np.squeeze(audio_data)
        if audio_data.ndim == 0:
            audio_data = audio_data.reshape(1)
        if audio_data.ndim != 1:
            raise ValueError(
                "`audio_data` must contain one mono waveform; "
                f"received shape {audio_data.shape}.")
        output_path = Path(file_path).expanduser()
        if output_path.exists() and output_path.is_dir():
            raise IsADirectoryError(f"Audio output path is a directory: {output_path}.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf = import_module("soundfile")
        sf.write(str(output_path), audio_data, int(sample_rate))
        return str(output_path)
