"""Canonical, dependency-light audio loading for ASR and VAD backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.base_model import BaseSpeechModel


@dataclass(frozen=True)
class AudioInput:
    """Materialized mono waveform with an explicit sampling rate."""

    waveform: Any
    sampling_rate: int
    path: Path | None = None

    def __post_init__(self) -> None:
        if (isinstance(self.sampling_rate, bool) or not isinstance(self.sampling_rate, Integral) or
                self.sampling_rate <= 0):
            raise ValueError("AudioInput `sampling_rate` must be a positive integer.")
        BaseSpeechModel.validate_audio(self.waveform)

    @property
    def duration(self) -> float:
        """Return duration in seconds."""
        shape = getattr(self.waveform, "shape", None)
        sample_count = int(shape[-1]) if shape else len(self.waveform)
        return sample_count / int(self.sampling_rate)


def _mapping_audio(value: Mapping[str, Any]) -> tuple[Any, int | None]:
    waveform = None
    for name in ("array", "waveform", "audio", "input_values"):
        if name in value:
            waveform = value[name]
            break
    if waveform is None:
        raise ValueError("Audio mappings must contain one of: array, waveform, audio, input_values.")
    sampling_rate = value.get("sampling_rate", value.get("sample_rate"))
    return waveform, sampling_rate


def _mono_array(waveform: Any):
    np = import_module("numpy")
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
        if str(getattr(waveform, "dtype", "")) == "torch.bfloat16":
            waveform = waveform.float()
        waveform = waveform.numpy()
    try:
        source = np.asarray(waveform)
    except (TypeError, ValueError) as exc:
        raise TypeError("Audio input must contain real numeric samples.") from exc
    if np.issubdtype(source.dtype, np.bool_):
        raise TypeError("Audio input must contain real numeric samples.")
    if np.issubdtype(source.dtype, np.signedinteger):
        limits = np.iinfo(source.dtype)
        scale = float(max(abs(limits.min), limits.max))
        array = source.astype(np.float32) / scale
    elif np.issubdtype(source.dtype, np.unsignedinteger):
        limits = np.iinfo(source.dtype)
        midpoint = float(limits.max + 1) / 2.0
        array = (source.astype(np.float32) - midpoint) / midpoint
    elif np.issubdtype(source.dtype, np.floating):
        array = source.astype(np.float32, copy=False)
    else:
        raise TypeError("Audio input must contain real numeric samples.")
    if array.ndim == 0:
        array = array.reshape(1)
    while array.ndim > 1 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim == 2:
        first_is_channels = array.shape[0] <= 8
        last_is_channels = array.shape[1] <= 8
        if first_is_channels and (not last_is_channels or array.shape[0] <= array.shape[1]):
            array = array.mean(axis=0)
        elif last_is_channels:
            array = array.mean(axis=1)
        else:
            raise ValueError("Two-dimensional audio must have a channel dimension of at most 8.")
    if array.ndim != 1:
        raise ValueError("Audio input must resolve to one mono waveform; "
                         f"received shape {array.shape}.")
    BaseSpeechModel.validate_audio(array)
    return array


def _resample(waveform, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return waveform
    np = import_module("numpy")
    target_length = max(1, int(round(len(waveform) * target_rate / source_rate)))
    if len(waveform) == 1:
        return np.full(target_length, waveform[0], dtype=np.float32)
    source_positions = np.arange(len(waveform), dtype=np.float64)
    target_positions = np.linspace(
        0,
        len(waveform) - 1,
        target_length,
        dtype=np.float64,
    )
    return np.interp(target_positions, source_positions, waveform).astype(
        np.float32,
        copy=False,
    )


def load_audio(
    audio: AudioInput | Mapping[str, Any] | str | Path | Any,
    *,
    sampling_rate: int | None = None,
    target_sampling_rate: int | None = None,
) -> AudioInput:
    """Load, downmix, and optionally resample one waveform.

    Array and tensor inputs must carry an explicit sampling rate. File
    inputs obtain it from SoundFile. Timestamps exposed by ASR/VAD
    wrappers remain in seconds, so resampling never changes their public
    time base.
    """
    original_path = None
    if isinstance(audio, AudioInput):
        if sampling_rate is not None and sampling_rate != audio.sampling_rate:
            raise ValueError("`sampling_rate` conflicts with the rate stored in AudioInput.")
        waveform = audio.waveform
        source_rate = audio.sampling_rate
        original_path = audio.path
    elif isinstance(audio, Mapping):
        waveform, mapped_rate = _mapping_audio(audio)
        if sampling_rate is not None and mapped_rate is not None and sampling_rate != mapped_rate:
            raise ValueError("`sampling_rate` conflicts with the rate stored in the audio mapping.")
        source_rate = sampling_rate if sampling_rate is not None else mapped_rate
    elif isinstance(audio, (str, Path)):
        original_path = Path(audio).expanduser()
        if not original_path.is_file():
            raise FileNotFoundError(f"Audio file was not found: {original_path}.")
        soundfile = import_module("soundfile")
        waveform, file_rate = soundfile.read(
            str(original_path),
            dtype="float32",
            always_2d=False,
        )
        if sampling_rate is not None and sampling_rate != file_rate:
            raise ValueError("`sampling_rate` does not match the audio file's sampling rate.")
        source_rate = int(file_rate)
    else:
        waveform = audio
        source_rate = sampling_rate

    if (isinstance(source_rate, bool) or not isinstance(source_rate, Integral) or source_rate <= 0):
        raise ValueError("Array and tensor audio inputs require a positive `sampling_rate`.")
    if target_sampling_rate is None:
        target_sampling_rate = int(source_rate)
    if (isinstance(target_sampling_rate, bool) or not isinstance(target_sampling_rate, Integral) or
            target_sampling_rate <= 0):
        raise ValueError("`target_sampling_rate` must be a positive integer.")

    waveform = _mono_array(waveform)
    waveform = _resample(
        waveform,
        int(source_rate),
        int(target_sampling_rate),
    )
    return AudioInput(
        waveform=waveform,
        sampling_rate=int(target_sampling_rate),
        path=original_path,
    )
