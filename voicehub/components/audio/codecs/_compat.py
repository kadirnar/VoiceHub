"""Small runtime primitives shared by the vendored DAC implementations.

The upstream Descript codecs inherit two convenience classes from
``descript-audiotools``.  Pulling that package into VoiceHub's runtime is
unnecessarily expensive and, more importantly, constrains ``protobuf`` to a
version that is incompatible with current ASR providers.  The codec models only
need a narrow part of that API during inference:

* a :class:`torch.nn.Module` with portable checkpoint loading and a ``device``
  property; and
* a tensor-backed audio container for resampling and loudness normalization.

This module intentionally implements only that stable surface.  It is not a
drop-in replacement for the data augmentation, plotting, or training utilities
provided by ``descript-audiotools``.
"""

from __future__ import annotations

import inspect
import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.processing.waveform import (
    load_pcm_wave,
    resample_waveform,
    save_pcm_wave,
)


def _torch_load(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    """Load a legacy codec checkpoint across PyTorch's ``weights_only`` change."""
    load_kwargs = {"map_location": map_location}
    try:
        return torch.load(path, weights_only=True, **load_kwargs)
    except TypeError:
        # ``weights_only`` was added after the oldest PyTorch supported by some
        # vendored checkpoints.
        return torch.load(path, **load_kwargs)
    except Exception as safe_load_error:
        try:
            # Descript's historic checkpoints can contain harmless constructor
            # metadata that is not accepted by the restricted unpickler.  This
            # matches their original loader and is only reached for legacy
            # artifacts.
            return torch.load(path, weights_only=False, **load_kwargs)
        except Exception:
            raise safe_load_error


class BaseModel(nn.Module):
    """Minimal Descript-compatible model base used by bundled codec graphs."""

    INTERN: list[str] = []
    EXTERN: list[str] = [
        "numpy.**",
        "scipy.**",
        "einops",
        "torch.**",
        "torchaudio.**",
        "tqdm",
    ]

    @property
    def device(self) -> torch.device:
        """Return the first parameter or buffer device."""
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.buffers(), None)
        if buffer is not None:
            return buffer.device
        return torch.device("cpu")

    @classmethod
    def load(
        cls,
        location: str | Path,
        *args: Any,
        package_name: str | None = None,
        strict: bool = False,
        **overrides: Any,
    ) -> BaseModel:
        """Load a torch.package archive or a Descript weights checkpoint."""
        try:
            return cls._load_package(location, package_name=package_name)
        except Exception:
            payload = _torch_load(location)

        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping) or "state_dict" not in payload:
            raise ValueError(
                f"Codec checkpoint {location!s} must contain a 'state_dict' mapping.")

        raw_metadata = payload.get("metadata")
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        raw_kwargs = metadata.get("kwargs")
        constructor_kwargs = dict(raw_kwargs) if isinstance(raw_kwargs, Mapping) else {}
        constructor_kwargs.update(overrides)

        signature = inspect.signature(cls)
        accepts_arbitrary_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values())
        if not accepts_arbitrary_kwargs:
            constructor_kwargs = {
                key: value
                for key, value in constructor_kwargs.items()
                if key in signature.parameters
            }

        model = cls(*args, **constructor_kwargs)
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Codec checkpoint {location!s} contains a non-mapping state_dict.")
        model.load_state_dict(state_dict, strict=strict)
        model.metadata = metadata
        return model

    def save(
        self,
        path: str | Path,
        metadata: Mapping[str, Any] | None = None,
        *,
        package: bool = True,
        intern: list[str] | None = None,
        extern: list[str] | None = None,
        mock: list[str] | None = None,
    ) -> str | Path:
        """Save in the package or weights layout used by Descript codecs."""
        constructor_kwargs: dict[str, Any] = {}
        for name, parameter in inspect.signature(self.__class__).parameters.items():
            if parameter.default is not inspect.Parameter.empty:
                constructor_kwargs[name] = getattr(self, name, parameter.default)

        checkpoint_metadata = dict(metadata or {})
        checkpoint_metadata["kwargs"] = constructor_kwargs
        existing_metadata = getattr(self, "metadata", None)
        if isinstance(existing_metadata, Mapping):
            merged_metadata = dict(existing_metadata)
            merged_metadata.update(checkpoint_metadata)
            checkpoint_metadata = merged_metadata
        self.metadata = checkpoint_metadata

        if package:
            self._save_package(
                path,
                intern=list(intern or ()),
                extern=list(extern or ()),
                mock=list(mock or ()),
            )
        else:
            torch.save(
                {
                    "state_dict": self.state_dict(),
                    "metadata": checkpoint_metadata,
                },
                path,
            )
        return path

    def _save_package(
        self,
        path: str | Path,
        *,
        intern: list[str],
        extern: list[str],
        mock: list[str],
    ) -> None:
        package_name = type(self).__name__
        resource_name = f"{package_name}.pth"
        importer = getattr(self, "importer", None)
        exporter_kwargs: dict[str, Any] = {}
        if importer is not None:
            exporter_kwargs["importer"] = (importer, torch.package.sys_importer)
            del self.importer

        try:
            with tempfile.NamedTemporaryFile(suffix=".pth") as temporary:
                with torch.package.PackageExporter(
                        temporary.name, **exporter_kwargs) as exporter:
                    exporter.intern(self.INTERN + intern)
                    exporter.extern(self.EXTERN + extern)
                    exporter.mock(mock)
                    exporter.save_pickle(package_name, resource_name, self)
                    exporter.save_pickle(
                        package_name,
                        f"{package_name}.metadata",
                        self.metadata,
                    )
                shutil.copyfile(temporary.name, path)
        finally:
            if importer is not None:
                self.importer = importer

    @classmethod
    def _load_package(
        cls,
        path: str | Path,
        *,
        package_name: str | None = None,
    ) -> BaseModel:
        resolved_package_name = package_name or cls.__name__
        importer = torch.package.PackageImporter(str(path))
        model = importer.load_pickle(
            resolved_package_name,
            f"{resolved_package_name}.pth",
            "cpu",
        )
        try:
            model.metadata = importer.load_pickle(
                resolved_package_name,
                f"{resolved_package_name}.metadata",
            )
        except Exception:
            pass
        model.importer = importer
        return model


def _coerce_audio_tensor(audio: Any) -> tuple[torch.Tensor, int]:
    tensor = audio if torch.is_tensor(audio) else torch.as_tensor(audio)
    original_ndim = tensor.ndim
    if original_ndim == 1:
        tensor = tensor[None, None, :]
    elif original_ndim == 2:
        tensor = tensor[None, :, :]
    elif original_ndim != 3:
        raise ValueError(
            "Audio must have shape (time,), (channels, time), or "
            f"(batch, channels, time); received {tuple(tensor.shape)}.")
    return tensor, original_ndim


def _restore_audio_shape(audio: torch.Tensor, original_ndim: int) -> torch.Tensor:
    if original_ndim == 1:
        return audio[0, 0]
    if original_ndim == 2:
        return audio[0]
    return audio


def _biquad_coefficients(
    *,
    sample_rate: int,
    frequency: float,
    filter_type: str,
    gain_db: float = 0.0,
    q: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized RBJ biquad coefficients in float64."""
    omega = 2.0 * math.pi * frequency / sample_rate
    cosine = math.cos(omega)
    sine = math.sin(omega)
    if filter_type == "high-pass":
        alpha = sine / (2.0 * q)
        numerator = (
            (1.0 + cosine) / 2.0,
            -(1.0 + cosine),
            (1.0 + cosine) / 2.0,
        )
        denominator = (
            1.0 + alpha,
            -2.0 * cosine,
            1.0 - alpha,
        )
    elif filter_type == "high-shelf":
        amplitude = 10.0 ** (gain_db / 40.0)
        alpha = sine / math.sqrt(2.0)
        shelf_term = 2.0 * math.sqrt(amplitude) * alpha
        numerator = (
            amplitude
            * (
                amplitude
                + 1.0
                + (amplitude - 1.0) * cosine
                + shelf_term
            ),
            -2.0
            * amplitude
            * (amplitude - 1.0 + (amplitude + 1.0) * cosine),
            amplitude
            * (
                amplitude
                + 1.0
                + (amplitude - 1.0) * cosine
                - shelf_term
            ),
        )
        denominator = (
            amplitude
            + 1.0
            - (amplitude - 1.0) * cosine
            + shelf_term,
            2.0
            * (amplitude - 1.0 - (amplitude + 1.0) * cosine),
            amplitude
            + 1.0
            - (amplitude - 1.0) * cosine
            - shelf_term,
        )
    else:  # pragma: no cover - private call sites are exhaustive.
        raise ValueError(f"Unknown biquad filter type {filter_type!r}.")
    scale = denominator[0]
    return (
        torch.tensor(
            tuple(value / scale for value in numerator),
            dtype=torch.float64,
        ),
        torch.tensor(
            (1.0, denominator[1] / scale, denominator[2] / scale),
            dtype=torch.float64,
        ),
    )


@torch.jit.script
def _apply_biquad(
    audio: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Apply a second-order IIR filter to ``[batch, channels, time]``."""
    output = torch.zeros_like(audio)
    sample_count = audio.shape[-1]
    for index in range(sample_count):
        value = numerator[0] * audio[..., index]
        if index >= 1:
            value = (
                value
                + numerator[1] * audio[..., index - 1]
                - denominator[1] * output[..., index - 1]
            )
        if index >= 2:
            value = (
                value
                + numerator[2] * audio[..., index - 2]
                - denominator[2] * output[..., index - 2]
            )
        output[..., index] = value
    return output


def _k_weight(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Apply the two-stage BS.1770 K-weighting filter."""
    shelf_b, shelf_a = _biquad_coefficients(
        sample_rate=sample_rate,
        frequency=1_500.0,
        filter_type="high-shelf",
        gain_db=4.0,
    )
    high_pass_b, high_pass_a = _biquad_coefficients(
        sample_rate=sample_rate,
        frequency=38.0,
        filter_type="high-pass",
        q=0.5,
    )
    coefficients = tuple(
        coefficient.to(device=audio.device, dtype=audio.dtype)
        for coefficient in (shelf_b, shelf_a, high_pass_b, high_pass_a)
    )
    weighted = _apply_biquad(audio, coefficients[0], coefficients[1])
    return _apply_biquad(weighted, coefficients[2], coefficients[3])


def integrated_loudness(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Measure gated ITU-R BS.1770-style integrated loudness natively.

    VoiceHub uses the standard K-weighting stages, 400 ms blocks with 75%
    overlap, the -70 LUFS absolute gate, and the -10 LU relative gate. This
    keeps codec normalization independent of TorchAudio and remains
    differentiable with respect to the input waveform.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    tensor, _ = _coerce_audio_tensor(audio)
    tensor = tensor.to(dtype=torch.float32)
    block_samples = max(1, int(round(0.4 * sample_rate)))
    if tensor.shape[-1] < block_samples:
        tensor = torch.nn.functional.pad(
            tensor,
            (0, block_samples - tensor.shape[-1]),
        )
    filtered = _k_weight(tensor, sample_rate)
    step_samples = max(1, block_samples // 4)
    blocks = filtered.unfold(
        dimension=-1,
        size=block_samples,
        step=step_samples,
    )
    channel_energy = blocks.square().mean(dim=-1)
    channel_count = channel_energy.shape[1]
    channel_gains = torch.ones(
        channel_count,
        dtype=channel_energy.dtype,
        device=channel_energy.device,
    )
    if channel_count > 3:
        channel_gains[3:min(channel_count, 5)] = math.sqrt(2.0)
    block_energy = (
        channel_energy
        * channel_gains.view(1, channel_count, 1)
    ).sum(dim=1)
    epsilon = torch.finfo(block_energy.dtype).tiny
    block_loudness = -0.691 + 10.0 * torch.log10(
        block_energy.clamp_min(epsilon)
    )
    absolute_mask = block_loudness > -70.0
    absolute_count = absolute_mask.sum(dim=-1).clamp_min(1)
    absolute_energy = (
        block_energy * absolute_mask
    ).sum(dim=-1) / absolute_count
    relative_threshold = (
        -0.691
        + 10.0 * torch.log10(absolute_energy.clamp_min(epsilon))
        - 10.0
    )
    gated_mask = absolute_mask & (
        block_loudness > relative_threshold.unsqueeze(-1)
    )
    gated_count = gated_mask.sum(dim=-1).clamp_min(1)
    gated_energy = (
        block_energy * gated_mask
    ).sum(dim=-1) / gated_count
    loudness = -0.691 + 10.0 * torch.log10(
        gated_energy.clamp_min(epsilon)
    )
    minimum = torch.full_like(loudness, -70.0)
    return torch.maximum(
        torch.nan_to_num(loudness, nan=-70.0, neginf=-70.0),
        minimum,
    )


def normalize_loudness(
    audio: torch.Tensor,
    sample_rate: int,
    target_db: float | torch.Tensor,
    *,
    peak_limit: float = 1.0,
) -> torch.Tensor:
    """Normalize waveform loudness and prevent samples from clipping."""
    tensor, original_ndim = _coerce_audio_tensor(audio)
    output_dtype = tensor.dtype
    normalized = tensor.to(dtype=torch.float32)
    current_db = integrated_loudness(normalized, sample_rate)
    target = torch.as_tensor(
        target_db,
        dtype=normalized.dtype,
        device=normalized.device,
    )
    gain = torch.pow(
        torch.tensor(10.0, dtype=normalized.dtype, device=normalized.device),
        (target - current_db) / 20.0,
    )
    normalized = normalized * gain[..., None, None]

    if math.isfinite(peak_limit):
        peak = normalized.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(
            peak > peak_limit,
            torch.as_tensor(
                peak_limit,
                dtype=normalized.dtype,
                device=normalized.device,
            ) / peak.clamp_min(torch.finfo(normalized.dtype).tiny),
            torch.ones_like(peak),
        )
        normalized = normalized * scale
    if output_dtype.is_floating_point:
        normalized = normalized.to(dtype=output_dtype)
    return _restore_audio_shape(normalized, original_ndim)


class AudioSignal:
    """Tensor-backed subset of ``audiotools.AudioSignal`` used by DAC codecs."""

    def __init__(self, audio: Any, sample_rate: int):
        if sample_rate is None or int(sample_rate) <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        tensor, _ = _coerce_audio_tensor(audio)
        self.sample_rate = int(sample_rate)
        self._audio_data = tensor
        self._loudness: torch.Tensor | None = None

    @classmethod
    def load_from_file_with_ffmpeg(cls, path: str | Path) -> AudioSignal:
        """Load channel-preserving PCM WAVE without FFmpeg or SoundFile."""
        source_path = Path(path).expanduser()
        if source_path.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError(
                "The native codec file boundary accepts uncompressed PCM "
                "WAVE input. Decode other containers to a tensor first."
            )
        audio, sample_rate = load_pcm_wave(
            source_path,
            preserve_channels=True,
        )
        return cls(audio, sample_rate)

    @property
    def audio_data(self) -> torch.Tensor:
        return self._audio_data

    @audio_data.setter
    def audio_data(self, value: torch.Tensor) -> None:
        tensor, _ = _coerce_audio_tensor(value)
        self._audio_data = tensor
        self._loudness = None

    @property
    def device(self) -> torch.device:
        return self.audio_data.device

    @property
    def rate(self) -> int:
        return self.sample_rate

    @property
    def signal_length(self) -> int:
        return int(self.audio_data.shape[-1])

    @property
    def signal_duration(self) -> float:
        return self.signal_length / self.sample_rate

    @property
    def shape(self) -> torch.Size:
        return self.audio_data.shape

    def clone(self) -> AudioSignal:
        return type(self)(self.audio_data.clone(), self.sample_rate)

    def write(self, path: str | Path) -> AudioSignal:
        """Write the first batch item as portable 16-bit PCM WAVE."""
        if self.audio_data.shape[0] != 1:
            raise ValueError(
                "AudioSignal.write requires one batch item; index the "
                "desired item first."
            )
        save_pcm_wave(
            path,
            self.audio_data[0],
            self.sample_rate,
        )
        return self

    def to(self, *args: Any, **kwargs: Any) -> AudioSignal:
        self.audio_data = self.audio_data.to(*args, **kwargs)
        return self

    def __getitem__(self, item: Any) -> AudioSignal:
        return type(self)(self.audio_data[item], self.sample_rate)

    def zero_pad(self, left: int, right: int) -> AudioSignal:
        if left < 0 or right < 0:
            raise ValueError("Padding values must be non-negative.")
        self.audio_data = torch.nn.functional.pad(
            self.audio_data, (int(left), int(right)))
        return self

    def resample(self, sample_rate: int) -> AudioSignal:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if sample_rate == self.sample_rate:
            return self

        shape = self.audio_data.shape
        flattened = self.audio_data.reshape(-1, shape[-1])
        self.audio_data = torch.stack(
            tuple(
                resample_waveform(
                    channel,
                    self.sample_rate,
                    sample_rate,
                )
                for channel in flattened
            ),
            dim=0,
        ).reshape(
            *shape[:-1],
            -1,
        )
        self.sample_rate = sample_rate
        return self

    ffmpeg_resample = resample

    def loudness(self) -> torch.Tensor:
        if self._loudness is None:
            self._loudness = integrated_loudness(
                self.audio_data, self.sample_rate)
        return self._loudness.to(self.device)

    ffmpeg_loudness = loudness

    def normalize(
        self,
        target_db: float | torch.Tensor = -24.0,
    ) -> AudioSignal:
        self.audio_data = normalize_loudness(
            self.audio_data,
            self.sample_rate,
            target_db,
            peak_limit=float("inf"),
        )
        return self

    def ensure_max_of_audio(self, maximum: float = 1.0) -> AudioSignal:
        if maximum <= 0:
            raise ValueError("maximum must be positive.")
        peak = self.audio_data.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(
            peak > maximum,
            maximum / peak.clamp_min(torch.finfo(self.audio_data.dtype).tiny),
            torch.ones_like(peak),
        )
        self.audio_data = self.audio_data * scale
        return self


__all__ = [
    "AudioSignal",
    "BaseModel",
    "integrated_loudness",
    "normalize_loudness",
]
