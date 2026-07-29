"""Dependency-free rich notebook views for Vui audio and spectrograms."""

from __future__ import annotations

import base64
import html
import io
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.processing.waveform import load_pcm_wave


@dataclass(frozen=True, slots=True)
class NotebookAudio:
    """Jupyter-renderable PCM WAVE payload."""

    wav_bytes: bytes
    autoplay: bool = True

    def _repr_html_(self) -> str:
        encoded = base64.b64encode(self.wav_bytes).decode("ascii")
        autoplay = " autoplay" if self.autoplay else ""
        return (f'<audio controls{autoplay} src="data:audio/wav;base64,'
                f'{encoded}"></audio>')


def _wave_bytes(audio: Tensor, sample_rate: int) -> bytes:
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("`sample_rate` must be a positive integer.")
    values = audio.detach().float().cpu()
    if values.ndim == 2:
        values = values.mean(dim=0)
    values = values.reshape(-1)
    if values.numel() <= 100:
        raise ValueError("Notebook audio must contain more than 100 samples.")
    if not torch.isfinite(values).all():
        raise ValueError("Notebook audio cannot contain NaN or infinity.")
    peak = values.abs().max()
    if peak > 1.0:
        values = values / peak
    pcm = (values.clamp(-1.0, 1.0) * 32_767.0).round().to(torch.int16)
    payload = array("h", pcm.tolist())
    if payload.itemsize != 2:
        raise RuntimeError("The platform does not expose 16-bit signed shorts.")
    if sys.byteorder != "little":
        payload.byteswap()
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(payload.tobytes())
    return output.getvalue()


def play(
    audio: Tensor | list[float] | tuple[float, ...] | str | Path,
    sr: int = 16_000,
    autoplay: bool = True,
) -> NotebookAudio:
    """Return an inline-renderable audio object for the last notebook cell."""
    if isinstance(audio, (str, Path)):
        values, source_rate = load_pcm_wave(audio)
        if source_rate != sr:
            from voicehub.processing.waveform import resample_waveform

            values = resample_waveform(values, source_rate, sr)
    else:
        values = audio if isinstance(audio, Tensor) else torch.tensor(audio)
    return NotebookAudio(
        wav_bytes=_wave_bytes(values, sr),
        autoplay=bool(autoplay),
    )


@dataclass(frozen=True, slots=True)
class MelSpectrogramView:
    """Compact SVG heatmap rendered directly by Jupyter."""

    values: Tensor
    title: str | None = None

    def _repr_svg_(self) -> str:
        values = self.values.detach().float().cpu().squeeze()
        if values.ndim != 2 or values.numel() == 0:
            raise ValueError("Mel values must have shape [channels, frames].")
        row_stride = max(1, values.shape[0] // 96)
        column_stride = max(1, values.shape[1] // 256)
        values = values[::row_stride, ::column_stride]
        minimum = values.min()
        scale = (values.max() - minimum).clamp_min(1e-8)
        normalized = (values - minimum) / scale
        cell_width = 900 / values.shape[1]
        cell_height = 280 / values.shape[0]
        rectangles: list[str] = []
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                intensity = float(normalized[row, column])
                red = round(255 * intensity)
                green = round(100 * intensity)
                blue = round(255 * (1.0 - intensity))
                rectangles.append(
                    f'<rect x="{column * cell_width:.3f}" '
                    f'y="{(values.shape[0] - row - 1) * cell_height:.3f}" '
                    f'width="{cell_width + 0.1:.3f}" '
                    f'height="{cell_height + 0.1:.3f}" '
                    f'fill="rgb({red},{green},{blue})"/>')
        title = (
            "" if self.title is None else f'<text x="450" y="20" text-anchor="middle">'
            f'{html.escape(self.title)}</text>')
        offset = 28 if title else 0
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" '
            'viewBox="0 0 900 320" role="img">'
            f'{title}<g transform="translate(0,{offset})">' + "".join(rectangles) + "</g></svg>")


def plot_mel_spec(
    mel_spec: Tensor | Any,
    title: str | None = None,
) -> MelSpectrogramView:
    """Return a Jupyter-renderable spectrogram without NumPy or Matplotlib."""
    return MelSpectrogramView(
        values=torch.as_tensor(mel_spec),
        title=title,
    )


__all__ = [
    "MelSpectrogramView",
    "NotebookAudio",
    "play",
    "plot_mel_spec",
]
