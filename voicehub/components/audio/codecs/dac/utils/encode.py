"""Programmatic native DAC file encoder."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from voicehub.components.audio.codecs._compat import AudioSignal
from voicehub.components.audio.codecs.dac.utils import load_model

logger = logging.getLogger(__name__)


def _wave_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        if source.suffix.lower() not in {".wav", ".wave"}:
            raise ValueError("Native DAC encoding accepts PCM WAVE files.")
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(f"DAC input was not found: {source}.")
    return tuple(
        sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in {".wav", ".wave"}
        )
    )


def _relative_path(path: Path, source: Path) -> Path:
    return Path(path.name) if source.is_file() else path.relative_to(source)


@torch.inference_mode()
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    n_quantizers: int | None = None,
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 5.0,
    verbose: bool = False,
) -> tuple[Path, ...]:
    """Encode one PCM WAVE file or directory tree to native ``.dac`` files."""
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path or None,
    )
    generator.to(device).eval()
    source = Path(input).expanduser()
    audio_files = _wave_files(source)
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, audio_path in enumerate(audio_files, start=1):
        if verbose:
            logger.info(
                "Encoding DAC file %d/%d: %s",
                index,
                len(audio_files),
                audio_path,
            )
        signal = AudioSignal.load_from_file_with_ffmpeg(audio_path)
        artifact = generator.compress(
            signal,
            win_duration,
            verbose=verbose,
            n_quantizers=n_quantizers,
        )
        relative = _relative_path(audio_path, source).with_suffix(".dac")
        output_path = destination / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        outputs.append(artifact.save(output_path))
    return tuple(outputs)


__all__ = ["encode"]
