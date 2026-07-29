"""Programmatic native DAC archive decoder."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from voicehub.components.audio.codecs.dac import DACFile
from voicehub.components.audio.codecs.dac.utils import load_model

logger = logging.getLogger(__name__)


def _dac_files(source: Path) -> tuple[Path, ...]:
    if source.is_file():
        if source.suffix.lower() != ".dac":
            raise ValueError("Native DAC decoding requires a `.dac` archive.")
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(f"DAC input was not found: {source}.")
    return tuple(sorted(path for path in source.rglob("*.dac") if path.is_file()))


def _relative_path(path: Path, source: Path) -> Path:
    return Path(path.name) if source.is_file() else path.relative_to(source)


@torch.inference_mode()
def decode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    device: str = "cuda",
    model_type: str = "44khz",
    verbose: bool = False,
) -> tuple[Path, ...]:
    """Decode one native ``.dac`` archive or directory tree to PCM WAVE."""
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path or None,
    )
    generator.to(device).eval()
    source = Path(input).expanduser()
    input_files = _dac_files(source)
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, archive_path in enumerate(input_files, start=1):
        if verbose:
            logger.info(
                "Decoding DAC file %d/%d: %s",
                index,
                len(input_files),
                archive_path,
            )
        artifact = DACFile.load(archive_path)
        reconstruction = generator.decompress(artifact, verbose=verbose)
        relative = _relative_path(archive_path, source).with_suffix(".wav")
        output_path = destination / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reconstruction.write(output_path)
        outputs.append(output_path)
    return tuple(outputs)


__all__ = ["decode"]
