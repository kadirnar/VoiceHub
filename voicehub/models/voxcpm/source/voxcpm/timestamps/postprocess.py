from __future__ import annotations

from .base import TimestampLevel


def align_audio_file(
    *,
    audio_path: str,
    text: str,
    sample_rate: int | None = None,
    backend: str = "stable-ts",
    level: TimestampLevel = "word",
    model_name: str = "base",
    device: str | None = None,
    language: str | None = None,
) -> dict:
    if backend != "stable-ts":
        raise ValueError(f"Unsupported timestamp backend: {backend}")

    from .stable_ts import StableTSAligner

    aligner = StableTSAligner(
        model_name=model_name,
        device=device,
        language=language,
    )
    result = aligner.align(
        audio_path=audio_path,
        text=text,
        sample_rate=sample_rate,
        level=level,
    )
    return result.to_dict()
