from __future__ import annotations

from typing import Any

from .base import TimestampItem, TimestampLevel, TimestampResult


class StableTSAligner:
    def __init__(
        self,
        model_name: str = "base",
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        try:
            import stable_whisper
        except ImportError as exc:
            raise ImportError(
                "stable-ts is required for timestamp alignment. " 'Install with: pip install "voxcpm[timestamps]"'
            ) from exc

        self.model = stable_whisper.load_model(model_name, device=device)
        self.model_name = model_name
        self.device = device
        self.language = language

    def align(
        self,
        *,
        audio_path: str,
        text: str,
        sample_rate: int | None = None,
        level: TimestampLevel = "word",
    ) -> TimestampResult:
        result = self.model.align(audio_path, text, language=self.language)
        items = extract_timestamp_items(result, level)
        return TimestampResult(
            audio_path=audio_path,
            sample_rate=sample_rate,
            backend="stable-ts",
            level=level,
            text=text,
            items=items,
        )


def extract_timestamp_items(result: Any, level: TimestampLevel) -> list[TimestampItem]:
    segments = _get_value(result, "segments", []) or []
    if level == "segment":
        return [
            TimestampItem(
                text=str(_get_value(segment, "text", "")).strip(),
                start=float(_get_value(segment, "start", 0.0) or 0.0),
                end=float(_get_value(segment, "end", 0.0) or 0.0),
                level="segment",
            )
            for segment in segments
            if str(_get_value(segment, "text", "")).strip()
        ]

    words = []
    for segment in segments:
        for word in _get_value(segment, "words", []) or []:
            text = str(_get_value(word, "word", _get_value(word, "text", ""))).strip()
            if not text:
                continue
            words.append(
                TimestampItem(
                    text=text,
                    start=float(_get_value(word, "start", 0.0) or 0.0),
                    end=float(_get_value(word, "end", 0.0) or 0.0),
                    level="word",
                )
            )

    if level == "char":
        return split_word_items_to_chars(words)
    return words


def split_word_items_to_chars(words: list[TimestampItem]) -> list[TimestampItem]:
    chars = []
    for word in words:
        text = word.text.strip()
        if not text:
            continue

        duration = max(word.end - word.start, 0.0)
        step = duration / len(text)
        for idx, char in enumerate(text):
            chars.append(
                TimestampItem(
                    text=char,
                    start=word.start + idx * step,
                    end=word.start + (idx + 1) * step,
                    level="char",
                )
            )
    return chars


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
