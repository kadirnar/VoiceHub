from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

TimestampLevel = Literal["segment", "word", "char"]


@dataclass
class TimestampItem:
    text: str
    start: float
    end: float
    level: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimestampResult:
    audio_path: str
    sample_rate: int | None
    backend: str
    level: str
    text: str
    items: list[TimestampItem]
    warning: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["items"] = [item.to_dict() for item in self.items]
        return payload
