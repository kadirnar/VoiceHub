"""Lightweight output contracts for the Parler-TTS DAC compatibility model.

The original wrapper used Transformers' Encodec output containers even though
the underlying codec is DAC.  Keeping the small mapping-and-tuple protocol
local removes that unrelated runtime dependency while preserving the behavior
expected by Parler-TTS generation code.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Any


class _DACModelOutput(OrderedDict):
    """Dataclass-backed output with mapping and positional access.

    Non-``None`` fields are exposed as mapping entries. String keys provide
    mapping access, while integer and slice keys address the populated values
    as a tuple. This is the subset of the model-output protocol used by the
    vendored Parler-TTS compatibility runtime.
    """

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                OrderedDict.__setitem__(self, field.name, value)

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            return OrderedDict.__getitem__(self, key)
        return self.to_tuple()[key]

    def __setattr__(self, name: str, value: Any) -> None:
        dataclass_fields = getattr(type(self), "__dataclass_fields__", {})
        if name in dataclass_fields and value is not None:
            OrderedDict.__setitem__(self, name, value)
        object.__setattr__(self, name, value)

    def __setitem__(self, key: str, value: Any) -> None:
        OrderedDict.__setitem__(self, key, value)
        object.__setattr__(self, key, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError(
            f"{type(self).__name__} entries cannot be deleted.",
        )

    def pop(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError(f"{type(self).__name__} entries cannot be removed.")

    def setdefault(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError(f"{type(self).__name__} entries cannot be added.")

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError(f"{type(self).__name__} entries cannot be updated.")

    def to_tuple(self) -> tuple[Any, ...]:
        """Return populated values in declaration order."""
        return tuple(OrderedDict.__getitem__(self, key) for key in self.keys())


@dataclass
class DACEncoderOutput(_DACModelOutput):
    """Discrete DAC codes and their optional per-frame scale metadata."""

    audio_codes: Any | None = None
    audio_scales: Any | None = None
    last_frame_pad_length: int | None = None


@dataclass
class DACDecoderOutput(_DACModelOutput):
    """Waveform reconstructed by the DAC decoder."""

    audio_values: Any | None = None


# Compatibility aliases for callers that referenced the historical container
# names through this module. They remain VoiceHub-owned classes.
EncodecEncoderOutput = DACEncoderOutput
EncodecDecoderOutput = DACDecoderOutput


__all__ = [
    "DACDecoderOutput",
    "DACEncoderOutput",
    "EncodecDecoderOutput",
    "EncodecEncoderOutput",
]
