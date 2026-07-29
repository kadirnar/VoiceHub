"""Validated, dependency-free configuration for the original Vui graph."""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _integer(name: str, value: Any, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return value


def _finite_float(name: str, value: Any, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"`{name}` must be finite and at least {minimum}.")
    return result


@dataclass(slots=True)
class VuiConfig:
    """Architecture hyperparameters for the released 100M Vui family."""

    max_text_tokens: int = 100
    text_size: int = -1
    max_audio_tokens: int = 100
    n_quantizers: int = 9
    codebook_size: int = 1_000
    special_token_id: int = 1_000
    audio_eos_id: int = 1_001
    audio_pad_id: int = 1_002
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    bias: bool = False
    dropout: float = 0.0
    use_rotary_emb: bool = True
    rope_dim: int | None = None
    rope_theta: float = 10_000.0
    rope_theta_rescale_factor: float = 1.0

    def __post_init__(self) -> None:
        for name in (
                "max_text_tokens",
                "max_audio_tokens",
                "n_quantizers",
                "codebook_size",
                "d_model",
                "n_layers",
                "n_heads",
        ):
            setattr(self, name, _integer(name, getattr(self, name)))
        self.text_size = _integer("text_size", self.text_size, minimum=-1)
        for name in ("special_token_id", "audio_eos_id", "audio_pad_id"):
            setattr(self, name, _integer(name, getattr(self, name), minimum=0))
        for name in ("bias", "use_rotary_emb"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        self.dropout = _finite_float("dropout", self.dropout, minimum=0.0)
        if self.dropout >= 1.0:
            raise ValueError("`dropout` must be smaller than one.")
        self.rope_theta = _finite_float(
            "rope_theta",
            self.rope_theta,
            minimum=1e-12,
        )
        self.rope_theta_rescale_factor = _finite_float(
            "rope_theta_rescale_factor",
            self.rope_theta_rescale_factor,
            minimum=1e-12,
        )
        if self.d_model % self.n_heads:
            raise ValueError("`d_model` must be divisible by `n_heads`.")
        if self.rope_dim is not None:
            self.rope_dim = _integer("rope_dim", self.rope_dim)
            if self.rope_dim > self.d_model // self.n_heads:
                raise ValueError("`rope_dim` cannot exceed one attention head.")
            if self.rope_dim % 2:
                raise ValueError("`rope_dim` must be even.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VuiConfig:
        if not isinstance(values, Mapping):
            raise TypeError("Vui model configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def dict(self) -> dict[str, Any]:
        """Pydantic-compatible serialization retained for old callers."""
        return self.model_dump()


@dataclass(slots=True)
class Config:
    """Top-level released-checkpoint configuration."""

    name: str = "base"
    checkpoint: str | dict[str, Any] | None = None
    model: VuiConfig | Mapping[str, Any] = field(default_factory=VuiConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("`name` must be a non-empty string.")
        valid_checkpoint = isinstance(self.checkpoint, (str, dict))
        if self.checkpoint is not None and not valid_checkpoint:
            raise TypeError("`checkpoint` must be a string, dictionary, or None.")
        if isinstance(self.model, Mapping):
            self.model = VuiConfig.from_dict(self.model)
        elif not isinstance(self.model, VuiConfig):
            raise TypeError("`model` must be a VuiConfig or mapping.")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Config:
        if not isinstance(values, Mapping):
            raise TypeError("Vui configuration must be a mapping.")
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def model_dump(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "model": self.model.model_dump(),
        }

    def dict(self) -> dict[str, Any]:
        """Pydantic-compatible serialization retained for old callers."""
        return self.model_dump()


ALL = []
current_module = sys.modules[__name__]
for name in dir(current_module):
    candidate = getattr(current_module, name)
    if name.isupper() and isinstance(candidate, Config):
        ALL.append(candidate)

CONFIGS = {value.name: value for value in ALL}

__all__ = ["ALL", "CONFIGS", "Config", "VuiConfig"]
