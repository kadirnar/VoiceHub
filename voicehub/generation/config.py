"""Validated configuration for VoiceHub's native generation engine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any

_TORCH_SEED_MIN = -(2**63)
_TORCH_SEED_MAX = 2**64 - 1


def _validate_integer(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"`{name}` must be greater than or equal to {minimum}.")
    return result


def _validate_probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"`{name}` must be finite and in the interval [0, 1].")
    return result


def _normalize_eos_token_ids(value: int | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, Integral) and not isinstance(value, bool):
        values = (value, )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        raise TypeError("`eos_token_id` must be an integer, a sequence of integers, or None.")
    if not values:
        raise ValueError("`eos_token_id` cannot be an empty tuple.")

    normalized = tuple(_validate_integer("eos_token_id", item, minimum=0) for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("`eos_token_id` cannot contain duplicates.")
    return normalized


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Options shared by native autoregressive speech architectures.

    ``seed`` initializes a request-owned :class:`torch.Generator`; it
    never mutates PyTorch's process-wide random state. A tuple may be
    used for ``eos_token_id`` when a model has more than one terminal
    token.
    """

    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    eos_token_id: int | Sequence[int] | None = None
    pad_token_id: int | None = None
    seed: int | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_new_tokens",
            _validate_integer("max_new_tokens", self.max_new_tokens, minimum=1),
        )
        if not isinstance(self.do_sample, bool):
            raise TypeError("`do_sample` must be a boolean.")
        if not isinstance(self.use_cache, bool):
            raise TypeError("`use_cache` must be a boolean.")

        if isinstance(self.temperature, bool) or not isinstance(self.temperature, Real):
            raise TypeError("`temperature` must be a real number.")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("`temperature` must be finite and greater than zero.")
        object.__setattr__(self, "temperature", temperature)

        if self.top_k is not None:
            object.__setattr__(
                self,
                "top_k",
                _validate_integer("top_k", self.top_k, minimum=0),
            )
        if self.top_p is not None:
            object.__setattr__(self, "top_p", _validate_probability("top_p", self.top_p))
        if self.min_p is not None:
            object.__setattr__(self, "min_p", _validate_probability("min_p", self.min_p))

        if isinstance(self.repetition_penalty, bool) or not isinstance(self.repetition_penalty, Real):
            raise TypeError("`repetition_penalty` must be a real number.")
        penalty = float(self.repetition_penalty)
        if not math.isfinite(penalty) or penalty <= 0.0:
            raise ValueError("`repetition_penalty` must be finite and greater than zero.")
        object.__setattr__(self, "repetition_penalty", penalty)

        eos_token_ids = _normalize_eos_token_ids(self.eos_token_id)
        normalized_eos: int | tuple[int, ...] | None
        if not eos_token_ids:
            normalized_eos = None
        elif len(eos_token_ids) == 1:
            normalized_eos = eos_token_ids[0]
        else:
            normalized_eos = eos_token_ids
        object.__setattr__(self, "eos_token_id", normalized_eos)

        if self.pad_token_id is not None:
            object.__setattr__(
                self,
                "pad_token_id",
                _validate_integer("pad_token_id", self.pad_token_id, minimum=0),
            )
        if self.seed is not None:
            seed = _validate_integer("seed", self.seed)
            if not _TORCH_SEED_MIN <= seed <= _TORCH_SEED_MAX:
                raise ValueError(
                    "`seed` must be in PyTorch's supported interval "
                    f"[{_TORCH_SEED_MIN}, {_TORCH_SEED_MAX}].")
            object.__setattr__(self, "seed", seed)

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        """Return terminal token IDs in a uniform representation."""
        return _normalize_eos_token_ids(self.eos_token_id)

    @property
    def effective_pad_token_id(self) -> int | None:
        """Return the explicit pad token or the first terminal token."""
        if self.pad_token_id is not None:
            return self.pad_token_id
        return self.eos_token_ids[0] if self.eos_token_ids else None

    def with_updates(self, **updates: Any) -> GenerationConfig:
        """Return a validated copy with selected fields replaced."""
        return replace(self, **updates)
