"""Dependency-light request and result types for causal token serving."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real


def _positive_token_ids(
    value: Sequence[int],
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError(f"`{name}` must be a sequence of token IDs.")
    normalized = []
    for token in value:
        if (isinstance(token, bool) or not isinstance(token, Integral) or not 0 <= token <= 2**63 - 1):
            raise ValueError(f"`{name}` must contain non-negative signed 64-bit integers.")
        normalized.append(int(token))
    if not normalized and not allow_empty:
        raise ValueError(f"`{name}` cannot be empty.")
    return tuple(normalized)


def _finite_real(
    value,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"`{name}` must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"`{name}` must be finite.")
    if minimum is not None:
        invalid = (normalized < minimum if minimum_inclusive else normalized <= minimum)
        if invalid:
            comparator = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"`{name}` must be {comparator} {minimum}.")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"`{name}` must be at most {maximum}.")
    return normalized


@dataclass(frozen=True, slots=True)
class TokenGenerationRequest:
    """Tokenizer-free request sent to a flat causal-LM server."""

    prompt_token_ids: Sequence[int]
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: Sequence[int] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_token_ids",
            _positive_token_ids(
                self.prompt_token_ids,
                name="prompt_token_ids",
                allow_empty=False,
            ),
        )
        object.__setattr__(
            self,
            "stop_token_ids",
            _positive_token_ids(
                self.stop_token_ids,
                name="stop_token_ids",
                allow_empty=True,
            ),
        )
        if (isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, Integral) or
                self.max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        object.__setattr__(self, "max_new_tokens", int(self.max_new_tokens))
        object.__setattr__(
            self,
            "temperature",
            _finite_real(
                self.temperature,
                name="temperature",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "repetition_penalty",
            _finite_real(
                self.repetition_penalty,
                name="repetition_penalty",
                minimum=0.0,
                minimum_inclusive=False,
            ),
        )
        for name in ("top_p", "min_p"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _finite_real(
                        value,
                        name=name,
                        minimum=0.0,
                        minimum_inclusive=name == "min_p",
                        maximum=1.0,
                    ),
                )
        if self.top_k is not None:
            if (isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral) or self.top_k < 0):
                raise ValueError("`top_k` must be a non-negative integer or None.")
            object.__setattr__(self, "top_k", int(self.top_k))
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
                raise TypeError("`seed` must be an integer or None.")
            seed = int(self.seed)
            if not 0 <= seed <= 2**63 - 1:
                raise ValueError("`seed` must be a non-negative signed 64-bit integer.")
            object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class TokenGenerationResult:
    """Generated suffix IDs and optional engine accounting."""

    token_ids: Sequence[int]
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_ids",
            _positive_token_ids(
                self.token_ids,
                name="token_ids",
                allow_empty=True,
            ),
        )


__all__ = [
    "TokenGenerationRequest",
    "TokenGenerationResult",
]
