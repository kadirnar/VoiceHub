"""Exact native byte tokenizer used by the released Vui 100M checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class ByteTokenizerOutput:
    """Minimal tokenizer result compatible with the Vui call sites."""

    input_ids: Tensor
    attention_mask: Tensor


class CustomByT5Tokenizer:
    """Dependency-free implementation of Vui's ByT5 token boundary.

    The pinned upstream runtime used ``google/byt5-small`` only as a
    byte tokenizer. ByT5 reserves IDs 0, 1, and 2 for padding, end-of-
    sequence, and unknown input; UTF-8 octets are shifted by three. The
    historical ``vocab_size`` property returns 256 and is preserved
    because it determines the released Vui embedding inventory.
    """

    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 2
    offset = 3
    vocab_size = 256

    def __len__(self) -> int:
        # ByT5 exposes 125 unused extra sentinel tokens through ``len``. Vui
        # never embeds them, but retaining the value helps compatibility.
        return 384

    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> CustomByT5Tokenizer:
        """Return the fixed tokenizer without downloading configuration."""
        return cls()

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **_kwargs: Any,
    ) -> Tensor:
        if not isinstance(text, str):
            raise TypeError("Vui tokenizer input must be a string.")
        token_ids = [octet + self.offset for octet in text.encode("utf-8")]
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(
        self,
        token_ids: Tensor | list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        values = token_ids.tolist() if isinstance(token_ids, Tensor) else list(token_ids)
        octets: list[int] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("Vui token IDs must be integers.")
            if value < self.offset:
                if skip_special_tokens:
                    continue
                raise ValueError("Special Vui token IDs cannot be decoded as UTF-8.")
            octet = value - self.offset
            if not 0 <= octet <= 255:
                if skip_special_tokens:
                    continue
                raise ValueError(f"Invalid Vui byte token ID: {value}.")
            octets.append(octet)
        return bytes(octets).decode("utf-8", errors="replace")

    def __call__(
        self,
        texts: str | list[str] | tuple[str, ...],
        *,
        padding: str | bool = False,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
        **_kwargs: Any,
    ) -> ByteTokenizerOutput:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if not values or any(not isinstance(text, str) for text in values):
            raise TypeError("Vui tokenizer input must contain one or more strings.")
        if padding not in (False, True, "longest"):
            raise ValueError("Vui tokenizer supports only longest-sequence padding.")
        if return_tensors not in (None, "pt"):
            raise ValueError("Vui tokenizer can return PyTorch tensors only.")

        rows = [self.encode(
            text,
            add_special_tokens=add_special_tokens,
        ) for text in values]
        maximum = max(row.numel() for row in rows)
        if len(rows) > 1 and padding is False and any(row.numel() != maximum for row in rows):
            raise ValueError("Variable-length token batches require padding='longest'.")
        input_ids = torch.full(
            (len(rows), maximum),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, :row.numel()] = row
            attention_mask[index, :row.numel()] = 1
        if single and return_tensors is None:
            input_ids = input_ids[0]
            attention_mask = attention_mask[0]
        return ByteTokenizerOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


__all__ = ["ByteTokenizerOutput", "CustomByT5Tokenizer"]
