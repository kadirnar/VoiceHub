"""Dependency-free implementation of the official Llama 3 text split."""

from __future__ import annotations

import unicodedata

LLAMA3_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
    r"\s+(?!\S)|\s+")
_CONTRACTIONS = ("re", "ve", "ll", "s", "t", "m", "d")


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_punctuation_piece(character: str) -> bool:
    return (not character.isspace() and not _is_letter(character) and not _is_number(character))


def llama3_pretokenize(text: str) -> tuple[str, ...]:
    """Apply Llama 3's Unicode split expression using only the stdlib.

    The ordered scanner is equivalent to :data:`LLAMA3_SPLIT_PATTERN`
    and preserves every source character. Number runs are intentionally
    divided into groups of at most three digits, matching the released
    tokenizer.
    """
    if not isinstance(text, str):
        raise TypeError("`text` must be a string.")
    pieces: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        start = index
        if text[index] == "'":
            remaining = text[index + 1:]
            contraction = next(
                (suffix for suffix in _CONTRACTIONS if remaining[:len(suffix)].lower() == suffix),
                None,
            )
            if contraction is not None:
                index += len(contraction) + 1
                pieces.append(text[start:index])
                continue

        if _is_letter(text[index]):
            index += 1
            while index < length and _is_letter(text[index]):
                index += 1
            pieces.append(text[start:index])
            continue
        if (text[index] not in "\r\n" and not _is_number(text[index]) and index + 1 < length and
                _is_letter(text[index + 1])):
            index += 2
            while index < length and _is_letter(text[index]):
                index += 1
            pieces.append(text[start:index])
            continue
        if _is_number(text[index]):
            index += 1
            while (index < length and index - start < 3 and _is_number(text[index])):
                index += 1
            pieces.append(text[start:index])
            continue

        punctuation_start = index
        if (text[index] == " " and index + 1 < length and _is_punctuation_piece(text[index + 1])):
            index += 1
        if index < length and _is_punctuation_piece(text[index]):
            index += 1
            while index < length and _is_punctuation_piece(text[index]):
                index += 1
            while index < length and text[index] in "\r\n":
                index += 1
            pieces.append(text[punctuation_start:index])
            continue
        index = start

        if text[index].isspace():
            whitespace_end = index + 1
            last_newline = index if text[index] in "\r\n" else None
            while whitespace_end < length and text[whitespace_end].isspace():
                if text[whitespace_end] in "\r\n":
                    last_newline = whitespace_end
                whitespace_end += 1
            if last_newline is not None:
                index = last_newline + 1
                pieces.append(text[start:index])
                continue
            index = (
                whitespace_end -
                1 if whitespace_end < length and whitespace_end - start > 1 else whitespace_end)
            pieces.append(text[start:index])
            continue

        index += 1
        pieces.append(text[start:index])

    result = tuple(pieces)
    if "".join(result) != text:
        raise RuntimeError("Llama 3 pretokenization did not preserve source text.")
    return result


__all__ = ["LLAMA3_SPLIT_PATTERN", "llama3_pretokenize"]
