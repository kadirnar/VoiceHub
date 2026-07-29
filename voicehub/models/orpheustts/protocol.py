"""Orpheus conversation and SNAC token protocol.

The language model represents each 24 kHz SNAC frame with seven tokens:
one token from the first hierarchy, two from the second, and four from
the third.  Each position owns a disjoint 4,096-token range.  Keeping
the mapping here prevents training and inference from drifting apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Any

START_SPEECH_TOKEN_ID = 128257
END_SPEECH_TOKEN_ID = 128258
START_HUMAN_TOKEN_ID = 128259
END_HUMAN_TOKEN_ID = 128260
START_AI_TOKEN_ID = 128261
END_AI_TOKEN_ID = 128262
PAD_TOKEN_ID = 128263
END_TEXT_TOKEN_ID = 128009
AUDIO_TOKEN_OFFSET = 128266
SNAC_CODEBOOK_SIZE = 4096
SNAC_FRAME_WIDTH = 7
SNAC_CHANNEL_OFFSETS = tuple(channel * SNAC_CODEBOOK_SIZE for channel in range(SNAC_FRAME_WIDTH))


def _integer_values(values: Any, *, layer: int) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "reshape"):
        values = values.reshape(-1).tolist()
    try:
        items = list(values)
    except TypeError as error:
        raise TypeError(f"SNAC hierarchy layer {layer} must be an iterable of token IDs.") from error
    normalized: list[int] = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("SNAC hierarchy layers must contain integer token IDs.")
        token_id = int(value)
        if not 0 <= token_id < SNAC_CODEBOOK_SIZE:
            raise ValueError(
                f"SNAC hierarchy layer {layer} contains out-of-range code "
                f"{token_id}; expected [0, {SNAC_CODEBOOK_SIZE}).")
        normalized.append(token_id)
    return normalized


def interleave_snac_codes(
    layers: Sequence[Any],
    *,
    deduplicate_frames: bool = True,
) -> list[int]:
    """Convert three raw SNAC hierarchy layers to Orpheus token IDs."""
    if not isinstance(deduplicate_frames, bool):
        raise TypeError("`deduplicate_frames` must be a boolean.")
    if len(layers) != 3:
        raise ValueError("Orpheus SNAC codes must contain three hierarchy layers.")
    layer_1, layer_2, layer_3 = (
        _integer_values(values, layer=index) for index, values in enumerate(layers, start=1))
    frame_count = len(layer_1)
    if frame_count == 0:
        raise ValueError("Orpheus SNAC codes cannot be empty.")
    if len(layer_2) != 2 * frame_count or len(layer_3) != 4 * frame_count:
        raise ValueError(
            "SNAC hierarchy lengths must have a 1:2:4 ratio; received "
            f"{frame_count}:{len(layer_2)}:{len(layer_3)}.")

    frames = ((
        layer_1[index],
        layer_2[2 * index],
        layer_3[4 * index],
        layer_3[4 * index + 1],
        layer_2[2 * index + 1],
        layer_3[4 * index + 2],
        layer_3[4 * index + 3],
    ) for index in range(frame_count))
    encoded: list[int] = []
    previous_first: int | None = None
    for frame in frames:
        if deduplicate_frames and frame[0] == previous_first:
            continue
        encoded.extend(
            AUDIO_TOKEN_OFFSET + SNAC_CHANNEL_OFFSETS[channel] + value for channel, value in enumerate(frame))
        previous_first = frame[0]
    if not encoded:
        raise ValueError("SNAC frame deduplication removed every audio frame.")
    return encoded


def normalize_orpheus_audio_tokens(tokens: Iterable[int]) -> list[int]:
    """Validate interleaved Orpheus audio IDs and remove the global offset."""
    try:
        values = tuple(tokens)
    except TypeError as error:
        raise TypeError("Orpheus audio tokens must be iterable.") from error
    if not values or len(values) % SNAC_FRAME_WIDTH:
        raise ValueError(
            "Orpheus audio tokens must contain one or more complete "
            f"{SNAC_FRAME_WIDTH}-token SNAC frames.")
    codes: list[int] = []
    for position, token in enumerate(values):
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise TypeError("Orpheus audio tokens must contain integer IDs.")
        code = int(token) - AUDIO_TOKEN_OFFSET
        channel = position % SNAC_FRAME_WIDTH
        lower_bound = SNAC_CHANNEL_OFFSETS[channel]
        upper_bound = lower_bound + SNAC_CODEBOOK_SIZE
        if not lower_bound <= code < upper_bound:
            raise ValueError("Invalid Orpheus SNAC token for channel "
                             f"{channel}: {int(token)}.")
        codes.append(code)
    return codes


def deinterleave_snac_codes(relative_codes: Sequence[int], ) -> tuple[list[int], list[int], list[int]]:
    """Convert validated offset-relative frame codes to three SNAC layers."""
    if not relative_codes or len(relative_codes) % SNAC_FRAME_WIDTH:
        raise ValueError("SNAC decoding requires complete seven-code frames.")
    groups = len(relative_codes) // SNAC_FRAME_WIDTH
    layer_1 = [int(relative_codes[SNAC_FRAME_WIDTH * index]) for index in range(groups)]
    layer_2 = [
        value for index in range(groups) for value in (
            int(relative_codes[SNAC_FRAME_WIDTH * index + 1]) - SNAC_CHANNEL_OFFSETS[1],
            int(relative_codes[SNAC_FRAME_WIDTH * index + 4]) - SNAC_CHANNEL_OFFSETS[4],
        )
    ]
    layer_3 = [
        value for index in range(groups) for value in (
            int(relative_codes[SNAC_FRAME_WIDTH * index + 2]) - SNAC_CHANNEL_OFFSETS[2],
            int(relative_codes[SNAC_FRAME_WIDTH * index + 3]) - SNAC_CHANNEL_OFFSETS[3],
            int(relative_codes[SNAC_FRAME_WIDTH * index + 5]) - SNAC_CHANNEL_OFFSETS[5],
            int(relative_codes[SNAC_FRAME_WIDTH * index + 6]) - SNAC_CHANNEL_OFFSETS[6],
        )
    ]
    return layer_1, layer_2, layer_3


__all__ = [
    "AUDIO_TOKEN_OFFSET",
    "END_AI_TOKEN_ID",
    "END_HUMAN_TOKEN_ID",
    "END_SPEECH_TOKEN_ID",
    "END_TEXT_TOKEN_ID",
    "PAD_TOKEN_ID",
    "SNAC_CHANNEL_OFFSETS",
    "SNAC_CODEBOOK_SIZE",
    "SNAC_FRAME_WIDTH",
    "START_AI_TOKEN_ID",
    "START_HUMAN_TOKEN_ID",
    "START_SPEECH_TOKEN_ID",
    "deinterleave_snac_codes",
    "interleave_snac_codes",
    "normalize_orpheus_audio_tokens",
]
