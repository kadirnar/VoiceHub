"""Native grammar-constrained decoding for Tiron token streams.

The state machine follows the public Tiron inference harness at revision
``d249c5a81fc6e0f1ecd34fd30cf2519f06fe671c``. The upstream implementation
is Apache-2.0 licensed. This port targets VoiceHub's model-neutral generation
engine and has no dependency on Transformers.

Only the public checkpoint's production ``speaker_blocks`` grammar is exposed:

* the first generated token is ``speaker1`` or ``nospeech``;
* a speaker token must be followed by an opening timestamp;
* text may continue until the aggregate timestamp probability wins;
* a closing timestamp may continue the current block, finish, or introduce
  exactly the next contiguous speaker slot; and
* undeclared padded vocabulary rows are never eligible for generation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import torch
from torch import Tensor

from voicehub.models.asr_tiron.metadata import (
    EOS_TOKEN_ID,
    NO_SPEECH_TOKEN_ID,
    NO_TIMESTAMPS_TOKEN_ID,
    SPEAKER_TOKEN_IDS,
    TIMESTAMP_BEGIN_ID,
    TIMESTAMP_END_ID,
    TIRON_CHECKPOINT_REVISION,
    TIRON_HARNESS_REVISION,
)

NO_REPEAT_NGRAM_SIZE = 15
MAX_INITIAL_TIMESTAMP_INDEX = 1_500
NEGATIVE_INFINITY = float("-inf")


def _integer(
    name: str,
    value: Any,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"`{name}` must be an integer.")
    result = int(value)
    if result < minimum:
        raise ValueError(f"`{name}` must be at least {minimum}.")
    return result


def _speaker_ids(values: Mapping[int, int] | Sequence[int], ) -> tuple[int, ...]:
    if isinstance(values, Mapping):
        if not values:
            raise ValueError("At least one Tiron speaker token is required.")
        expected_slots = tuple(range(1, len(values) + 1))
        if tuple(sorted(values)) != expected_slots:
            raise ValueError("Tiron speaker-token mappings must use contiguous one-based "
                             "slots.")
        resolved = tuple(_integer(
            f"speaker_token_ids[{slot}]",
            values[slot],
        ) for slot in expected_slots)
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        resolved = tuple(_integer(f"speaker_token_ids[{index}]", value) for index, value in enumerate(values))
    else:
        raise TypeError("`speaker_token_ids` must be a mapping or ordered sequence.")
    if not resolved:
        raise ValueError("At least one Tiron speaker token is required.")
    if len(set(resolved)) != len(resolved):
        raise ValueError("Tiron speaker token IDs must be distinct.")
    if tuple(sorted(resolved)) != resolved:
        raise ValueError("Tiron speaker token IDs must be strictly increasing.")
    return resolved


def validate_public_tiron_token_layout(
    *,
    eos_token_id: int,
    no_speech_token_id: int,
    no_timestamps_token_id: int,
    timestamp_begin_id: int,
    timestamp_end_id: int,
    speaker_token_ids: Sequence[int],
) -> None:
    """Fail loudly if an artifact no longer matches the published layout."""
    actual = {
        "eos_token_id": eos_token_id,
        "no_speech_token_id": no_speech_token_id,
        "no_timestamps_token_id": no_timestamps_token_id,
        "timestamp_begin_id": timestamp_begin_id,
        "timestamp_end_id": timestamp_end_id,
        "speaker_token_ids": tuple(speaker_token_ids),
    }
    expected = {
        "eos_token_id": EOS_TOKEN_ID,
        "no_speech_token_id": NO_SPEECH_TOKEN_ID,
        "no_timestamps_token_id": NO_TIMESTAMPS_TOKEN_ID,
        "timestamp_begin_id": TIMESTAMP_BEGIN_ID,
        "timestamp_end_id": TIMESTAMP_END_ID,
        "speaker_token_ids": SPEAKER_TOKEN_IDS,
    }
    drift = {name: (expected[name], actual[name]) for name in expected if actual[name] != expected[name]}
    if drift:
        details = ", ".join(
            f"{name}: expected {wanted!r}, found {found!r}" for name, (wanted, found) in drift.items())
        raise ValueError(
            "The Tiron tokenizer layout differs from the pinned public "
            f"checkpoint ({details}).")


def _is_timestamp(
    token_id: int,
    *,
    timestamp_begin_id: int,
    timestamp_end_id: int,
) -> bool:
    return timestamp_begin_id <= token_id <= timestamp_end_id


def _blocked_ngram_tokens(
    token_ids: list[int],
    ngram_size: int,
) -> set[int]:
    """Return next IDs that would recreate an existing n-gram."""
    if ngram_size <= 0 or len(token_ids) < ngram_size:
        return set()
    if ngram_size == 1:
        return set(token_ids)
    prefix = tuple(token_ids[-(ngram_size - 1):])
    blocked: set[int] = set()
    for index in range(len(token_ids) - ngram_size + 1):
        if tuple(token_ids[index:index + ngram_size - 1]) == prefix:
            blocked.add(token_ids[index + ngram_size - 1])
    return blocked


class TironConstraintLogitsProcessor:
    """Mask logits to the public Tiron ``speaker_blocks`` grammar.

    The processor derives every row's state from ``input_ids`` and
    therefore keeps no cross-request mutable state. Instances may safely
    be reused by sequential generation requests.
    """

    def __init__(
        self,
        *,
        prompt_length: int = 3,
        speaker_token_ids: Mapping[int, int] | Sequence[int] = SPEAKER_TOKEN_IDS,
        timestamp_begin_id: int = TIMESTAMP_BEGIN_ID,
        timestamp_end_id: int = TIMESTAMP_END_ID,
        no_timestamps_token_id: int = NO_TIMESTAMPS_TOKEN_ID,
        no_speech_token_id: int = NO_SPEECH_TOKEN_ID,
        eos_token_id: int = EOS_TOKEN_ID,
        declared_token_count: int | None = None,
        max_speakers: int | None = None,
        allow_initial_no_speech: bool = True,
        no_repeat_ngram_size: int = NO_REPEAT_NGRAM_SIZE,
        max_initial_timestamp_index: int = MAX_INITIAL_TIMESTAMP_INDEX,
    ) -> None:
        self.prompt_length = _integer(
            "prompt_length",
            prompt_length,
            minimum=1,
        )
        self.speaker_token_ids = _speaker_ids(speaker_token_ids)
        self.timestamp_begin_id = _integer(
            "timestamp_begin_id",
            timestamp_begin_id,
        )
        self.timestamp_end_id = _integer(
            "timestamp_end_id",
            timestamp_end_id,
        )
        self.no_timestamps_token_id = _integer(
            "no_timestamps_token_id",
            no_timestamps_token_id,
        )
        self.no_speech_token_id = _integer(
            "no_speech_token_id",
            no_speech_token_id,
        )
        self.eos_token_id = _integer("eos_token_id", eos_token_id)
        if self.timestamp_end_id < self.timestamp_begin_id:
            raise ValueError("`timestamp_end_id` must not precede "
                             "`timestamp_begin_id`.")
        if self.no_timestamps_token_id + 1 != self.timestamp_begin_id:
            raise ValueError(
                "Tiron requires the timestamp range immediately after "
                "`no_timestamps_token_id`.")
        if self.speaker_token_ids[0] != self.timestamp_end_id + 1:
            raise ValueError("Tiron speaker tokens must immediately follow the timestamp "
                             "range.")
        if any(right != left + 1 for left, right in zip(
                self.speaker_token_ids,
                self.speaker_token_ids[1:],
        )):
            raise ValueError("Tiron speaker token IDs must be contiguous.")

        if declared_token_count is None:
            declared_token_count = self.speaker_token_ids[-1] + 1
        self.declared_token_count = _integer(
            "declared_token_count",
            declared_token_count,
            minimum=self.speaker_token_ids[-1] + 1,
        )
        if max_speakers is None:
            self.max_speakers = len(self.speaker_token_ids)
        else:
            self.max_speakers = _integer(
                "max_speakers",
                max_speakers,
                minimum=1,
            )
            if self.max_speakers > len(self.speaker_token_ids):
                raise ValueError("`max_speakers` exceeds the checkpoint's speaker slots.")
        if not isinstance(allow_initial_no_speech, bool):
            raise TypeError("`allow_initial_no_speech` must be a boolean.")
        self.allow_initial_no_speech = allow_initial_no_speech
        self.no_repeat_ngram_size = _integer(
            "no_repeat_ngram_size",
            no_repeat_ngram_size,
        )
        self.max_initial_timestamp_index = _integer(
            "max_initial_timestamp_index",
            max_initial_timestamp_index,
        )
        available_timestamp_count = (self.timestamp_end_id - self.timestamp_begin_id + 1)
        if self.max_initial_timestamp_index >= available_timestamp_count:
            self.max_initial_timestamp_index = available_timestamp_count - 1

    def __call__(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        """Return logits masked independently for every batch row."""
        if not isinstance(input_ids, Tensor) or not isinstance(logits, Tensor):
            raise TypeError("Tiron token history and logits must be tensors.")
        if input_ids.ndim != 2 or logits.ndim != 2:
            raise ValueError("Tiron expects [batch, sequence] IDs and "
                             "[batch, vocabulary] logits.")
        if input_ids.shape[0] != logits.shape[0] or input_ids.shape[0] < 1:
            raise ValueError("Tiron token-history and logits batches must match.")
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError("Tiron token history is shorter than its prompt.")
        if (input_ids.dtype == torch.bool or input_ids.is_floating_point() or input_ids.is_complex()):
            raise TypeError("Tiron token history must use an integer dtype.")
        if not logits.is_floating_point():
            raise TypeError("Tiron logits must use a floating-point dtype.")
        if input_ids.device != logits.device:
            raise ValueError("Tiron token history and logits must share a device.")
        required_vocabulary = max(
            self.declared_token_count,
            self.eos_token_id + 1,
        )
        if logits.shape[1] < required_vocabulary:
            raise ValueError("Tiron logits are smaller than the declared token space.")

        for row_index in range(input_ids.shape[0]):
            generated = input_ids[
                row_index,
                self.prompt_length:,
            ].tolist()
            self._apply_row(logits[row_index], generated)
        return logits

    def _next_speaker_ids(self, generated: list[int]) -> tuple[int, ...]:
        effective = self.speaker_token_ids[:self.max_speakers]
        seen = [effective.index(value) for value in generated if value in effective]
        if not seen:
            return (effective[0], )
        next_index = max(seen) + 1
        if next_index >= len(effective):
            return ()
        return (effective[next_index], )

    def _force_timestamps(self, row: Tensor) -> None:
        row[:self.timestamp_begin_id] = NEGATIVE_INFINITY
        row[self.timestamp_end_id + 1:] = NEGATIVE_INFINITY

    def _force_text(self, row: Tensor) -> None:
        row[self.eos_token_id:] = NEGATIVE_INFINITY

    def _apply_row(self, row: Tensor, generated: list[int]) -> None:
        if self.declared_token_count < row.shape[0]:
            row[self.declared_token_count:] = NEGATIVE_INFINITY
        row[self.no_timestamps_token_id] = NEGATIVE_INFINITY
        no_speech_score = row[self.no_speech_token_id].clone()
        row[self.no_speech_token_id] = NEGATIVE_INFINITY
        eos_score = row[self.eos_token_id].clone()

        if not generated:
            speaker_score = row[self.speaker_token_ids[0]].clone()
            row[:] = NEGATIVE_INFINITY
            row[self.speaker_token_ids[0]] = speaker_score
            if self.allow_initial_no_speech:
                row[self.no_speech_token_id] = no_speech_score
            return

        if generated[-1] == self.no_speech_token_id:
            row[:] = NEGATIVE_INFINITY
            row[self.eos_token_id] = eos_score
            return

        last = generated[-1]
        last_is_timestamp = _is_timestamp(
            last,
            timestamp_begin_id=self.timestamp_begin_id,
            timestamp_end_id=self.timestamp_end_id,
        )
        previous_is_timestamp = (
            len(generated) >= 2 and _is_timestamp(
                generated[-2],
                timestamp_begin_id=self.timestamp_begin_id,
                timestamp_end_id=self.timestamp_end_id,
            ))
        previous_is_speaker = (len(generated) >= 2 and generated[-2] in self.speaker_token_ids)
        allow_eos = True

        if last in self.speaker_token_ids:
            is_first_timestamp = not any(
                _is_timestamp(
                    value,
                    timestamp_begin_id=self.timestamp_begin_id,
                    timestamp_end_id=self.timestamp_end_id,
                ) for value in generated)
            self._force_timestamps(row)
            if is_first_timestamp:
                last_allowed = (self.timestamp_begin_id + self.max_initial_timestamp_index)
                row[last_allowed + 1:self.timestamp_end_id + 1] = NEGATIVE_INFINITY
            allow_eos = False
        elif last_is_timestamp:
            if previous_is_speaker or previous_is_timestamp:
                self._force_text(row)
                allow_eos = False
            else:
                timestamp_scores = row[self.timestamp_begin_id:self.timestamp_end_id + 1].clone()
                next_speakers = self._next_speaker_ids(generated)
                speaker_scores = {token_id: row[token_id].clone() for token_id in next_speakers}
                row[:] = NEGATIVE_INFINITY
                row[self.timestamp_begin_id:self.timestamp_end_id + 1] = timestamp_scores
                for token_id, score in speaker_scores.items():
                    row[token_id] = score
        else:
            # Within text, only ordinary text or a closing timestamp is legal.
            row[self.eos_token_id:self.timestamp_begin_id] = NEGATIVE_INFINITY
            row[self.timestamp_end_id + 1:] = NEGATIVE_INFINITY
            allow_eos = False

            log_probabilities = torch.nn.functional.log_softmax(
                row.float(),
                dim=-1,
            )
            timestamp_mass = log_probabilities[self.timestamp_begin_id:self.timestamp_end_id +
                                               1].logsumexp(dim=-1)
            maximum_text_probability = log_probabilities[:self.eos_token_id].max()
            if timestamp_mass > maximum_text_probability:
                row[:self.timestamp_begin_id] = NEGATIVE_INFINITY
                row[self.timestamp_end_id + 1:] = NEGATIVE_INFINITY

        if self.no_repeat_ngram_size:
            for token_id in _blocked_ngram_tokens(
                    generated,
                    self.no_repeat_ngram_size,
            ):
                if 0 <= token_id < row.shape[0]:
                    row[token_id] = NEGATIVE_INFINITY

        if allow_eos:
            row[self.eos_token_id] = eos_score


__all__ = [
    "EOS_TOKEN_ID",
    "MAX_INITIAL_TIMESTAMP_INDEX",
    "NO_REPEAT_NGRAM_SIZE",
    "NO_SPEECH_TOKEN_ID",
    "NO_TIMESTAMPS_TOKEN_ID",
    "SPEAKER_TOKEN_IDS",
    "TIMESTAMP_BEGIN_ID",
    "TIMESTAMP_END_ID",
    "TIRON_CHECKPOINT_REVISION",
    "TIRON_HARNESS_REVISION",
    "TironConstraintLogitsProcessor",
    "validate_public_tiron_token_layout",
]
