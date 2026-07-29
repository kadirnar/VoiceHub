"""Native greedy, prefix-beam, and timestamp decoding for CTC models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

_NEGATIVE_INFINITY = -float("inf")


def _log_add(*values: float) -> float:
    finite = tuple(value for value in values if value != _NEGATIVE_INFINITY)
    if not finite:
        return _NEGATIVE_INFINITY
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(value - maximum) for value in finite))


def _validate_inputs(
    logits: Tensor,
    lengths: Tensor | None,
    blank_id: int,
) -> Tensor:
    if not isinstance(logits, Tensor) or logits.ndim != 3:
        raise ValueError("CTC logits must have shape [batch, time, vocabulary].")
    if not logits.is_floating_point():
        raise TypeError("CTC logits must use a floating-point dtype.")
    if logits.shape[0] == 0 or logits.shape[1] == 0 or logits.shape[2] < 2:
        raise ValueError(
            "CTC logits require non-empty batch/time axes and at least two "
            "vocabulary entries.")
    if isinstance(blank_id, bool) or not isinstance(blank_id, int):
        raise TypeError("`blank_id` must be an integer.")
    if not 0 <= blank_id < logits.shape[-1]:
        raise ValueError("`blank_id` is outside the CTC vocabulary.")
    if lengths is None:
        return torch.full(
            (logits.shape[0], ),
            logits.shape[1],
            dtype=torch.long,
            device=logits.device,
        )
    if not isinstance(lengths, Tensor):
        raise TypeError("`lengths` must be a PyTorch tensor or None.")
    if (lengths.ndim != 1 or lengths.shape[0] != logits.shape[0] or lengths.dtype == torch.bool or
            lengths.is_floating_point() or lengths.is_complex()):
        raise ValueError("`lengths` must be an integer vector with one value per row.")
    if (lengths <= 0).any() or (lengths > logits.shape[1]).any():
        raise ValueError("CTC lengths must be in the interval [1, time].")
    return lengths


@dataclass(frozen=True)
class CTCTokenSpan:
    """Frame interval assigned to one decoded non-blank token."""

    token_id: int
    start_frame: int
    end_frame: int
    score: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("CTC token spans require 0 <= start < end.")
        if not math.isfinite(self.score):
            raise ValueError("CTC token span score must be finite.")


@dataclass(frozen=True)
class CTCDecodeResult:
    """One collapsed CTC hypothesis."""

    tokens: tuple[int, ...]
    score: float
    token_spans: tuple[CTCTokenSpan, ...] = ()


def _collapse_greedy(
    token_ids: Sequence[int],
    frame_scores: Sequence[float],
    *,
    blank_id: int,
) -> tuple[tuple[int, ...], tuple[CTCTokenSpan, ...]]:
    tokens: list[int] = []
    spans: list[CTCTokenSpan] = []
    previous = blank_id
    active_token: int | None = None
    active_start = 0
    active_scores: list[float] = []

    def finish(end_frame: int) -> None:
        nonlocal active_token, active_scores
        if active_token is None:
            return
        tokens.append(active_token)
        spans.append(
            CTCTokenSpan(
                token_id=active_token,
                start_frame=active_start,
                end_frame=end_frame,
                score=sum(active_scores) / len(active_scores),
            ))
        active_token = None
        active_scores = []

    for frame, (token, score) in enumerate(zip(token_ids, frame_scores)):
        if token == blank_id:
            finish(frame)
            previous = blank_id
            continue
        if token != previous:
            finish(frame)
            active_token = token
            active_start = frame
        active_scores.append(score)
        previous = token
    finish(len(token_ids))
    return tuple(tokens), tuple(spans)


def ctc_greedy_decode(
    logits: Tensor,
    *,
    lengths: Tensor | None = None,
    blank_id: int = 0,
) -> tuple[CTCDecodeResult, ...]:
    """Collapse framewise argmax paths with confidence and frame spans."""
    lengths = _validate_inputs(logits, lengths, blank_id)
    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
    frame_scores, frame_tokens = log_probabilities.max(dim=-1)
    results = []
    for row, length in enumerate(lengths.tolist()):
        tokens, spans = _collapse_greedy(
            frame_tokens[row, :length].tolist(),
            frame_scores[row, :length].exp().tolist(),
            blank_id=blank_id,
        )
        score = float(frame_scores[row, :length].sum().item())
        results.append(CTCDecodeResult(
            tokens=tokens,
            score=score,
            token_spans=spans,
        ))
    return tuple(results)


def _hotword_score(
    prefix: tuple[int, ...],
    hotwords: Mapping[tuple[int, ...], float],
) -> float:
    score = 0.0
    for phrase, weight in hotwords.items():
        phrase_length = len(phrase)
        if phrase_length > len(prefix):
            continue
        for start in range(len(prefix) - phrase_length + 1):
            if prefix[start:start + phrase_length] == phrase:
                score += weight
    return score


def _normalize_hotwords(hotwords: Mapping[Sequence[int], float] | None, ) -> dict[tuple[int, ...], float]:
    normalized: dict[tuple[int, ...], float] = {}
    for tokens, weight in (hotwords or {}).items():
        phrase = tuple(tokens)
        if (not phrase or
                any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in phrase)):
            raise ValueError("CTC hotwords must contain non-negative token IDs.")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError("CTC hotword weights must be real numbers.")
        weight = float(weight)
        if not math.isfinite(weight):
            raise ValueError("CTC hotword weights must be finite.")
        if phrase in normalized:
            raise ValueError(f"Duplicate CTC hotword phrase {phrase!r}.")
        normalized[phrase] = weight
    return normalized


def _prefix_beam_row(
    log_probabilities: Tensor,
    *,
    blank_id: int,
    beam_size: int,
    token_beam_size: int,
    hotwords: Mapping[tuple[int, ...], float],
) -> tuple[tuple[int, ...], float]:
    # Each prefix owns its blank-ending and non-blank-ending acoustic scores.
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, _NEGATIVE_INFINITY)}
    for frame in log_probabilities:
        candidate_scores, candidate_ids = frame.topk(min(token_beam_size, frame.shape[0]))
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def update(
            prefix: tuple[int, ...],
            *,
            blank: float = _NEGATIVE_INFINITY,
            non_blank: float = _NEGATIVE_INFINITY,
        ) -> None:
            previous_blank, previous_non_blank = next_beams.get(
                prefix,
                (_NEGATIVE_INFINITY, _NEGATIVE_INFINITY),
            )
            next_beams[prefix] = (
                _log_add(previous_blank, blank),
                _log_add(previous_non_blank, non_blank),
            )

        for prefix, (blank_score, non_blank_score) in beams.items():
            total_score = _log_add(blank_score, non_blank_score)
            for token, token_score in zip(
                    candidate_ids.tolist(),
                    candidate_scores.tolist(),
            ):
                if token == blank_id:
                    update(
                        prefix,
                        blank=total_score + token_score,
                    )
                    continue
                if prefix and token == prefix[-1]:
                    # Repeating without a separating blank leaves the collapsed
                    # prefix unchanged. Repeating after a blank creates a new
                    # instance of the same token.
                    update(
                        prefix,
                        non_blank=non_blank_score + token_score,
                    )
                    repeated = prefix + (token, )
                    update(
                        repeated,
                        non_blank=blank_score + token_score,
                    )
                else:
                    extended = prefix + (token, )
                    update(
                        extended,
                        non_blank=total_score + token_score,
                    )
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: (
                    _log_add(*item[1]) + _hotword_score(item[0], hotwords),
                    item[0],
                ),
                reverse=True,
            )[:beam_size])
    prefix, scores = max(
        beams.items(),
        key=lambda item: (
            _log_add(*item[1]) + _hotword_score(item[0], hotwords),
            item[0],
        ),
    )
    return prefix, _log_add(*scores) + _hotword_score(prefix, hotwords)


def ctc_prefix_beam_search(
    logits: Tensor,
    *,
    lengths: Tensor | None = None,
    blank_id: int = 0,
    beam_size: int = 10,
    token_beam_size: int | None = None,
    hotwords: Mapping[Sequence[int], float] | None = None,
    return_timestamps: bool = True,
) -> tuple[CTCDecodeResult, ...]:
    """Decode globally competing CTC prefixes with optional hotword bias.

    Hotword scores are applied only when a complete token phrase occurs;
    no reward is given to unfinished prefixes. Timestamp spans are
    recovered with Viterbi forced alignment of the winning token
    sequence.
    """
    lengths = _validate_inputs(logits, lengths, blank_id)
    if isinstance(beam_size, bool) or not isinstance(beam_size, int) or beam_size <= 0:
        raise ValueError("`beam_size` must be a positive integer.")
    token_beam_size = beam_size if token_beam_size is None else token_beam_size
    if (isinstance(token_beam_size, bool) or not isinstance(token_beam_size, int) or token_beam_size <= 0):
        raise ValueError("`token_beam_size` must be a positive integer.")
    if not isinstance(return_timestamps, bool):
        raise TypeError("`return_timestamps` must be a boolean.")
    normalized_hotwords = _normalize_hotwords(hotwords)
    vocabulary_size = logits.shape[-1]
    if any(token >= vocabulary_size for phrase in normalized_hotwords for token in phrase):
        raise ValueError("A CTC hotword token is outside the vocabulary.")

    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
    results = []
    for row, length in enumerate(lengths.tolist()):
        row_probabilities = log_probabilities[row, :length]
        tokens, score = _prefix_beam_row(
            row_probabilities,
            blank_id=blank_id,
            beam_size=beam_size,
            token_beam_size=token_beam_size,
            hotwords=normalized_hotwords,
        )
        spans = (
            ctc_forced_alignment(
                row_probabilities,
                tokens,
                blank_id=blank_id,
                log_probabilities=True,
            ) if return_timestamps and tokens else ())
        results.append(CTCDecodeResult(
            tokens=tokens,
            score=score,
            token_spans=spans,
        ))
    return tuple(results)


def ctc_forced_alignment(
    emissions: Tensor,
    tokens: Sequence[int],
    *,
    blank_id: int = 0,
    log_probabilities: bool = False,
) -> tuple[CTCTokenSpan, ...]:
    """Viterbi-align a known CTC token sequence to emission frames."""
    if not isinstance(emissions, Tensor) or emissions.ndim != 2:
        raise ValueError("CTC alignment emissions must have shape [time, vocabulary].")
    if not emissions.is_floating_point():
        raise TypeError("CTC alignment emissions must use a floating-point dtype.")
    token_ids = tuple(tokens)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 or
           token >= emissions.shape[-1] or token == blank_id for token in token_ids):
        raise ValueError("Alignment tokens must be non-blank IDs inside the vocabulary.")
    if not token_ids:
        return ()
    minimum_frames = len(token_ids) + sum(left == right for left, right in zip(token_ids, token_ids[1:]))
    if emissions.shape[0] < minimum_frames:
        raise ValueError(f"Cannot align {len(token_ids)} CTC tokens in "
                         f"{emissions.shape[0]} frames.")
    log_probs = emissions.float()
    if not log_probabilities:
        log_probs = functional.log_softmax(log_probs, dim=-1)
    if not bool(torch.isfinite(log_probs).any(dim=-1).all()):
        raise ValueError("Every alignment frame needs a finite emission.")

    states = [blank_id]
    for token in token_ids:
        states.extend((token, blank_id))
    state_tensor = torch.tensor(states, dtype=torch.long, device=emissions.device)
    time_steps = emissions.shape[0]
    state_count = len(states)
    scores = torch.full(
        (time_steps, state_count),
        _NEGATIVE_INFINITY,
        dtype=torch.float32,
        device=emissions.device,
    )
    predecessors = torch.full(
        (time_steps, state_count),
        -1,
        dtype=torch.long,
        device=emissions.device,
    )
    scores[0, 0] = log_probs[0, blank_id]
    scores[0, 1] = log_probs[0, token_ids[0]]

    for frame in range(1, time_steps):
        for state in range(state_count):
            candidates = [(scores[frame - 1, state], state)]
            if state > 0:
                candidates.append((scores[frame - 1, state - 1], state - 1))
            if (state > 1 and states[state] != blank_id and states[state] != states[state - 2]):
                candidates.append((scores[frame - 1, state - 2], state - 2))
            best_score, best_state = max(
                candidates,
                key=lambda item: float(item[0].item()),
            )
            scores[frame, state] = best_score + log_probs[
                frame,
                state_tensor[state],
            ]
            predecessors[frame, state] = best_state

    final_candidates = (
        (scores[-1, state_count - 1], state_count - 1),
        (scores[-1, state_count - 2], state_count - 2),
    )
    final_score, state = max(
        final_candidates,
        key=lambda item: float(item[0].item()),
    )
    if not bool(torch.isfinite(final_score)):
        raise ValueError("No valid CTC alignment path exists.")
    path = [state]
    for frame in range(time_steps - 1, 0, -1):
        state = int(predecessors[frame, state].item())
        if state < 0:
            raise RuntimeError("CTC alignment backtracking reached an invalid state.")
        path.append(state)
    path.reverse()

    spans = []
    for token_index, token in enumerate(token_ids):
        token_state = 2 * token_index + 1
        frames = [index for index, value in enumerate(path) if value == token_state]
        if not frames:
            raise RuntimeError("CTC alignment omitted a required token state.")
        probabilities = log_probs[frames, token].exp()
        spans.append(
            CTCTokenSpan(
                token_id=token,
                start_frame=frames[0],
                end_frame=frames[-1] + 1,
                score=float(probabilities.mean().item()),
            ))
    return tuple(spans)


__all__ = [
    "CTCDecodeResult",
    "CTCTokenSpan",
    "ctc_forced_alignment",
    "ctc_greedy_decode",
    "ctc_prefix_beam_search",
]
