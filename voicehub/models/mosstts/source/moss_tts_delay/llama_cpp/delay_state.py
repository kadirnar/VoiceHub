"""
Delay state machine for MOSS-TTS-Delay autoregressive generation.

Ports the delay-pattern generation logic from
``moss_tts_delay/modeling_moss_tts.py:generate()`` to pure NumPy.
Operates on batch_size=1 for simplicity (consumer hardware target).

The delay pattern introduces a diagonal time-shift across codebooks:
  Head 0 predicts text at t, Head k predicts codebook k-1 at t-(k-1).
  During the "delay slot" flush after audio ends, the model emits n_vq
  additional steps to drain the staircase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ._constants import (
    N_VQ,
    PAD_TOKEN_ID,
    IM_START_TOKEN_ID,
    IM_END_TOKEN_ID,
    AUDIO_START_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_USER_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
    AUDIO_PAD_CODE,
)
from .sampling import sample_token

log = logging.getLogger(__name__)

INT64_MAX = np.iinfo(np.int64).max


@dataclass
class SamplingConfig:
    """Hyperparameters for the generation loop."""

    text_temperature: float = 1.5
    text_top_p: float = 1.0
    text_top_k: int = 50
    audio_temperature: float = 1.7
    audio_top_p: float = 0.8
    audio_top_k: int = 25
    audio_repetition_penalty: float = 1.0


@dataclass
class DelayState:
    """Mutable state for one generation sequence (batch_size=1)."""

    audio_length: int = 0
    delayed_length: int = INT64_MAX
    is_audio: bool = False
    is_stopping: bool = False
    time_step: int = 0
    text_history: list[int] = field(default_factory=list)

    _audio_buf: np.ndarray | None = field(default=None, repr=False)
    _audio_len: int = 0
    _audio_cap: int = 0

    def audio_history(self) -> np.ndarray | None:
        """Return (T, N_VQ) int64 view of audio history, or None if empty."""
        if self._audio_len == 0:
            return None
        return self._audio_buf[:self._audio_len]

    def append_audio(self, codes: np.ndarray) -> None:
        """Append a single (N_VQ,) frame to the audio history buffer."""
        if self._audio_len >= self._audio_cap:
            new_cap = max(self._audio_cap * 2, 256)
            new_buf = np.empty((new_cap, N_VQ), dtype=np.int64)
            if self._audio_buf is not None:
                new_buf[:self._audio_len] = self._audio_buf[:self._audio_len]
            self._audio_buf = new_buf
            self._audio_cap = new_cap
        self._audio_buf[self._audio_len] = codes
        self._audio_len += 1


_PRE_EXCLUDE_IDS = np.array([
    PAD_TOKEN_ID,
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
], dtype=np.int64)

_AUDIO_ALLOWED_IDS = np.array([
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
], dtype=np.int64)


def init_delay_state(input_ids: np.ndarray) -> DelayState:
    """Initialize the delay state from the prefill input_ids.

    Args:
        input_ids: (S, 33) — the full prompt sequence (already packed).
    """
    state = DelayState()
    seq_len = input_ids.shape[0]
    text_channel = input_ids[:, 0]

    last_text_token = int(text_channel[-1])
    is_continuation = (
        last_text_token == AUDIO_START_TOKEN_ID
        or last_text_token == AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID
    )

    if is_continuation:
        audio_start_idx = _find_last_equal(text_channel, AUDIO_START_TOKEN_ID)
        if audio_start_idx >= 0:
            state.audio_length = seq_len - audio_start_idx
            state.is_audio = True

    state.text_history = text_channel.tolist()

    cap = max(seq_len + 1024, 256)
    state._audio_buf = np.empty((cap, N_VQ), dtype=np.int64)
    state._audio_buf[:seq_len] = input_ids[:, 1:]
    state._audio_len = seq_len
    state._audio_cap = cap

    return state


def step(
    state: DelayState,
    text_logits: np.ndarray,
    audio_logits: np.ndarray,
    config: SamplingConfig,
) -> np.ndarray:
    """Execute one autoregressive step.

    Returns:
        next_input_ids: (33,) int64 — [text_token, audio_0, ..., audio_31]
    """
    if state.is_stopping:
        pad_result = np.full(1 + N_VQ, AUDIO_PAD_CODE, dtype=np.int64)
        pad_result[0] = PAD_TOKEN_ID
        return pad_result

    n_vq = N_VQ

    # --- Text token decision ---
    if state.delayed_length < n_vq:
        next_text = AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID
    elif state.delayed_length == n_vq:
        next_text = AUDIO_END_TOKEN_ID
        state.is_audio = False
    else:
        text_temp = config.text_temperature if config.text_temperature > 0 else 1.0
        text_do_sample = config.text_temperature > 0
        scaled = text_logits / text_temp

        if not state.is_audio:
            scaled[_PRE_EXCLUDE_IDS] = -np.inf
        else:
            mask = np.ones(scaled.shape[0], dtype=bool)
            mask[_AUDIO_ALLOWED_IDS] = False
            scaled[mask] = -np.inf

        if state.time_step == 0:
            scaled[AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID] = -np.inf
        if state.time_step <= n_vq:
            scaled[IM_END_TOKEN_ID] = -np.inf

        next_text = int(sample_token(
            scaled[np.newaxis, :],
            top_p=config.text_top_p,
            top_k=config.text_top_k,
            do_sample=text_do_sample,
        )[0])

    if next_text == AUDIO_START_TOKEN_ID:
        state.is_audio = True
    if next_text == IM_END_TOKEN_ID:
        state.is_stopping = True

    # --- Audio token decision ---
    next_audio = np.full(n_vq, AUDIO_PAD_CODE, dtype=np.int64)

    pre_audio_mask = np.arange(n_vq) < state.audio_length
    if state.delayed_length == INT64_MAX:
        post_audio_mask = np.ones(n_vq, dtype=bool)
    else:
        post_audio_mask = np.arange(n_vq) > (state.delayed_length - 1)
    sampling_mask = pre_audio_mask & post_audio_mask

    if sampling_mask.any():
        audio_temp = config.audio_temperature if config.audio_temperature > 0 else 1.0
        audio_do_sample = config.audio_temperature > 0

        scaled_audio = audio_logits / audio_temp
        scaled_audio[:, AUDIO_PAD_CODE] = -np.inf

        prev_audio = state.audio_history()

        if sampling_mask[0]:
            ch0_logits = scaled_audio[0:1, :]
            ch0_prev = prev_audio[:, 0:1] if prev_audio is not None else None
            next_audio[0] = int(sample_token(
                ch0_logits,
                prev_tokens=ch0_prev,
                repetition_penalty=config.audio_repetition_penalty,
                top_p=config.audio_top_p,
                top_k=config.audio_top_k,
                do_sample=audio_do_sample,
            )[0])

        rest_mask = sampling_mask[1:]
        if rest_mask.any():
            rest_indices = np.where(rest_mask)[0]
            rest_logits = scaled_audio[1 + rest_indices, :]
            rest_prev = prev_audio[:, 1 + rest_indices] if prev_audio is not None else None
            sampled = sample_token(
                rest_logits,
                prev_tokens=rest_prev,
                repetition_penalty=config.audio_repetition_penalty,
                top_p=config.audio_top_p,
                top_k=config.audio_top_k,
                do_sample=audio_do_sample,
            )
            next_audio[1 + rest_indices] = sampled.astype(np.int64)

    # --- State updates ---
    if next_text in (
        AUDIO_START_TOKEN_ID,
        AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
        AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
    ):
        state.audio_length += 1
    if next_text == AUDIO_END_TOKEN_ID:
        state.audio_length = 0

    if state.delayed_length == INT64_MAX and next_text == AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID:
        state.delayed_length = 0
    if state.delayed_length != INT64_MAX:
        state.delayed_length += 1
    if state.delayed_length > n_vq:
        state.delayed_length = INT64_MAX

    state.time_step += 1
    state.text_history.append(next_text)
    state.append_audio(next_audio)

    result = np.empty(1 + n_vq, dtype=np.int64)
    result[0] = next_text
    result[1:] = next_audio
    return result


def apply_delay_pattern(codes: np.ndarray, pad_code: int = AUDIO_PAD_CODE) -> np.ndarray:
    """Apply delay pattern to audio codes.

    Args:
        codes: (T, n_vq) — original audio codes
    Returns:
        delayed: (T + n_vq - 1, n_vq) — delay-shifted codes
    """
    T, n_vq = codes.shape
    delayed = np.full((T + n_vq - 1, n_vq), pad_code, dtype=codes.dtype)
    for i in range(n_vq):
        delayed[i: i + T, i] = codes[:, i]
    return delayed


def apply_de_delay_pattern(delay_codes: np.ndarray) -> np.ndarray:
    """Remove delay pattern from generated codes."""
    total_len, n_vq = delay_codes.shape
    T = total_len - n_vq + 1
    if T <= 0:
        return np.zeros((0, n_vq), dtype=delay_codes.dtype)
    codes = np.zeros((T, n_vq), dtype=delay_codes.dtype)
    for i in range(n_vq):
        codes[:, i] = delay_codes[i: i + T, i]
    return codes


def extract_audio_segments(generation_audio: np.ndarray) -> list[np.ndarray]:
    """Extract non-padding audio segments after de-delaying."""
    codes = apply_de_delay_pattern(generation_audio)
    if codes.shape[0] == 0:
        return []

    is_pad = np.all(codes == AUDIO_PAD_CODE, axis=1)
    non_pad_idx = np.where(~is_pad)[0]
    if len(non_pad_idx) == 0:
        return []

    segments = []
    start = non_pad_idx[0]
    for i in range(1, len(non_pad_idx)):
        if non_pad_idx[i] != non_pad_idx[i - 1] + 1:
            segments.append(codes[start: non_pad_idx[i - 1] + 1])
            start = non_pad_idx[i]
    segments.append(codes[start: non_pad_idx[-1] + 1])
    return segments


def _find_last_equal(arr: np.ndarray, value: int) -> int:
    matches = np.where(arr == value)[0]
    if len(matches) == 0:
        return -1
    return int(matches[-1])
