"""
Lightweight prompt builder for MOSS-TTS-Delay inference.

Constructs the multi-channel input_ids tensor (S, 33) from text and optional
reference audio codes — without PyTorch or HuggingFace transformers.

Uses the ``tokenizers`` library (Rust-backed, no torch dependency) for BPE
tokenization compatible with the Qwen3 tokenizer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

from ._constants import (
    N_VQ,
    AUDIO_PAD_CODE,
    AUDIO_START_TOKEN_ID,
    AUDIO_END_TOKEN_ID,
    AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
    AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
    AUDIO_USER_SLOT_TOKEN_ID,
    IM_START_TOKEN_ID,
    IM_END_TOKEN_ID,
)
from .delay_state import apply_delay_pattern, apply_de_delay_pattern, extract_audio_segments

log = logging.getLogger(__name__)

AUDIO_PLACEHOLDER = "<|audio|>"


class Tokenizer:
    """Thin wrapper around the HuggingFace ``tokenizers`` library."""

    def __init__(self, tokenizer_dir: str | Path):
        from tokenizers import Tokenizer as HFTokenizer

        tokenizer_path = Path(tokenizer_dir) / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {tokenizer_dir}")
        self._tok = HFTokenizer.from_file(str(tokenizer_path))
        log.info("Tokenizer loaded from %s (vocab=%d)", tokenizer_path, self._tok.get_vocab_size())

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def id_to_token(self, token_id: int) -> str | None:
        return self._tok.id_to_token(token_id)


def _get_special_token_str(tokenizer: Tokenizer, token_id: int) -> str:
    tok = tokenizer.id_to_token(token_id)
    if tok is None:
        raise ValueError(f"Token ID {token_id} not in vocabulary")
    return tok


def build_generation_prompt(
    tokenizer: Tokenizer,
    text: str,
    reference_codes: np.ndarray | None = None,
    instruction: str | None = None,
    tokens: int | None = None,
    quality: str | None = None,
    language: str | None = None,
    sound_event: str | None = None,
    ambient_sound: str | None = None,
) -> np.ndarray:
    """Build the full multi-channel input_ids for generation.

    Returns:
        input_ids: (S, 33) int64
    """
    audio_start_tok = _get_special_token_str(tokenizer, AUDIO_START_TOKEN_ID)
    audio_end_tok = _get_special_token_str(tokenizer, AUDIO_END_TOKEN_ID)
    gen_slot_tok = _get_special_token_str(tokenizer, AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID)
    delay_slot_tok = _get_special_token_str(tokenizer, AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID)
    user_slot_tok = _get_special_token_str(tokenizer, AUDIO_USER_SLOT_TOKEN_ID)

    has_ref = reference_codes is not None and reference_codes.shape[0] > 0

    if has_ref:
        ref_str = f"[S1]:\n{AUDIO_PLACEHOLDER}"
    else:
        ref_str = "None"

    user_content = (
        f"<user_inst>\n"
        f"- Reference(s):\n{ref_str}\n"
        f"- Instruction:\n{instruction}\n"
        f"- Tokens:\n{tokens}\n"
        f"- Quality:\n{quality}\n"
        f"- Sound Event:\n{sound_event}\n"
        f"- Ambient Sound:\n{ambient_sound}\n"
        f"- Language:\n{language}\n"
        f"- Text:\n{text}\n"
        f"</user_inst>"
    )

    ref_lengths = [reference_codes.shape[0]] if has_ref else []
    user_content = _replace_audio_placeholders(
        user_content, ref_lengths, n_vq=N_VQ,
        gen_slot_token=user_slot_tok, delay_slot_token=user_slot_tok,
        audio_start_token=audio_start_tok, audio_end_token=audio_end_tok,
    )

    im_start = _get_special_token_str(tokenizer, IM_START_TOKEN_ID)
    im_end = _get_special_token_str(tokenizer, IM_END_TOKEN_ID)

    full_text = f"{im_start}user\n{user_content}{im_end}\n{im_start}assistant\n"

    ref_audio_list = [reference_codes] if has_ref else []
    unified_codes = _get_unified_codes(
        tokenizer, full_text, ref_audio_list,
        is_user=True, truncation=False,
    )

    assistant_gen = f"{audio_start_tok}"
    gen_ids = np.array(tokenizer.encode(assistant_gen), dtype=np.int64)
    gen_multi = np.full((len(gen_ids), 1 + N_VQ), AUDIO_PAD_CODE, dtype=np.int64)
    gen_multi[:, 0] = gen_ids

    return np.concatenate([unified_codes, gen_multi], axis=0)


def _replace_audio_placeholders(
    content: str,
    lengths: list[int],
    n_vq: int,
    gen_slot_token: str,
    delay_slot_token: str,
    audio_start_token: str,
    audio_end_token: str,
) -> str:
    num_ph = content.count(AUDIO_PLACEHOLDER)
    if num_ph != len(lengths):
        raise ValueError(
            f"Placeholder count ({num_ph}) != lengths count ({len(lengths)})"
        )

    lengths_iter = iter(lengths)

    def _build_block(length: int) -> str:
        if length == 0:
            return f"{audio_start_token}{audio_end_token}"
        step_tokens = gen_slot_token * length + delay_slot_token * (n_vq - 1)
        return f"{audio_start_token}{step_tokens}{audio_end_token}"

    def replacer(match: re.Match) -> str:
        return _build_block(next(lengths_iter))

    return re.sub(re.escape(AUDIO_PLACEHOLDER), replacer, content)


def _get_unified_codes(
    tokenizer: Tokenizer,
    content: str,
    audio_codes_list: list[np.ndarray],
    is_user: bool = True,
    truncation: bool = False,
) -> np.ndarray:
    """Build the multi-channel (text + audio) packed sequence."""
    text_ids = np.array(tokenizer.encode(content), dtype=np.int64)
    n_vq = N_VQ

    if len(audio_codes_list) == 0:
        audio_channel = np.full((len(text_ids), n_vq), AUDIO_PAD_CODE, dtype=np.int64)
        return np.concatenate([text_ids[:, np.newaxis], audio_channel], axis=1)

    audio_start_indices = np.where(text_ids == AUDIO_START_TOKEN_ID)[0]
    audio_end_indices = np.where(text_ids == AUDIO_END_TOKEN_ID)[0]

    if len(audio_start_indices) != len(audio_codes_list) or len(audio_end_indices) != len(audio_codes_list):
        raise ValueError(
            f"Audio markers ({len(audio_start_indices)} starts, {len(audio_end_indices)} ends) "
            f"don't match codes ({len(audio_codes_list)})"
        )

    delay_parts: list[np.ndarray] = []
    prefix_idx = 0

    for start_idx, end_idx, codes in zip(audio_start_indices, audio_end_indices, audio_codes_list):
        start_idx = int(start_idx)
        end_idx = int(end_idx)

        delayed = apply_delay_pattern(codes, AUDIO_PAD_CODE)

        pad_before = np.full(
            (start_idx - prefix_idx + 1, n_vq), AUDIO_PAD_CODE, dtype=np.int64,
        )
        delay_parts.extend([pad_before, delayed])
        prefix_idx = end_idx

    if truncation:
        delay_parts[-1] = delay_parts[-1][:-(n_vq - 1), :]
    else:
        last_end = int(audio_end_indices[-1])
        pad_after = np.full(
            (len(text_ids) - last_end, n_vq), AUDIO_PAD_CODE, dtype=np.int64,
        )
        delay_parts.append(pad_after)

    delay_audio = np.concatenate(delay_parts, axis=0)

    if len(text_ids) != delay_audio.shape[0]:
        text_ids = text_ids[:delay_audio.shape[0]]

    return np.concatenate([text_ids[:, np.newaxis], delay_audio], axis=1)


def parse_generation_output(
    tokenizer: Tokenizer,
    generation_ids: np.ndarray,
    prompt_len: int,
) -> tuple[str, np.ndarray]:
    """Parse the generated output into text content and audio codes."""
    gen_part = generation_ids[prompt_len:]
    text_channel = gen_part[:, 0].tolist()
    audio_channels = gen_part[:, 1:]

    audio_start_tok = _get_special_token_str(tokenizer, AUDIO_START_TOKEN_ID)
    gen_slot_tok = _get_special_token_str(tokenizer, AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID)
    delay_slot_tok = _get_special_token_str(tokenizer, AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID)
    audio_end_tok = _get_special_token_str(tokenizer, AUDIO_END_TOKEN_ID)

    raw_text = tokenizer.decode(text_channel)

    pattern = re.compile(
        rf"(?:{re.escape(audio_start_tok)})?"
        rf"(?:{re.escape(gen_slot_tok)})*"
        rf"(?:{re.escape(delay_slot_tok)})*"
        rf"{re.escape(audio_end_tok)}"
    )

    def repl(match: re.Match) -> str:
        seg = match.group(0)
        if gen_slot_tok in seg:
            return AUDIO_PLACEHOLDER
        return ""

    text = pattern.sub(repl, raw_text)

    segments = extract_audio_segments(audio_channels)

    if segments:
        audio_codes = np.concatenate(segments, axis=0)
    else:
        audio_codes = np.zeros((0, N_VQ), dtype=np.int64)

    return text, audio_codes
