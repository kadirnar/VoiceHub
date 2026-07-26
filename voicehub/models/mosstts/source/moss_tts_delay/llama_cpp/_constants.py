"""
Token ID constants for MOSS-TTS-Delay (torch-free).

Loads from the ``config.json`` shipped alongside the canonical
``MossTTSDelayConfig`` when available, with hardcoded fallbacks that
match ``configuration_moss_tts.py`` defaults.  This module intentionally
avoids importing ``transformers`` or ``torch``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"

_HARDCODED_DEFAULTS = {
    "n_vq": 32,
    "pad_token_id": 151643,
    "im_start_token_id": 151644,
    "im_end_token_id": 151645,
    "audio_start_token_id": 151652,
    "audio_end_token_id": 151653,
    "audio_user_slot_token_id": 151654,
    "audio_assistant_gen_slot_token_id": 151656,
    "audio_assistant_delay_slot_token_id": 151662,
    "audio_pad_code": 1024,
    "audio_vocab_size": 1024,
    "sampling_rate": 24000,
}


def _load_defaults() -> dict:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                cfg = json.load(f)
            log.debug("Loaded token IDs from %s", _CONFIG_PATH)
            merged = {**_HARDCODED_DEFAULTS, **cfg}
            return merged
        except Exception as exc:
            log.warning("Failed to read %s (%s), using hardcoded defaults", _CONFIG_PATH, exc)
    return dict(_HARDCODED_DEFAULTS)


_DEFAULTS = _load_defaults()

N_VQ: int = _DEFAULTS["n_vq"]
PAD_TOKEN_ID: int = _DEFAULTS["pad_token_id"]
IM_START_TOKEN_ID: int = _DEFAULTS["im_start_token_id"]
IM_END_TOKEN_ID: int = _DEFAULTS["im_end_token_id"]
AUDIO_START_TOKEN_ID: int = _DEFAULTS["audio_start_token_id"]
AUDIO_END_TOKEN_ID: int = _DEFAULTS["audio_end_token_id"]
AUDIO_USER_SLOT_TOKEN_ID: int = _DEFAULTS["audio_user_slot_token_id"]
AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID: int = _DEFAULTS["audio_assistant_gen_slot_token_id"]
AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID: int = _DEFAULTS["audio_assistant_delay_slot_token_id"]
AUDIO_PAD_CODE: int = _DEFAULTS["audio_pad_code"]
AUDIO_VOCAB_SIZE: int = _DEFAULTS["audio_vocab_size"]
SAMPLE_RATE: int = _DEFAULTS["sampling_rate"]
