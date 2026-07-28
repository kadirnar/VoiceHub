"""Lazy tokenizer cache for MeloTTS language frontends.

Importing an inference runtime must not trigger network access. Tokenizers are
therefore downloaded only when their language frontend is used and are shared
across subsequent synthesis calls.
"""

from functools import lru_cache


@lru_cache(maxsize=None)
def get_tokenizer(model_id: str):
    """Return one cached Transformers tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


__all__ = ["get_tokenizer"]
