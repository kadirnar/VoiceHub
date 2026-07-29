"""Lazy public exports for VoiceHub's native XTTS v2 architecture."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.architectures.xtts2."
_EXPORTS = {
    "XTTS2AudioConfig": _PACKAGE + "configuration",
    "XTTS2CheckpointInventory": _PACKAGE + "checkpoint",
    "XTTS2Config": _PACKAGE + "configuration",
    "XTTS2GPT": _PACKAGE + "gpt",
    "XTTS2Model": _PACKAGE + "modeling",
    "XTTS2ModelArgs": _PACKAGE + "configuration",
    "XTTS2Tokenizer": _PACKAGE + "tokenizer",
    "convert_trusted_legacy_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "inspect_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "load_xtts2_checkpoint": _PACKAGE + "checkpoint",
    "save_xtts2_checkpoint": _PACKAGE + "checkpoint",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
