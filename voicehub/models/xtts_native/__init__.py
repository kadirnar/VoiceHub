"""Lazy public exports for VoiceHub-native XTTS v2."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.xtts_native."
_EXPORTS = {
    "XTTS": _PACKAGE + "modeling_xtts",
    "XTTSConfig": _PACKAGE + "configuration_xtts",
    "XTTSForTextToSpeech": _PACKAGE + "modeling_xtts",
    "XTTSTrainingAdapter": _PACKAGE + "training_xtts",
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
