"""VoiceHub-native VITS and MMS-TTS provider with lazy public exports."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.vits."
_EXPORTS = {
    "MmsTTSForTextToSpeech": _PACKAGE + "inference",
    "NativeVitsAdversarialTrainingAdapter": _PACKAGE + "training",
    "NativeVitsGeneratorTrainingAdapter": _PACKAGE + "training",
    "VitsConfig": _PACKAGE + "configuration_vits",
    "VitsForTextToSpeech": _PACKAGE + "modeling_vits",
    "VitsReconstructionTrainingAdapter": _PACKAGE + "training",
    "VitsTTS": _PACKAGE + "inference",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve provider components only when explicitly requested."""
    try:
        module_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
