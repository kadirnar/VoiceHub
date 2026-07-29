"""VoiceHub-native VoxCPM2 inference and fine-tuning."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.voxcpm_native."
_EXPORTS = {
    "VoxCPMConfig": _PACKAGE + "configuration_voxcpm",
    "VoxCPMForTextToSpeech": _PACKAGE + "modeling_voxcpm",
    "VoxCPMTTS": _PACKAGE + "modeling_voxcpm",
    "VoxCPMTrainingAdapter": _PACKAGE + "training_voxcpm",
}

__all__ = list(_EXPORTS)


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
