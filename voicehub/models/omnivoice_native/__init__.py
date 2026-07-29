"""VoiceHub-native OmniVoice inference and fine-tuning."""

from __future__ import annotations

import importlib
from typing import Any

_PACKAGE = "voicehub.models.omnivoice_native."
_EXPORTS = {
    "OmniVoiceConfig": _PACKAGE + "configuration_omnivoice",
    "OmniVoiceForTextToSpeech": _PACKAGE + "modeling_omnivoice",
    "OmniVoiceTTS": _PACKAGE + "modeling_omnivoice",
    "OmniVoiceTrainingAdapter": _PACKAGE + "training_omnivoice",
    "OmniVoiceTrainingCollator": _PACKAGE + "training_omnivoice",
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
