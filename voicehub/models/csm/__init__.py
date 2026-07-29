"""VoiceHub-native Sesame CSM model family."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.architectures.csm import CSMArchitectureConfig, CSMModel
    from voicehub.models.csm.inference import CSMTTS, CSMConfig, CSMForTextToSpeech

_PUBLIC_IMPORTS = {
    "CSMArchitectureConfig": (
        "voicehub.architectures.csm.configuration",
        "CSMArchitectureConfig",
    ),
    "CSMModel": (
        "voicehub.architectures.csm.modeling",
        "CSMModel",
    ),
    "CSMConfig": (
        "voicehub.models.csm.inference",
        "CSMConfig",
    ),
    "CSMForTextToSpeech": (
        "voicehub.models.csm.inference",
        "CSMForTextToSpeech",
    ),
    "CSMTTS": (
        "voicehub.models.csm.inference",
        "CSMTTS",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _PUBLIC_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_PUBLIC_IMPORTS)
