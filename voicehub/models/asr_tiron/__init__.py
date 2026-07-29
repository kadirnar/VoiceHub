"""Native Tiron speaker-attributed automatic speech recognition."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from voicehub.models.asr_tiron.configuration_asr_tiron import TironASRConfig
    from voicehub.models.asr_tiron.constraints import TironConstraintLogitsProcessor
    from voicehub.models.asr_tiron.modeling_asr_tiron import TironForSpeechRecognition

_PUBLIC_COMPONENTS = {
    "TironASRConfig": ("voicehub.models.asr_tiron.configuration_asr_tiron"),
    "TironConstraintLogitsProcessor": ("voicehub.models.asr_tiron.constraints"),
    "TironForSpeechRecognition": ("voicehub.models.asr_tiron.modeling_asr_tiron"),
}


def __getattr__(name: str) -> Any:
    """Resolve public components without eagerly importing PyTorch."""
    try:
        module_name = _PUBLIC_COMPONENTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "TironASRConfig",
    "TironConstraintLogitsProcessor",
    "TironForSpeechRecognition",
]
