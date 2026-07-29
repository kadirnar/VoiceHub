"""Native MarbleNet VAD architecture with lazy public component imports."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "MarbleNetVADConfig": (
        "voicehub.architectures.marblenet_vad.configuration",
        "MarbleNetVADConfig",
    ),
    "MarbleNetVADModel": (
        "voicehub.architectures.marblenet_vad.modeling",
        "MarbleNetVADModel",
    ),
    "MarbleNetVADOutput": (
        "voicehub.architectures.marblenet_vad.modeling",
        "MarbleNetVADOutput",
    ),
    "MarbleNetVADSafeTensorsCheckpointAdapter": (
        "voicehub.architectures.marblenet_vad.checkpoint",
        "MarbleNetVADSafeTensorsCheckpointAdapter",
    ),
    "convert_nemo_marblenet_checkpoint": (
        "voicehub.architectures.marblenet_vad.checkpoint",
        "convert_nemo_marblenet_checkpoint",
    ),
    "marblenet_vad_loss": (
        "voicehub.architectures.marblenet_vad.objective",
        "marblenet_vad_loss",
    ),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))


__all__ = sorted(_EXPORTS)
