"""VoiceHub-native VoxCPM2 architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.architectures.voxcpm2.codec import VoxCPMAudioVAE
    from voicehub.architectures.voxcpm2.configuration import VoxCPM2ArchitectureConfig
    from voicehub.architectures.voxcpm2.modeling import VoxCPM2Model
    from voicehub.architectures.voxcpm2.runtime import VoxCPM2Runtime

_PACKAGE = "voicehub.architectures.voxcpm2."
_PUBLIC_IMPORTS = {
    "VoxCPM2ArchitectureConfig": (
        _PACKAGE + "configuration",
        "VoxCPM2ArchitectureConfig",
    ),
    "VoxCPM2Model": (_PACKAGE + "modeling", "VoxCPM2Model"),
    "VoxCPMAudioVAE": (_PACKAGE + "codec", "VoxCPMAudioVAE"),
    "VoxCPM2Runtime": (_PACKAGE + "runtime", "VoxCPM2Runtime"),
    "load_voxcpm2_runtime": (
        _PACKAGE + "runtime",
        "load_voxcpm2_runtime",
    ),
    "VoxCPMLoRAConfig": (_PACKAGE + "lora", "VoxCPMLoRAConfig"),
    "inject_voxcpm_lora": (_PACKAGE + "lora", "inject_voxcpm_lora"),
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
