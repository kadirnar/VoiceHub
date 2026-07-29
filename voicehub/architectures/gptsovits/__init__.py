"""VoiceHub-native GPT-SoVITS classic-S2 architecture.

The package intentionally keeps heavyweight PyTorch modules lazy.
Importing ``voicehub.architectures.gptsovits`` is therefore safe for
registry discovery.
"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "GPTSoVITSS1Config": (
        "voicehub.architectures.gptsovits.configuration",
        "GPTSoVITSS1Config",
    ),
    "GPTSoVITSS2Config": (
        "voicehub.architectures.gptsovits.configuration",
        "GPTSoVITSS2Config",
    ),
    "GPTSoVITSSemanticModel": (
        "voicehub.architectures.gptsovits.semantic",
        "GPTSoVITSSemanticModel",
    ),
    "GPTSoVITSSynthesizer": (
        "voicehub.architectures.gptsovits.modeling",
        "GPTSoVITSSynthesizer",
    ),
    "GPTSoVITSRuntime": (
        "voicehub.architectures.gptsovits.runtime",
        "GPTSoVITSRuntime",
    ),
    "GPTSoVITSStagedTrainingModel": (
        "voicehub.architectures.gptsovits.training",
        "GPTSoVITSStagedTrainingModel",
    ),
    "build_s2_discriminator": (
        "voicehub.architectures.gptsovits.modeling",
        "build_s2_discriminator",
    ),
    "build_s2_generator": (
        "voicehub.architectures.gptsovits.modeling",
        "build_s2_generator",
    ),
    "build_staged_training_model": (
        "voicehub.architectures.gptsovits.training",
        "build_staged_training_model",
    ),
    "convert_gptsovits_legacy_checkpoints": (
        "voicehub.architectures.gptsovits.checkpoint",
        "convert_gptsovits_legacy_checkpoints",
    ),
    "export_gptsovits_checkpoint": (
        "voicehub.architectures.gptsovits.checkpoint",
        "export_gptsovits_checkpoint",
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


__all__ = sorted(_EXPORTS)
