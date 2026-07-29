"""Lazy public imports for VoiceHub's native Sesame CSM architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voicehub.architectures.csm.artifacts import CSMArtifacts, resolve_csm_artifacts
    from voicehub.architectures.csm.checkpoint import (
        CSMCheckpointReport,
        export_csm_checkpoint,
        inspect_csm_checkpoint,
        load_csm_checkpoint,
        validate_csm_checkpoint,
    )
    from voicehub.architectures.csm.configuration import CSMArchitectureConfig, CSMTransformerConfig
    from voicehub.architectures.csm.modeling import CSMModel, CSMOutput
    from voicehub.architectures.csm.processing import CSMCodeSegment, CSMProcessor, CSMTextTokenizer
    from voicehub.architectures.csm.registration import (
        DEFAULT_CSM_ALIASES,
        create_csm_architecture_spec,
        register_csm_architecture,
    )
    from voicehub.architectures.csm.runtime import CSMCodec, CSMRuntime, load_csm_runtime

_PUBLIC_IMPORTS = {
    "CSMArchitectureConfig": (
        "voicehub.architectures.csm.configuration",
        "CSMArchitectureConfig",
    ),
    "CSMArtifacts": (
        "voicehub.architectures.csm.artifacts",
        "CSMArtifacts",
    ),
    "CSMCheckpointReport": (
        "voicehub.architectures.csm.checkpoint",
        "CSMCheckpointReport",
    ),
    "CSMCodeSegment": (
        "voicehub.architectures.csm.processing",
        "CSMCodeSegment",
    ),
    "CSMCodec": (
        "voicehub.architectures.csm.runtime",
        "CSMCodec",
    ),
    "CSMModel": (
        "voicehub.architectures.csm.modeling",
        "CSMModel",
    ),
    "CSMOutput": (
        "voicehub.architectures.csm.modeling",
        "CSMOutput",
    ),
    "CSMProcessor": (
        "voicehub.architectures.csm.processing",
        "CSMProcessor",
    ),
    "CSMRuntime": (
        "voicehub.architectures.csm.runtime",
        "CSMRuntime",
    ),
    "CSMTextTokenizer": (
        "voicehub.architectures.csm.processing",
        "CSMTextTokenizer",
    ),
    "CSMTransformerConfig": (
        "voicehub.architectures.csm.configuration",
        "CSMTransformerConfig",
    ),
    "DEFAULT_CSM_ALIASES": (
        "voicehub.architectures.csm.registration",
        "DEFAULT_CSM_ALIASES",
    ),
    "create_csm_architecture_spec": (
        "voicehub.architectures.csm.registration",
        "create_csm_architecture_spec",
    ),
    "export_csm_checkpoint": (
        "voicehub.architectures.csm.checkpoint",
        "export_csm_checkpoint",
    ),
    "inspect_csm_checkpoint": (
        "voicehub.architectures.csm.checkpoint",
        "inspect_csm_checkpoint",
    ),
    "load_csm_checkpoint": (
        "voicehub.architectures.csm.checkpoint",
        "load_csm_checkpoint",
    ),
    "load_csm_runtime": (
        "voicehub.architectures.csm.runtime",
        "load_csm_runtime",
    ),
    "register_csm_architecture": (
        "voicehub.architectures.csm.registration",
        "register_csm_architecture",
    ),
    "resolve_csm_artifacts": (
        "voicehub.architectures.csm.artifacts",
        "resolve_csm_artifacts",
    ),
    "validate_csm_checkpoint": (
        "voicehub.architectures.csm.checkpoint",
        "validate_csm_checkpoint",
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
