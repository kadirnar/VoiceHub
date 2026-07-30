"""Source-integrated neural audio codecs with lazy structural contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASE_EXPORTS = frozenset({
    "AudioAutoencoderView",
    "AudioCodec",
    "AudioCodecComponentView",
    "AudioCodecProtocol",
    "CodecCodeBatch",
    "DenseCodecCodeBatch",
    "DenseCodecCodes",
    "RaggedCodecCodeBatch",
    "RaggedCodecCodes",
    "codec_is_stochastic_vae",
    "codec_target_is_stochastic",
    "coerce_codec_codes",
    "is_audio_codec",
    "separate_audio_codec",
})

_CATALOG_EXPORTS = frozenset({
    "CODEC_ALIASES",
    "CODEC_CATALOG",
    "LLM_TTS_CODEC_FEATURE",
    "REGISTERED_LLM_TTS_MODEL_TYPES",
    "CodecCatalogEntry",
    "CodecIntegration",
    "CodecOptimizationManifest",
    "CodecOptimizationSurface",
    "CodecOwnerBinding",
    "CodecPrimitive",
    "CodecPrimitiveManifest",
    "CodecRepresentation",
    "CodecStage",
    "CodecStageAvailability",
    "CodecStageManifest",
    "get_codec_catalog_entry",
    "get_codec_entries_for_model",
    "get_codec_entry",
    "get_codec_primitive_manifest",
    "list_codec_catalog_entries",
    "list_codec_entries",
    "list_codec_primitive_manifests",
    "list_registered_llm_tts_codec_model_types",
    "normalize_codec_id",
    "validate_codec_catalog_registry_coverage",
})

_EXPORT_MODULES = {
    **{
        name: "voicehub.components.audio.codecs.base"
        for name in _BASE_EXPORTS
    },
    **{
        name: "voicehub.components.audio.codecs.catalog"
        for name in _CATALOG_EXPORTS
    },
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORT_MODULES))
