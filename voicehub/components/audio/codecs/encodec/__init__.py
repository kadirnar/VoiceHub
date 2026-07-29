"""VoiceHub-native Encodec with lazy public imports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PACKAGE = "voicehub.components.audio.codecs.encodec."
_EXPORTS = {
    "CONV_NORMALIZATIONS": _PACKAGE + "layers",
    "ConvLayerNorm": _PACKAGE + "layers",
    "EncodedFrame": _PACKAGE + "model",
    "EncodecConfig": _PACKAGE + "configuration",
    "EncodecModel": _PACKAGE + "model",
    "EncodecRelease": _PACKAGE + "metadata",
    "EncodecTrainingOutput": _PACKAGE + "model",
    "EuclideanCodebook": _PACKAGE + "quantization",
    "NormConv1d": _PACKAGE + "layers",
    "NormConvTranspose1d": _PACKAGE + "layers",
    "QuantizedResult": _PACKAGE + "quantization",
    "ResidualVectorQuantization": _PACKAGE + "quantization",
    "ResidualVectorQuantizer": _PACKAGE + "quantization",
    "SConv1d": _PACKAGE + "layers",
    "SConvTranspose1d": _PACKAGE + "layers",
    "SEANetDecoder": _PACKAGE + "layers",
    "SEANetEncoder": _PACKAGE + "layers",
    "SEANetResnetBlock": _PACKAGE + "layers",
    "SLSTM": _PACKAGE + "layers",
    "VectorQuantization": _PACKAGE + "quantization",
    "convert_official_encodec_checkpoint": _PACKAGE + "checkpoint",
    "encodec_24khz_config": _PACKAGE + "configuration",
    "encodec_48khz_config": _PACKAGE + "configuration",
    "encodec_release": _PACKAGE + "metadata",
    "file_sha256": _PACKAGE + "artifacts",
    "linear_overlap_add": _PACKAGE + "model",
    "load_encodec_model": _PACKAGE + "checkpoint",
    "load_encodec_model_from_safetensors": _PACKAGE + "checkpoint",
    "load_encodec_safetensors": _PACKAGE + "checkpoint",
    "load_pretrained_weights": _PACKAGE + "checkpoint",
    "normalize_encodec_model_name": _PACKAGE + "metadata",
    "official_tensor_shapes": _PACKAGE + "checkpoint",
    "resolve_encodec_checkpoint": _PACKAGE + "artifacts",
    "save_encodec_safetensors": _PACKAGE + "checkpoint",
    "tensor_inventory_fingerprint": _PACKAGE + "checkpoint",
    "verify_native_graph_contract": _PACKAGE + "checkpoint",
    "verify_official_checkpoint": _PACKAGE + "artifacts",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_EXPORTS))
