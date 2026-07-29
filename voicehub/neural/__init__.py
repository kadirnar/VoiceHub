"""VoiceHub-owned neural building blocks above the PyTorch substrate."""

from voicehub.neural.attention import AttentionOutput, MultiHeadAttention
from voicehub.neural.cache import CacheEntry, DynamicKVCache
from voicehub.neural.normalization import Float32LayerNorm, RMSNorm
from voicehub.neural.onnx import (
    SUPPORTED_ONNX_OPERATORS,
    NativeONNXError,
    NativeONNXGraph,
    ONNXExecutionError,
    UnsupportedONNXGraphError,
)
from voicehub.neural.rotary import RotaryEmbedding, apply_rotary_embedding, rotate_half
from voicehub.neural.transformer import FeedForward, TransformerLayer, TransformerLayerConfig, TransformerStack

__all__ = [
    "AttentionOutput",
    "CacheEntry",
    "DynamicKVCache",
    "FeedForward",
    "Float32LayerNorm",
    "MultiHeadAttention",
    "NativeONNXError",
    "NativeONNXGraph",
    "ONNXExecutionError",
    "RMSNorm",
    "RotaryEmbedding",
    "SUPPORTED_ONNX_OPERATORS",
    "TransformerLayer",
    "TransformerLayerConfig",
    "TransformerStack",
    "UnsupportedONNXGraphError",
    "apply_rotary_embedding",
    "rotate_half",
]
