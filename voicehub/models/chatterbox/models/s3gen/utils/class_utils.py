import torch

from voicehub.models.chatterbox.models.s3gen.transformer.activation import Swish
from voicehub.models.chatterbox.models.s3gen.transformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from voicehub.models.chatterbox.models.s3gen.transformer.embedding import (
    EspnetRelPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    WhisperPositionalEncoding,
)
from voicehub.models.chatterbox.models.s3gen.transformer.subsampling import (
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    EmbedinigNoSubsampling,
    LegacyLinearNoSubsampling,
    LinearNoSubsampling,
)

COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    'paraformer_dummy': torch.nn.Identity
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
