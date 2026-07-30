"""Torch-free logical operation identifiers for optional kernel providers."""

LLM_GATED_SILU = "tts.llm.gated_silu"
VITS_TANH_SIGMOID_GATE = "tts.vits.tanh_sigmoid_gate"
VITS_FUSED_ADD_TANH_SIGMOID = "tts.vits.fused_add_tanh_sigmoid"
DIFFUSION_FUSED_BIAS_GELU = "tts.diffusion.fused_bias_gelu"
DIFFUSION_FUSED_MODULATE = "tts.diffusion.fused_modulate"
AUDIO_CODEC_SNAKE = "audio.codec.snake"
AUDIO_CODEC_SNAKE_BETA = "audio.codec.snake_beta"
AUDIO_CODEC_EUCLIDEAN_VQ = "audio.codec.euclidean_vq_search"

__all__ = [
    "AUDIO_CODEC_EUCLIDEAN_VQ",
    "AUDIO_CODEC_SNAKE",
    "AUDIO_CODEC_SNAKE_BETA",
    "DIFFUSION_FUSED_BIAS_GELU",
    "DIFFUSION_FUSED_MODULATE",
    "LLM_GATED_SILU",
    "VITS_FUSED_ADD_TANH_SIGMOID",
    "VITS_TANH_SIGMOID_GATE",
]
