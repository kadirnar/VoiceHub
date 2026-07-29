"""Torch-free logical operation identifiers for optional kernel providers."""

LLM_GATED_SILU = "tts.llm.gated_silu"
VITS_TANH_SIGMOID_GATE = "tts.vits.tanh_sigmoid_gate"
VITS_FUSED_ADD_TANH_SIGMOID = "tts.vits.fused_add_tanh_sigmoid"
DIFFUSION_FUSED_BIAS_GELU = "tts.diffusion.fused_bias_gelu"

__all__ = [
    "DIFFUSION_FUSED_BIAS_GELU",
    "LLM_GATED_SILU",
    "VITS_FUSED_ADD_TANH_SIGMOID",
    "VITS_TANH_SIGMOID_GATE",
]
