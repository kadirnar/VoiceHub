"""Truthful serving capabilities for audio and visual diffusion runtimes.

This package is metadata-only at import time.  Optional vLLM-Omni and
SGLang packages are never imported unless an explicit feature probe or
plugin registration is requested.
"""

from voicehub.diffusion_serving.bridge import bridge_vllm_omni_tts_config
from voicehub.diffusion_serving.capabilities import (
    DiffusionServingBackend,
    DiffusionServingCapability,
    DiffusionServingCompatibilityError,
    DiffusionTTSServingPlan,
    get_diffusion_serving_capability,
    list_diffusion_serving_capabilities,
    resolve_diffusion_tts_backend,
)
from voicehub.diffusion_serving.vllm_omni import (
    VLLMOmniDiffusionPlugin,
    VLLMOmniFeatureStatus,
    detect_vllm_omni_features,
)

__all__ = [
    "DiffusionServingBackend",
    "DiffusionServingCapability",
    "DiffusionServingCompatibilityError",
    "DiffusionTTSServingPlan",
    "VLLMOmniDiffusionPlugin",
    "VLLMOmniFeatureStatus",
    "bridge_vllm_omni_tts_config",
    "detect_vllm_omni_features",
    "get_diffusion_serving_capability",
    "list_diffusion_serving_capabilities",
    "resolve_diffusion_tts_backend",
]
