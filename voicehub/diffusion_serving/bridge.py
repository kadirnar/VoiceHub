"""Bridges verified diffusion TTS serving to the existing Omni speech
client."""

from __future__ import annotations

from dataclasses import replace

from voicehub.diffusion_serving.capabilities import (
    DiffusionServingBackend,
    DiffusionServingCompatibilityError,
    DiffusionTTSServingPlan,
    resolve_diffusion_tts_backend,
)
from voicehub.llm_serving import LLMBackend, LLMBackendConfig, LLMBackendTransport


def bridge_vllm_omni_tts_config(
    model_type: str,
    config: LLMBackendConfig,
) -> tuple[DiffusionTTSServingPlan, LLMBackendConfig]:
    """Validate and reuse an existing vLLM-Omni speech configuration.

    No diffusion-specific HTTP client is created.  The returned config
    is suitable for VoiceHub's existing ``LLMServingClient`` and
    ``/v1/audio/speech`` transport.
    """
    if not isinstance(config, LLMBackendConfig):
        raise TypeError("`config` must be an LLMBackendConfig.")
    plan = resolve_diffusion_tts_backend(
        model_type,
        DiffusionServingBackend.VLLM_OMNI,
    )
    if config.backend is not LLMBackend.VLLM:
        raise DiffusionServingCompatibilityError(
            "A vLLM-Omni diffusion TTS plan requires "
            "LLMBackendConfig(backend='vllm').")
    if config.transport not in {
            LLMBackendTransport.AUTO,
            LLMBackendTransport.SPEECH,
    }:
        raise DiffusionServingCompatibilityError(
            "Complete vLLM-Omni TTS uses the speech transport, not token "
            "generation.")
    return plan, replace(config, transport=LLMBackendTransport.SPEECH)


__all__ = ["bridge_vllm_omni_tts_config"]
