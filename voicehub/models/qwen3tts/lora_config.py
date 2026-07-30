"""Dependency-light LoRA topology declarations for native Qwen3-TTS."""

from __future__ import annotations

from collections.abc import Iterable

QWEN3_TTS_ATTENTION_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
QWEN3_TTS_MLP_LORA_TARGETS = (
    "gate_proj",
    "up_proj",
    "down_proj",
)
QWEN3_TTS_LORA_TARGETS = (
    *QWEN3_TTS_ATTENTION_LORA_TARGETS,
    *QWEN3_TTS_MLP_LORA_TARGETS,
)


def normalize_qwen3_tts_lora_targets(value: Iterable[str], ) -> tuple[str, ...]:
    """Validate public target names before expanding them to exact paths."""
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "Qwen3-TTS `training_lora_target_modules` must be a sequence "
            "of projection names, not one string.")
    try:
        targets = tuple(value)
    except TypeError as error:
        raise TypeError(
            "Qwen3-TTS `training_lora_target_modules` must be an iterable "
            "of projection names.") from error
    if not targets:
        raise ValueError("Qwen3-TTS `training_lora_target_modules` must not be empty.")
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise ValueError("Qwen3-TTS LoRA target names must be non-empty strings.")
    normalized = tuple(target.strip() for target in targets)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Qwen3-TTS LoRA target names must not contain duplicates.")
    unknown = tuple(target for target in normalized if target not in QWEN3_TTS_LORA_TARGETS)
    if unknown:
        supported = ", ".join(QWEN3_TTS_LORA_TARGETS)
        raise ValueError(
            "Unsupported Qwen3-TTS LoRA target(s): "
            f"{', '.join(unknown)}. Supported projections: {supported}.")
    return normalized


__all__ = [
    "QWEN3_TTS_ATTENTION_LORA_TARGETS",
    "QWEN3_TTS_LORA_TARGETS",
    "QWEN3_TTS_MLP_LORA_TARGETS",
    "normalize_qwen3_tts_lora_targets",
]
