"""Lazy declaration for VoiceHub's native F5-TTS architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.f5tts.metadata import (
    F5TTS_CHECKPOINT_LICENSE,
    F5TTS_CHECKPOINT_REPOSITORY,
    F5TTS_CHECKPOINT_REVISION,
    F5TTS_SOURCE_LICENSE,
    F5TTS_SOURCE_REVISION,
    VOCOS_CHECKPOINT_REVISION,
    VOCOS_REPOSITORY,
    VOCOS_SOURCE_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_F5TTS_ALIASES = (
    "native-f5tts",
    "f5-tts",
    "f5tts-v1-base",
)


def create_f5tts_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="f5tts",
        version="1",
        model_builder=("voicehub.architectures.f5tts.modeling:build_f5tts_model"),
        config=("voicehub.architectures.f5tts.configuration:"
                "F5TTSArchitectureConfig"),
        processor=("voicehub.architectures.f5tts.frontend:NativeF5TextFrontend"),
        decoder="voicehub.architectures.f5tts.vocoder:NativeVocos",
        objective=("voicehub.architectures.f5tts.modeling:"
                   "F5ConditionalFlowMatcher"),
        checkpoint_adapter=("voicehub.architectures.f5tts.checkpoint:"
                            "load_f5tts_checkpoint"),
        components={
            "dit": "voicehub.architectures.f5tts.modeling:F5DiT",
            "vocoder-checkpoint-adapter":
            ("voicehub.architectures.f5tts.checkpoint:"
             "load_vocos_checkpoint"),
            "legacy-importer": ("voicehub.architectures.f5tts.checkpoint:"
                                "convert_legacy_f5tts_checkpoint"),
            "artifact-resolver": ("voicehub.architectures.f5tts.artifacts:"
                                  "resolve_f5tts_artifacts"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=(
                "compile",
                "attention-backend",
                "custom-kernels",
                "diffusion-cache",
                "diffusion-sampling",
            ),
            features=(
                "diffusion-family",
                "diffusion-serving-native",
                "diffusion-kind-conditional-flow-matching",
                "diffusion-operation-denoiser",
                "diffusion-operation-classifier-free-guidance",
                "diffusion-operation-euler-solver",
                "diffusion-operation-midpoint-solver",
                "diffusion-sampling-schedule",
                "diffusion-sampling-guidance",
                "diffusion-sampling-prediction-cache",
                "diffusion-sampling-stork2",
                "voice-cloning",
                "conditional-flow-matching",
                "classifier-free-guidance",
                "flash-attention-4-optional",
                "fused-bias-gelu-kernels",
                "native-euler-ode",
                "native-midpoint-ode",
                "native-vocos",
                "checkpoint-conversion",
                "full-flow-fine-tuning",
                "pretokenized-pinyin",
                "raw-chinese-g2p-requires-explicit-normalizer",
            ),
        ),
        upstream_revision=F5TTS_SOURCE_REVISION,
        license_id=F5TTS_SOURCE_LICENSE,
        metadata={
            "diffusion_architecture_kind":
            "conditional-flow-matching",
            "diffusion_operations": (
                "denoiser",
                "classifier-free-guidance",
                "euler-solver",
                "midpoint-solver",
            ),
            "diffusion_sampling_capabilities": (
                "schedule",
                "guidance",
                "prediction-cache",
                "stork2",
            ),
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source": ("https://github.com/SWivid/F5-TTS/tree/"
                       f"{F5TTS_SOURCE_REVISION}"),
            "reference_checkpoint":
            F5TTS_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision":
            F5TTS_CHECKPOINT_REVISION,
            "checkpoint_license":
            F5TTS_CHECKPOINT_LICENSE,
            "vocoder":
            VOCOS_REPOSITORY,
            "vocoder_checkpoint_revision":
            VOCOS_CHECKPOINT_REVISION,
            "vocoder_source_revision":
            VOCOS_SOURCE_REVISION,
            "training_boundary": (
                "Full released F5-TTS v1 DiT conditional-flow objective; "
                "Vocos remains frozen during flow fine-tuning."),
            "full_finetuning_ready":
            True,
            "quality_validated_inference_dtypes": ("float32", ),
            "reduced_precision_inference_policy": (
                "Explicit opt-in is required because whole-model BF16 and "
                "mixed DiT-BF16 inference failed the end-to-end quality "
                "gate; FP16 has no retained quality validation."),
        },
    )


def register_f5tts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_F5TTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_f5tts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_F5TTS_ALIASES",
    "create_f5tts_architecture_spec",
    "register_f5tts_architecture",
]
