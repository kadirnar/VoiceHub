"""Lazy architecture declaration for VoiceHub-native CosyVoice."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.cosyvoice_native.metadata import (
    COSYVOICE3_MODEL_ID,
    COSYVOICE3_MODEL_REVISION,
    COSYVOICE_SOURCE_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_COSYVOICE_ALIASES = (
    "cosyvoice",
    "cosyvoice3",
    "native-cosyvoice",
)


def create_cosyvoice_architecture_spec() -> ArchitectureSpec:
    """Describe the native graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="cosyvoice-native",
        version="3",
        model_builder=("voicehub.architectures.cosyvoice_native.modeling:"
                       "CosyVoiceNativeModel"),
        config=("voicehub.architectures.cosyvoice_native.configuration:"
                "CosyVoiceArchitectureConfig"),
        processor=("voicehub.architectures.cosyvoice_native.tokenization:"
                   "CosyVoiceTextTokenizer"),
        decoder=("voicehub.architectures.cosyvoice_native.vocoder:"
                 "CosyVoiceHiFTGenerator"),
        objective=("voicehub.architectures.cosyvoice_native.modeling:"
                   "CosyVoiceNativeModel.forward"),
        checkpoint_adapter=(
            "voicehub.architectures.cosyvoice_native.checkpoint:"
            "load_cosyvoice_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.architectures.cosyvoice_native.artifacts:"
             "resolve_cosyvoice_artifacts"),
            "flow-matcher": ("voicehub.architectures.cosyvoice_native.flow:"
                             "CosyVoiceFlowMatchingModel"),
            "language-model":
            ("voicehub.architectures.cosyvoice_native.language_model:"
             "CosyVoiceLanguageModel"),
            "legacy-converter": (
                "voicehub.architectures.cosyvoice_native.checkpoint:"
                "convert_audited_cosyvoice_legacy_checkpoint"),
            "runtime": ("voicehub.architectures.cosyvoice_native.runtime:"
                        "load_cosyvoice_runtime"),
            "trainer-adapter":
            ("voicehub.models.cosyvoice_native.training_cosyvoice:"
             "CosyVoiceTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", ),
            features=(
                "audited-legacy-conversion",
                "causal-hift",
                "family-extensible-component-boundaries",
                "flow-matching-finetuning",
                "full-llm-finetuning",
                "hifigan-adversarial-finetuning",
                "multilingual",
                "native-qwen2",
                "strict-safetensors",
                "voice-cloning-with-precomputed-prompt",
            ),
        ),
        upstream_revision=COSYVOICE_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "reference_checkpoint":
            COSYVOICE3_MODEL_ID,
            "reference_checkpoint_revision":
            COSYVOICE3_MODEL_REVISION,
            "checkpoint_license":
            "Apache-2.0",
            "executable_checkpoint_compatibility":
            "cosyvoice3-only",
            "training_boundary": (
                "LM, conditional flow matcher, and HiFT generator/"
                "discriminator are trainable with their author objectives. "
                "Text and speech-token frontends remain frozen."),
            "parity_boundary": (
                "Official tensor inventories are exact. Numerical waveform "
                "parity is not claimed without checkpoint-level evidence."),
        },
    )


def register_cosyvoice_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_COSYVOICE_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_cosyvoice_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_COSYVOICE_ALIASES",
    "create_cosyvoice_architecture_spec",
    "register_cosyvoice_architecture",
]
