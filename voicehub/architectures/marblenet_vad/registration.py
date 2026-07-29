"""Lazy native architecture declaration for MarbleNet Frame-VAD."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.marblenet_vad.metadata import NEMO_SOURCE_REVISION
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_MARBLENET_VAD_ALIASES = (
    "nemo-marblenet",
    "nemo-vad",
    "native-marblenet-vad",
)


def create_marblenet_vad_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="marblenet-vad",
        version="1",
        model_builder=("voicehub.architectures.marblenet_vad.modeling:MarbleNetVADModel"),
        config=("voicehub.architectures.marblenet_vad.configuration:"
                "MarbleNetVADConfig"),
        objective=("voicehub.architectures.marblenet_vad.objective:"
                   "marblenet_vad_loss"),
        checkpoint_adapter=(
            "voicehub.architectures.marblenet_vad.checkpoint:"
            "MarbleNetVADSafeTensorsCheckpointAdapter"),
        components={
            "frontend": ("voicehub.architectures.marblenet_vad.frontend:"
                         "MarbleNetFilterbankFeatures"),
            "pickle-converter":
            ("voicehub.architectures.marblenet_vad.checkpoint:"
             "convert_nemo_marblenet_checkpoint"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.VOICE_ACTIVITY_DETECTION, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=(
                "safetensors",
                "trusted-pickle-conversion",
            ),
            training=True,
            streaming=False,
            batched_inference=True,
            features=(
                "log-mel",
                "depthwise-separable-convolution",
                "frame-scores",
                "frame-cross-entropy",
                "raw-audio-fine-tuning",
            ),
        ),
        upstream_revision=NEMO_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "marblenet-vad",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "training_boundary": (
                "VoiceHub reproduces the published graph, frame cross-entropy, "
                "optimizer schedule, and documented augmentations. The "
                "original training corpora are not redistributed."),
        },
    )


def register_marblenet_vad_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_MARBLENET_VAD_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_marblenet_vad_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_MARBLENET_VAD_ALIASES",
    "create_marblenet_vad_architecture_spec",
    "register_marblenet_vad_architecture",
]
