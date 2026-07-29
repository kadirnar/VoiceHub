"""Lazy architecture declaration for VoiceHub-native MedASR."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.medasr.metadata import (
    MEDASR_CHECKPOINT,
    MEDASR_MODEL_ID,
    MEDASR_MODEL_REVISION,
    MEDASR_SOURCE_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_MEDASR_ALIASES = (
    "google-medasr",
    "medasr",
)


def create_medasr_architecture_spec() -> ArchitectureSpec:
    """Describe the audited LASR CTC graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="lasr-ctc",
        version="1",
        model_builder="voicehub.architectures.medasr.modeling:MedASRForCTC",
        config="voicehub.architectures.medasr.configuration:MedASRConfig",
        processor=("voicehub.architectures.medasr.processing:MedASRProcessor"),
        decoder=("voicehub.architectures.medasr.tokenization:"
                 "MedASRTokenizer.decode_ctc"),
        objective="voicehub.objectives.ctc:CTCLoss",
        checkpoint_adapter=("voicehub.architectures.medasr.checkpoint:"
                            "MedASRCheckpointAdapter"),
        components={
            "audio-frontend": ("voicehub.architectures.medasr.frontend:"
                               "MedASRFeatureExtractor"),
            "encoder": ("voicehub.architectures.medasr.modeling:MedASREncoder"),
            "tokenizer": ("voicehub.architectures.medasr.tokenization:MedASRTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.AUTOMATIC_SPEECH_RECOGNITION, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            features=(
                "conformer",
                "connectionist-temporal-classification",
                "medical-dictation",
                "gradient-checkpointing",
                "portable-export",
            ),
        ),
        upstream_revision=MEDASR_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family": "lasr-ctc",
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "reference_checkpoint": MEDASR_MODEL_ID,
            "reference_checkpoint_revision": MEDASR_MODEL_REVISION,
            "reference_checkpoint_terms": ("Health AI Developer Foundations Terms of Use"),
            "reference_checkpoint_header_fingerprint": (MEDASR_CHECKPOINT["header_fingerprint"]),
            "access": "gated",
            "decoding": "greedy-ctc-only",
            "language": "English",
        },
    )


def register_medasr_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_MEDASR_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be ArchitectureRegistry or None.")
    spec = create_medasr_architecture_spec()
    target.register(
        spec,
        aliases=aliases,
        exist_ok=exist_ok,
    )
    return spec


__all__ = [
    "DEFAULT_MEDASR_ALIASES",
    "create_medasr_architecture_spec",
    "register_medasr_architecture",
]
