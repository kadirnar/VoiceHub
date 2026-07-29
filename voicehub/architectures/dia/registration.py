"""Lazy architecture declaration for VoiceHub-native Dia."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.dia.metadata import (
    NARI_DIA_CHECKPOINT_REVISION,
    NARI_DIA_HEADER_FINGERPRINT,
    NARI_DIA_PARAMETER_COUNT,
    NARI_DIA_SOURCE_REVISION,
    NARI_DIA_TENSOR_COUNT,
    TRANSFORMERS_DIA_SOURCE_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_DIA_ALIASES = (
    "native-dia",
    "dia-tts",
    "dia-1.6b",
)


def create_dia_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="dia",
        version="1",
        model_builder=("voicehub.architectures.dia.modeling:"
                       "DiaForConditionalGeneration"),
        config=("voicehub.architectures.dia.configuration:"
                "DiaArchitectureConfig"),
        processor="voicehub.architectures.dia.processing:DiaProcessor",
        objective="voicehub.objectives.sequence:sequence_cross_entropy",
        checkpoint_adapter=("voicehub.architectures.dia.checkpoint:"
                            "HuggingFaceDiaCheckpointAdapter"),
        components={
            "audio-codec": ("voicehub.architectures.dac.modeling:DacModel"),
            "byte-tokenizer": ("voicehub.architectures.dia.processing:DiaByteTokenizer"),
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
            optimization_passes=("compile", ),
            features=(
                "byte-text-encoder",
                "multi-speaker-dialogue",
                "nine-codebook-delay-pattern",
                "classifier-free-guidance",
                "native-dac",
                "teacher-forced-cross-entropy",
                "audio-prompt-conditioning",
                "checkpoint-conversion",
                "no-remote-code",
            ),
        ),
        upstream_revision=NARI_DIA_SOURCE_REVISION,
        license_id="Apache-2.0",
        metadata={
            "family":
            "dia",
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "main_library_source": ("https://github.com/nari-labs/dia/tree/"
                                    f"{NARI_DIA_SOURCE_REVISION}"),
            "transformers_reference_revision": (TRANSFORMERS_DIA_SOURCE_REVISION),
            "transformers_source": (
                "https://github.com/huggingface/transformers/tree/"
                f"{TRANSFORMERS_DIA_SOURCE_REVISION}/src/transformers/models/"
                "dia"),
            "reference_checkpoint":
            "nari-labs/Dia-1.6B-0626",
            "reference_checkpoint_revision":
            NARI_DIA_CHECKPOINT_REVISION,
            "reference_parameter_count":
            NARI_DIA_PARAMETER_COUNT,
            "reference_tensor_count":
            NARI_DIA_TENSOR_COUNT,
            "full_finetuning_ready":
            True,
            "generation_cache":
            "prefix-recompute-baseline",
            "reference_safetensors_header_fingerprint": (NARI_DIA_HEADER_FINGERPRINT),
        },
    )


def register_dia_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_DIA_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_dia_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_DIA_ALIASES",
    "create_dia_architecture_spec",
    "register_dia_architecture",
]
