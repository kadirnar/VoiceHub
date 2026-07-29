"""Lazy declaration for the VoiceHub-native Parler-TTS architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.parlertts.metadata import (
    PARLER_TTS_CHECKPOINT,
    PARLER_TTS_CHECKPOINT_LICENSE,
    PARLER_TTS_CHECKPOINT_REVISION,
    PARLER_TTS_HEADER_FINGERPRINT,
    PARLER_TTS_PARAMETER_COUNT,
    PARLER_TTS_SOURCE,
    PARLER_TTS_SOURCE_LICENSE,
    PARLER_TTS_SOURCE_REVISION,
    PARLER_TTS_TENSOR_COUNT,
    TRANSFORMERS_T5_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_PARLER_TTS_ALIASES = (
    "parler-tts",
    "native-parlertts",
    "parler-tts-mini-v1",
)


def create_parlertts_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="parlertts",
        version="1",
        model_builder=("voicehub.architectures.parlertts.modeling:"
                       "ParlerTTSForConditionalGeneration"),
        config=("voicehub.architectures.parlertts.configuration:"
                "ParlerTTSArchitectureConfig"),
        processor=("voicehub.architectures.parlertts.processing:"
                   "ParlerTextTokenizer"),
        decoder=("voicehub.architectures.parlertts.modeling:"
                 "ParlerDacAudioEncoder"),
        objective=("voicehub.architectures.parlertts.modeling:"
                   "ParlerTTSForCausalLM"),
        checkpoint_adapter=("voicehub.architectures.parlertts.checkpoint:"
                            "load_parlertts_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.architectures.parlertts.artifacts:"
             "resolve_parlertts_artifacts"),
            "checkpoint-exporter":
            ("voicehub.architectures.parlertts.checkpoint:"
             "export_parlertts_checkpoint"),
            "t5-encoder": ("voicehub.architectures.parlertts.t5:"
                           "NativeT5EncoderModel"),
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
            optimization_passes=("compile", "sdpa"),
            features=(
                "description-conditioned-tts",
                "flan-t5-text-encoder",
                "delayed-parallel-codebooks",
                "native-dac",
                "full-decoder-fine-tuning",
                "frozen-encoder-fine-tuning",
                "per-codebook-cross-entropy",
            ),
        ),
        upstream_revision=PARLER_TTS_SOURCE_REVISION,
        license_id=PARLER_TTS_SOURCE_LICENSE,
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            PARLER_TTS_SOURCE,
            "reference_checkpoint":
            PARLER_TTS_CHECKPOINT,
            "reference_checkpoint_revision":
            PARLER_TTS_CHECKPOINT_REVISION,
            "checkpoint_license":
            PARLER_TTS_CHECKPOINT_LICENSE,
            "reference_tensor_count":
            PARLER_TTS_TENSOR_COUNT,
            "reference_parameter_count":
            PARLER_TTS_PARAMETER_COUNT,
            "reference_header_fingerprint":
            PARLER_TTS_HEADER_FINGERPRINT,
            "t5_reference_revision":
            TRANSFORMERS_T5_REVISION,
            "training_boundary": (
                "Exact upstream delayed-codebook cross-entropy. The DAC is "
                "frozen; the T5 encoder is trainable by default and can be "
                "explicitly frozen for lower-memory adaptation."),
            "upstream_trainable_scope": (
                "text_encoder",
                "decoder",
                "embed_prompts",
            ),
            "always_frozen_components": ("audio_encoder", ),
            "full_finetuning_ready":
            True,
        },
    )


def register_parlertts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_PARLER_TTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_parlertts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_PARLER_TTS_ALIASES",
    "create_parlertts_architecture_spec",
    "register_parlertts_architecture",
]
