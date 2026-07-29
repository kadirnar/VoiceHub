"""Lazy declaration for VoiceHub's native XTTS v2 architecture."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.architectures.xtts2.metadata import (
    XTTS2_CHECKPOINT_LICENSE,
    XTTS2_CHECKPOINT_REPOSITORY,
    XTTS2_CHECKPOINT_REVISION,
    XTTS2_CONFIG_SHA256,
    XTTS2_NATIVE_PARAMETER_COUNT,
    XTTS2_NATIVE_TENSOR_COUNT,
    XTTS2_SOURCE_LICENSE,
    XTTS2_SOURCE_REPOSITORY,
    XTTS2_SOURCE_REVISION,
    XTTS2_VOCAB_SHA256,
)
from voicehub.tasks import SpeechTask

DEFAULT_XTTS2_ALIASES = (
    "native-xtts",
    "xtts",
    "xtts-v2",
)


def create_xtts2_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="xtts2",
        version="2",
        model_builder="voicehub.architectures.xtts2.modeling:XTTS2Model",
        config="voicehub.architectures.xtts2.configuration:XTTS2Config",
        processor="voicehub.architectures.xtts2.tokenizer:XTTS2Tokenizer",
        decoder="voicehub.architectures.xtts2.decoder:HifiDecoder",
        objective="voicehub.architectures.xtts2.gpt:XTTS2GPT.forward",
        checkpoint_adapter=("voicehub.architectures.xtts2.checkpoint:"
                            "load_xtts2_checkpoint"),
        components={
            "checkpoint-exporter": ("voicehub.architectures.xtts2.checkpoint:"
                                    "save_xtts2_checkpoint"),
            "conditioning-encoder": ("voicehub.architectures.xtts2.conditioning:"
                                     "ConditioningEncoder"),
            "legacy-converter":
            ("voicehub.architectures.xtts2.checkpoint:"
             "convert_trusted_legacy_xtts2_checkpoint"),
            "speaker-encoder": ("voicehub.architectures.xtts2.decoder:"
                                "ResNetSpeakerEncoder"),
            "trainer-adapter": ("voicehub.models.xtts_native.training_xtts:"
                                "XTTSTrainingAdapter"),
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
            optimization_passes=("compile", "sdpa"),
            features=(
                "17-languages",
                "24-khz-waveform",
                "autoregressive-acoustic-tokens",
                "gpt-finetuning",
                "hifigan-decoder",
                "perceiver-conditioning",
                "precomputed-audio-code-training",
                "reference-voice-cloning",
                "resnet-speaker-encoder",
                "strict-safetensors-runtime",
            ),
        ),
        upstream_revision=XTTS2_SOURCE_REVISION,
        license_id=XTTS2_SOURCE_LICENSE,
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            XTTS2_SOURCE_REPOSITORY,
            "source_revision":
            XTTS2_SOURCE_REVISION,
            "reference_checkpoint":
            XTTS2_CHECKPOINT_REPOSITORY,
            "reference_checkpoint_revision":
            XTTS2_CHECKPOINT_REVISION,
            "reference_config_sha256":
            XTTS2_CONFIG_SHA256,
            "reference_vocab_sha256":
            XTTS2_VOCAB_SHA256,
            "reference_native_tensor_count":
            XTTS2_NATIVE_TENSOR_COUNT,
            "reference_native_parameter_count":
            XTTS2_NATIVE_PARAMETER_COUNT,
            "checkpoint_license":
            XTTS2_CHECKPOINT_LICENSE,
            "steady_state_checkpoint_format":
            "safetensors",
            "training_boundary": (
                "The complete autoregressive GPT is trainable with the "
                "source text and acoustic-token cross-entropies. The DVAE "
                "is an offline, frozen data-preparation boundary; the "
                "speaker encoder and HiFi-GAN decoder remain frozen."),
            "legacy_boundary": (
                "Coqui publishes model.pth. Native runtime loading never "
                "deserializes it; an explicit weights-only one-time "
                "conversion produces the required model.safetensors."),
        },
    )


def register_xtts2_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_XTTS2_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_xtts2_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_XTTS2_ALIASES",
    "create_xtts2_architecture_spec",
    "register_xtts2_architecture",
]
