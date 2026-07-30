"""Lazy architecture declaration for VoiceHub-native NeuTTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.neutts.metadata import (
    NEUCODEC_REFERENCE,
    NEUCODEC_SOURCE_REVISION,
    NEUTTS_SOURCE_REVISION,
    NEUTTS_TRAINING_SOURCE,
    NEUTTS_VARIANTS,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_NEUTTS_ALIASES = (
    "native-neutts",
    "neuphonic-neutts",
    "neutts-air",
    "neutts-nano",
    "neutts-2e",
)


def create_neutts_architecture_spec() -> ArchitectureSpec:
    """Describe the native LM, tokenizer, NeuCodec, and Air objective."""
    return ArchitectureSpec(
        architecture_id="neutts",
        version="1",
        model_builder=("voicehub.architectures.neutts.modeling:NeuTTSBackbone"),
        config=("voicehub.architectures.neutts.configuration:"
                "NeuTTSBackboneConfig"),
        processor=("voicehub.architectures.neutts.tokenization:NeuTTSTokenizer"),
        decoder="voicehub.architectures.neutts.neucodec:NeuCodecModel",
        objective=("voicehub.models.neutts.training:NeuTTSTrainingAdapter"),
        checkpoint_adapter=("voicehub.architectures.neutts.checkpoint:"
                            "NeuTTSCheckpointAdapter"),
        components={
            "artifact-resolver": ("voicehub.architectures.neutts.artifacts:"
                                  "resolve_neutts_artifacts"),
            "audio-codec": ("voicehub.architectures.neutts.neucodec:NeuCodecModel"),
            "codec-artifact-resolver":
            ("voicehub.architectures.neutts.artifacts:"
             "resolve_neucodec_artifacts"),
            "codec-checkpoint-adapter":
            ("voicehub.architectures.neutts.checkpoint:"
             "NeuCodecCheckpointAdapter"),
            "runtime": ("voicehub.architectures.neutts.modeling:NeuTTSRuntime"),
            "sft-dataset": ("voicehub.models.neutts.training:NeuTTSSFTDataset"),
            "wrapper": ("voicehub.models.neutts.inference:"
                        "NeuTTSForTextToSpeech"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", ),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", "sdpa"),
            features=(
                "llm-tts-codec",
                "completion-only-codec-language-modeling",
                "emotion-control",
                "frozen-neucodec",
                "llama-linear-rope",
                "multilingual",
                "native-byte-bpe-tokenizer",
                "native-neucodec",
                "phoneme-injection",
                "preencoded-code-fine-tuning",
                "qwen2-qwen3-llama-backbones",
                "raw-audio-fine-tuning",
                "strict-safetensors-reload",
                "voice-cloning",
            ),
        ),
        upstream_revision=NEUTTS_SOURCE_REVISION,
        license_id="NeuTTS-Open-License-1.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            "neuphonic/neutts",
            "source_revision":
            NEUTTS_SOURCE_REVISION,
            "reference_checkpoints": {
                model_id: {
                    "revision": values["revision"],
                    "license": values["license"],
                    "tensor_count": values["tensor_count"],
                }
                for model_id, values in NEUTTS_VARIANTS.items()
            },
            "codec_checkpoint":
            NEUCODEC_REFERENCE["model_id"],
            "codec_checkpoint_revision":
            NEUCODEC_REFERENCE["revision"],
            "codec_checkpoint_sha256":
            NEUCODEC_REFERENCE["sha256"],
            "codec_checkpoint_size":
            NEUCODEC_REFERENCE["size"],
            "codec_tensor_count":
            NEUCODEC_REFERENCE["tensor_count"],
            "codec_source_revision":
            NEUCODEC_SOURCE_REVISION,
            "training_source":
            NEUTTS_TRAINING_SOURCE["repository"],
            "training_source_revision":
            NEUTTS_TRAINING_SOURCE["revision"],
            "training_recipe":
            NEUTTS_TRAINING_SOURCE["recipe"],
            "verified_training_family":
            NEUTTS_TRAINING_SOURCE["model_family"],
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "The pinned completion-only objective is verified for "
                "NeuTTS-Air. Its language model is trainable while native "
                "NeuCodec remains frozen; raw audio or precomputed codes are "
                "accepted. Nano and 2E objectives fail closed until an "
                "author-equivalent recipe is verified."),
            "phonemizer_boundary": (
                "Phoneme variants require precomputed phonemes or an "
                "explicitly injected phonemizer; no eSpeak process or "
                "grapheme approximation runs implicitly."),
        },
    )


def register_neutts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_NEUTTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register NeuTTS without importing its PyTorch graph."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_neutts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_NEUTTS_ALIASES",
    "create_neutts_architecture_spec",
    "register_neutts_architecture",
]
