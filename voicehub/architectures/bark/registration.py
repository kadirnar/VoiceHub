"""Lazy architecture declaration for VoiceHub-native Bark."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

from .metadata import (
    BARK_CHECKPOINT,
    BARK_CHECKPOINT_REVISION,
    BARK_INVENTORY_FINGERPRINT,
    BARK_ORIGINAL_SOURCE_REVISION,
    BARK_STATE_VALUES,
    BARK_TENSOR_COUNT,
    BARK_TRANSFORMERS_SOURCE_REVISION,
)

DEFAULT_BARK_ALIASES = (
    "native-bark",
    "bark-tts",
    "suno-bark-small",
)


def create_bark_architecture_spec() -> ArchitectureSpec:
    """Describe Bark without importing PyTorch or allocating the graph."""
    return ArchitectureSpec(
        architecture_id="bark",
        version="1",
        model_builder="voicehub.architectures.bark.modeling:BarkModel",
        config=("voicehub.architectures.bark.configuration:"
                "BarkArchitectureConfig"),
        processor="voicehub.architectures.bark.processing:BarkProcessor",
        decoder=("voicehub.components.audio.codecs.encodec.model:"
                 "EncodecModel"),
        objective=("voicehub.architectures.bark.training:"
                   "BarkTrainingAdapter"),
        checkpoint_adapter=("voicehub.architectures.bark.checkpoint:"
                            "load_bark_safetensors"),
        components={
            "artifact-resolver": ("voicehub.architectures.bark.artifacts:"
                                  "resolve_bark_artifacts"),
            "checkpoint-converter":
            ("voicehub.architectures.bark.checkpoint:"
             "convert_official_bark_checkpoint"),
            "checkpoint-exporter": ("voicehub.architectures.bark.checkpoint:"
                                    "save_bark_safetensors"),
            "coarse-model": ("voicehub.architectures.bark.modeling:"
                             "BarkCoarseModel"),
            "fine-model": ("voicehub.architectures.bark.modeling:"
                           "BarkFineModel"),
            "inference-runtime": ("voicehub.models.bark.inference:"
                                  "BarkForTextToSpeech"),
            "semantic-model": ("voicehub.architectures.bark.modeling:"
                               "BarkSemanticModel"),
            "wordpiece-tokenizer": ("voicehub.architectures.bark.processing:"
                                    "BarkWordPieceTokenizer"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda", "mps"),
            dtypes=("float32", "float16", "bfloat16"),
            checkpoint_formats=("safetensors", "pytorch"),
            training=True,
            streaming=False,
            batched_inference=True,
            distributed_training=True,
            export_formats=("safetensors", ),
            optimization_passes=("compile", ),
            features=(
                "three-stage-token-generation",
                "native-multilingual-wordpiece",
                "speaker-history-conditioning",
                "native-encodec",
                "pretokenized-semantic-fine-tuning",
                "pretokenized-coarse-fine-tuning",
                "pretokenized-fine-codebook-fine-tuning",
                "frozen-codec",
                "strict-checkpoint-inventory",
                "restricted-pytorch-import",
                "safetensors-export",
                "no-external-runtime",
            ),
        ),
        upstream_revision=BARK_ORIGINAL_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "family": "bark",
            "implementation": "voicehub-native",
            "tensor_backend": "pytorch",
            "reference_checkpoint": BARK_CHECKPOINT,
            "reference_checkpoint_revision": BARK_CHECKPOINT_REVISION,
            "reference_tensor_count": BARK_TENSOR_COUNT,
            "reference_state_values": BARK_STATE_VALUES,
            "reference_inventory_fingerprint": (BARK_INVENTORY_FINGERPRINT),
            "transformers_reference_revision": (BARK_TRANSFORMERS_SOURCE_REVISION),
            "official_safetensors_published": False,
            "checkpoint_import_boundary": ("digest-pinned-weights-only-explicit-trust"),
            "training_scope": "pretokenized-stage-specific",
            "raw_audio_finetuning_ready": False,
            "full_finetuning_ready": False,
            "always_frozen_components": ("codec_model", ),
            "sampling_rate": 24_000,
            "languages": (
                "de",
                "en",
                "es",
                "fr",
                "hi",
                "it",
                "ja",
                "ko",
                "pl",
                "pt",
                "ru",
                "tr",
                "zh",
            ),
        },
    )


def register_bark_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_BARK_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_bark_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_BARK_ALIASES",
    "create_bark_architecture_spec",
    "register_bark_architecture",
]
