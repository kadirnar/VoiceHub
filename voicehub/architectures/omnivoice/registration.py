"""Lazy architecture declaration for VoiceHub-native OmniVoice."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.omnivoice.metadata import (
    HIGGS_AUDIO_V2_HEADER_FINGERPRINT,
    HIGGS_AUDIO_V2_MODEL_ID,
    HIGGS_AUDIO_V2_PARAMETER_COUNT,
    HIGGS_AUDIO_V2_REVISION,
    HIGGS_AUDIO_V2_SAFETENSORS_SHA256,
    HIGGS_AUDIO_V2_TENSOR_COUNT,
    OMNIVOICE_MODEL_HEADER_FINGERPRINT,
    OMNIVOICE_MODEL_ID,
    OMNIVOICE_MODEL_PARAMETER_COUNT,
    OMNIVOICE_MODEL_REVISION,
    OMNIVOICE_MODEL_SAFETENSORS_SHA256,
    OMNIVOICE_MODEL_TENSOR_COUNT,
    OMNIVOICE_UPSTREAM_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_OMNIVOICE_ALIASES = (
    "k2-omnivoice",
    "native-omnivoice",
    "omni-voice",
)


def create_omnivoice_architecture_spec() -> ArchitectureSpec:
    """Describe the audited OmniVoice graph without importing PyTorch."""
    return ArchitectureSpec(
        architecture_id="omnivoice",
        version="1",
        model_builder=("voicehub.architectures.omnivoice.modeling:OmniVoiceModel"),
        config=("voicehub.architectures.omnivoice.configuration:"
                "OmniVoiceArchitectureConfig"),
        processor=("voicehub.architectures.omnivoice.processing:"
                   "OmniVoiceSampleProcessor"),
        decoder=("voicehub.architectures.omnivoice.codec:"
                 "HiggsAudioV2Tokenizer"),
        objective=("voicehub.architectures.omnivoice.modeling:"
                   "OmniVoiceModel.forward"),
        checkpoint_adapter=("voicehub.architectures.omnivoice.checkpoint:"
                            "load_omnivoice_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.architectures.omnivoice.artifacts:"
             "resolve_omnivoice_artifacts"),
            "checkpoint-exporter":
            ("voicehub.architectures.omnivoice.checkpoint:"
             "export_omnivoice_checkpoint"),
            "generator": ("voicehub.architectures.omnivoice.generation:"
                          "OmniVoiceGenerator"),
            "padding-collator": ("voicehub.architectures.omnivoice.processing:"
                                 "OmniVoicePaddingCollator"),
            "packing-collator": ("voicehub.architectures.omnivoice.processing:"
                                 "OmniVoicePackingCollator"),
            "runtime": ("voicehub.architectures.omnivoice.runtime:"
                        "load_omnivoice_runtime"),
            "text-tokenizer": ("voicehub.architectures.omnivoice.processing:"
                               "OmniVoiceTokenizer"),
            "trainer-adapter": ("voicehub.models.omnivoice.training:"
                                "OmniVoiceTrainingAdapter"),
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
                "llm-tts-codec",
                "bidirectional-qwen3-backbone",
                "classifier-free-guidance",
                "frozen-higgs-audio-v2",
                "full-finetuning",
                "gradient-checkpointing",
                "iterative-masked-token-decoding",
                "multilingual",
                "preencoded-code-finetuning",
                "raw-audio-finetuning",
                "strict-safetensors-reload",
                "voice-cloning",
                "voice-design",
                "weighted-codebook-masked-cross-entropy",
            ),
        ),
        upstream_revision=OMNIVOICE_UPSTREAM_REVISION,
        license_id="Apache-2.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            "https://github.com/k2-fsa/OmniVoice",
            "source_revision":
            OMNIVOICE_UPSTREAM_REVISION,
            "reference_checkpoint":
            OMNIVOICE_MODEL_ID,
            "reference_checkpoint_revision":
            OMNIVOICE_MODEL_REVISION,
            "reference_checkpoint_sha256": (OMNIVOICE_MODEL_SAFETENSORS_SHA256),
            "reference_tensor_count":
            OMNIVOICE_MODEL_TENSOR_COUNT,
            "reference_parameter_count":
            OMNIVOICE_MODEL_PARAMETER_COUNT,
            "reference_safetensors_header_fingerprint": (OMNIVOICE_MODEL_HEADER_FINGERPRINT),
            "codec_checkpoint":
            HIGGS_AUDIO_V2_MODEL_ID,
            "codec_checkpoint_revision":
            HIGGS_AUDIO_V2_REVISION,
            "codec_checkpoint_sha256":
            HIGGS_AUDIO_V2_SAFETENSORS_SHA256,
            "codec_tensor_count":
            HIGGS_AUDIO_V2_TENSOR_COUNT,
            "codec_parameter_count":
            HIGGS_AUDIO_V2_PARAMETER_COUNT,
            "codec_safetensors_header_fingerprint": (HIGGS_AUDIO_V2_HEADER_FINGERPRINT),
            "checkpoint_license":
            "Apache-2.0",
            "training_boundary": (
                "The complete OmniVoice graph is trainable with the "
                "published independently averaged, normalized eight-codebook "
                "masked cross-entropy objective. Higgs Audio V2 remains "
                "frozen. Training accepts either raw 24 kHz mono waveforms or "
                "pre-encoded [8, frames] codec IDs."),
            "inference_boundary": (
                "Generation performs iterative masked-token decoding for one "
                "utterance per call. Voice cloning requires both reference "
                "audio and its explicit transcript."),
        },
    )


def register_omnivoice_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_OMNIVOICE_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    """Register the native OmniVoice declaration."""
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_omnivoice_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_OMNIVOICE_ALIASES",
    "create_omnivoice_architecture_spec",
    "register_omnivoice_architecture",
]
