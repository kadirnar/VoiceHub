"""Lazy native-architecture declaration for Qwen3-TTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.qwen3_tts.metadata import QWEN3_TTS_CHECKPOINTS, QWEN3_TTS_SOURCE
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_QWEN3_TTS_ALIASES = (
    "native-qwen3-tts",
    "qwen3-tts-12hz",
)


def create_qwen3_tts_architecture_spec() -> ArchitectureSpec:
    return ArchitectureSpec(
        architecture_id="qwen3-tts",
        version="1",
        model_builder=("voicehub.architectures.qwen3_tts.modeling:"
                       "Qwen3TTSForConditionalGeneration"),
        config=("voicehub.architectures.qwen3_tts.configuration:"
                "Qwen3TTSArchitectureConfig"),
        processor=("voicehub.architectures.qwen3_tts.runtime:Qwen3TTSProcessor"),
        decoder=("voicehub.architectures.qwen3_tts.codec:Qwen3TTSSpeechDecoder"),
        objective="voicehub.objectives.sequence:sequence_cross_entropy",
        checkpoint_adapter=("voicehub.architectures.qwen3_tts.checkpoint:"
                            "load_qwen3_tts_model_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.architectures.qwen3_tts.artifacts:"
             "resolve_qwen3_tts_artifacts"),
            "speaker-encoder": ("voicehub.architectures.qwen3_tts.modeling:"
                                "Qwen3TTSSpeakerEncoder"),
            "text-tokenizer": ("voicehub.architectures.qwen3_tts.tokenization:"
                               "Qwen3TTSTextTokenizer"),
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
            optimization_passes=("compile", "lora"),
            features=(
                "autoregressive-codebooks",
                "custom-voice",
                "delayed-codebook-sft",
                "multilingual",
                "speaker-encoder",
                "voice-clone-xvector",
                "voice-design",
            ),
        ),
        upstream_revision=str(QWEN3_TTS_SOURCE["revision"]),
        license_id="Apache-2.0",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            QWEN3_TTS_SOURCE,
            "reference_checkpoints":
            QWEN3_TTS_CHECKPOINTS,
            "training_boundary": (
                "Exact official 12 Hz Base single-speaker SFT objective with "
                "pre-extracted 16-codebook targets and frozen speaker encoder."),
            "icl_reference_encoder": (
                "The published Mimi-derived encoder is not yet native; "
                "x-vector cloning is available and ICL reference-audio "
                "cloning remains explicitly unavailable."),
            "reference_audio_boundary": (
                "Native paths and URLs accept PCM WAVE; other containers "
                "must be supplied as predecoded tensors."),
        },
    )


def register_qwen3_tts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_QWEN3_TTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_qwen3_tts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_QWEN3_TTS_ALIASES",
    "create_qwen3_tts_architecture_spec",
    "register_qwen3_tts_architecture",
]
