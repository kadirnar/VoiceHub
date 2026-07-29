"""Lazy architecture declaration for VoiceHub-native OuteTTS."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.outetts.metadata import (
    OUTETTS_CHECKPOINTS,
    OUTETTS_DAC,
    OUTETTS_SOURCE_LICENSE,
    OUTETTS_SOURCE_REVISION,
    OUTETTS_TRAINING_SOURCE,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_OUTETTS_ALIASES = (
    "native-outetts",
    "oute-tts",
    "outetts-v3",
)


def create_outetts_architecture_spec() -> ArchitectureSpec:
    """Describe the native V3 LM, tokenizer, DAC, and SFT objective."""
    return ArchitectureSpec(
        architecture_id="outetts",
        version="1",
        model_builder=("voicehub.architectures.outetts.modeling:OuteTTSForCausalLM"),
        config=("voicehub.architectures.causal_lm.configuration:CausalLMConfig"),
        processor=("voicehub.architectures.outetts.tokenization:OuteTTSTokenizer"),
        decoder="voicehub.architectures.dac.modeling:DacModel",
        objective=("voicehub.models.outetts.training:OuteTTSTrainingAdapter"),
        checkpoint_adapter=(
            "voicehub.architectures.causal_lm.checkpoint:"
            "HuggingFaceCausalLMCheckpointAdapter"),
        components={
            "artifact-resolver": "voicehub.architectures.outetts.artifacts:"
            "resolve_outetts_artifacts",
            "audio-codec": "voicehub.architectures.dac.modeling:DacModel",
            "prompt-processor": "voicehub.architectures.outetts.prompting:"
            "OuteTTSPromptProcessor",
            "runtime": "voicehub.architectures.outetts.runtime:OuteTTSRuntime",
            "sft-dataset": "voicehub.models.outetts.training:OuteTTSSFTDataset",
            "wrapper": "voicehub.models.outetts.inference:"
            "OuteTTSForTextToSpeech",
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
                "completion-only-codec-language-modeling",
                "frozen-dac",
                "llama-and-qwen-backbones",
                "native-byte-bpe-tokenizer",
                "precomputed-profile-fine-tuning",
                "speaker-conditioning",
                "strict-safetensors-reload",
                "v3-prompt-protocol",
            ),
        ),
        upstream_revision=OUTETTS_SOURCE_REVISION,
        license_id=OUTETTS_SOURCE_LICENSE,
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source":
            "edwko/OuteTTS",
            "source_revision":
            OUTETTS_SOURCE_REVISION,
            "reference_checkpoints": {
                model_id: {
                    "revision": values["revision"],
                    "license": values["license"],
                    "tensor_count": values["tensor_count"],
                }
                for model_id, values in OUTETTS_CHECKPOINTS.items()
            },
            "codec_checkpoint":
            OUTETTS_DAC["repository"],
            "codec_revision":
            OUTETTS_DAC["revision"],
            "codec_license":
            OUTETTS_DAC["license"],
            "training_source":
            OUTETTS_TRAINING_SOURCE["repository"],
            "training_source_revision":
            OUTETTS_TRAINING_SOURCE["revision"],
            "training_recipe":
            OUTETTS_TRAINING_SOURCE["recipe"],
            "full_finetuning_ready":
            True,
            "training_boundary": (
                "Full LM fine-tuning accepts prepared V3 speaker profiles "
                "or tokenized examples. Raw audio requires author-equivalent "
                "word timestamps and therefore fails closed."),
            "inference_boundary": (
                "Native regular/chunked generation is supported. GGUF, "
                "EXL2, vLLM, server, guided-words, batch, and streaming "
                "provider backends are rejected."),
        },
    )


def register_outetts_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_OUTETTS_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_outetts_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_OUTETTS_ALIASES",
    "create_outetts_architecture_spec",
    "register_outetts_architecture",
]
