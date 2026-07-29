"""Lazy declaration for VoiceHub's native OpenVoice V2 converter."""

from __future__ import annotations

from collections.abc import Iterable

from voicehub.architectures.openvoice.metadata import (
    OPENVOICE_CHECKPOINT_REVISION,
    OPENVOICE_CONVERTER_CHECKPOINT,
    OPENVOICE_MODEL_ID,
    OPENVOICE_SOURCE_REVISION,
)
from voicehub.architectures.registry import ARCHITECTURE_REGISTRY, ArchitectureRegistry
from voicehub.architectures.specifications import ArchitectureCapabilities, ArchitectureSpec
from voicehub.tasks import SpeechTask

DEFAULT_OPENVOICE_ALIASES = (
    "native-openvoice",
    "openvoice",
    "openvoice-v2",
)


def create_openvoice_architecture_spec() -> ArchitectureSpec:
    """Describe the audited converter without importing its PyTorch graph."""
    return ArchitectureSpec(
        architecture_id="openvoice-v2-converter",
        version="2",
        model_builder=("voicehub.architectures.openvoice.modeling:"
                       "OpenVoiceToneColorConverter"),
        config=("voicehub.architectures.openvoice.configuration:"
                "OpenVoiceConverterConfig"),
        processor=("voicehub.architectures.openvoice.processing:"
                   "OpenVoiceAudioProcessor"),
        decoder=("voicehub.models.openvoice.source.openvoice.models:Generator"),
        objective=("voicehub.architectures.openvoice.modeling:"
                   "OpenVoiceToneColorConverter"),
        checkpoint_adapter=("voicehub.architectures.openvoice.checkpoint:"
                            "load_openvoice_checkpoint"),
        components={
            "artifact-resolver":
            ("voicehub.architectures.openvoice.artifacts:"
             "resolve_openvoice_artifacts"),
            "runtime": ("voicehub.architectures.openvoice.runtime:OpenVoiceRuntime"),
            "checkpoint-exporter":
            ("voicehub.architectures.openvoice.checkpoint:"
             "save_openvoice_checkpoint"),
            "trainer-adapter": ("voicehub.models.openvoice.training:"
                                "OpenVoiceTrainingAdapter"),
        },
        capabilities=ArchitectureCapabilities(
            tasks=(SpeechTask.TEXT_TO_SPEECH, ),
            devices=("cpu", "cuda"),
            dtypes=("float32", ),
            checkpoint_formats=("safetensors", "pytorch"),
            training=True,
            streaming=False,
            batched_inference=False,
            distributed_training=False,
            export_formats=("safetensors", ),
            optimization_passes=("compile", ),
            features=(
                "tone-color-conversion",
                "voice-cloning",
                "native-magnitude-stft",
                "native-reference-encoder",
                "normalizing-flow",
                "hifigan-decoder",
                "strict-checkpoint-inventory",
                "digest-pinned-weights-only-pytorch-import",
                "safetensors-export",
                "reconstructed-paired-waveform-finetuning",
                "no-upstream-training-parity-claim",
                "no-external-runtime",
            ),
        ),
        upstream_revision=OPENVOICE_SOURCE_REVISION,
        license_id="MIT",
        metadata={
            "implementation":
            "voicehub-native",
            "tensor_backend":
            "pytorch",
            "source_repository":
            "myshell-ai/OpenVoice",
            "source_revision":
            OPENVOICE_SOURCE_REVISION,
            "checkpoint_repository":
            OPENVOICE_MODEL_ID,
            "checkpoint_revision":
            OPENVOICE_CHECKPOINT_REVISION,
            "reference_checkpoint":
            dict(OPENVOICE_CONVERTER_CHECKPOINT),
            "native_checkpoint_format":
            "voicehub-openvoice-v2-v1",
            "official_training_recipe_available":
            False,
            "training_scope":
            "reconstructed-paired-waveform",
            "full_upstream_finetuning_parity":
            False,
            "inference_reloadable_export":
            True,
            "raw_text_frontend":
            ("caller-supplied base waveform or explicit native MeloTTS "
             "linguistic features"),
        },
    )


def register_openvoice_architecture(
    *,
    registry: ArchitectureRegistry | None = None,
    aliases: Iterable[str] = DEFAULT_OPENVOICE_ALIASES,
    exist_ok: bool = False,
) -> ArchitectureSpec:
    target = ARCHITECTURE_REGISTRY if registry is None else registry
    if not isinstance(target, ArchitectureRegistry):
        raise TypeError("`registry` must be an ArchitectureRegistry or None.")
    spec = create_openvoice_architecture_spec()
    target.register(spec, aliases=aliases, exist_ok=exist_ok)
    return spec


__all__ = [
    "DEFAULT_OPENVOICE_ALIASES",
    "create_openvoice_architecture_spec",
    "register_openvoice_architecture",
]
