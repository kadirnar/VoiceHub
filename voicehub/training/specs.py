"""Declarative training profiles for all source-integrated architectures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from voicehub.registry import get_model_spec


class TrainingFamily(str, Enum):
    """The objective and optimization shape used by a TTS architecture."""

    CAUSAL_LM = "causal-lm"
    SEQ2SEQ = "sequence-to-sequence"
    FLOW_MATCHING = "flow-matching"
    ACOUSTIC = "acoustic-regression"
    COMPOSITE = "composite"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ModelTrainingSpec:
    """Everything the generic adapter needs to expose a trainable model."""

    model_type: str
    family: TrainingFamily
    module_paths: tuple[str, ...] = ("model", "model.model")
    component_paths: tuple[str, ...] = ()
    label_names: tuple[str, ...] = (
        "labels",
        "targets",
        "target",
    )
    prediction_keys: tuple[str, ...] = (
        "logits",
        "predictions",
        "audio_values",
        "waveform",
    )
    loss_keys: tuple[str, ...] = (
        "loss",
        "total_loss",
    )
    loss_weights: tuple[tuple[str, float], ...] = ()
    regression_loss: str = "mse"
    source_entrypoints: tuple[str, ...] = ()
    native_training: bool = False
    separate_optimizers: bool = False

    @property
    def install_extra(self) -> str:
        """Return the model dependency extra paired with this profile."""
        return get_model_spec(self.model_type).install_extra


_COMMON_LM_PATHS = (
    "model",
    "model.model",
    "model.llm",
    "model.gpt",
    "model.language_model",
    "model.generator",
)
_COMMON_ACOUSTIC_PATHS = (
    "model",
    "model.model",
    "model.tts_model",
    "model.generator",
    "model.synthesizer",
    "model.decoder",
)
_COMMON_COMPOSITE_PATHS = (
    "model",
    "model.model",
    "model.tts_model",
    "model.generator",
    "model.llm",
)


def _profile(
    model_type: str,
    family: TrainingFamily,
    *,
    module_paths: tuple[str, ...] | None = None,
    component_paths: tuple[str, ...] = (),
    label_names: tuple[str, ...] | None = None,
    loss_weights: tuple[tuple[str, float], ...] = (),
    regression_loss: str = "mse",
    source_entrypoints: tuple[str, ...] = (),
    native_training: bool = False,
    separate_optimizers: bool | None = None,
) -> ModelTrainingSpec:
    if module_paths is None:
        module_paths = (
            _COMMON_LM_PATHS if family is TrainingFamily.CAUSAL_LM else
            _COMMON_COMPOSITE_PATHS if family is TrainingFamily.COMPOSITE else _COMMON_ACOUSTIC_PATHS)
    return ModelTrainingSpec(
        model_type=model_type,
        family=family,
        module_paths=module_paths,
        component_paths=component_paths,
        label_names=label_names or ModelTrainingSpec.label_names,
        loss_weights=loss_weights,
        regression_loss=regression_loss,
        source_entrypoints=source_entrypoints,
        native_training=native_training,
        separate_optimizers=(
            family is TrainingFamily.COMPOSITE if separate_optimizers is None else separate_optimizers),
    )


_MODEL_TRAINING_SPECS = (
    _profile("orpheustts", TrainingFamily.CAUSAL_LM),
    _profile("dia", TrainingFamily.SEQ2SEQ),
    _profile("vui", TrainingFamily.ACOUSTIC, regression_loss="l1"),
    _profile(
        "chatterbox",
        TrainingFamily.FLOW_MATCHING,
        component_paths=("model.t3", "model.s3gen"),
    ),
    _profile("kokoro", TrainingFamily.ACOUSTIC, regression_loss="l1"),
    _profile("echo", TrainingFamily.FLOW_MATCHING),
    _profile("conversationtts", TrainingFamily.CAUSAL_LM),
    _profile("llasa", TrainingFamily.CAUSAL_LM),
    _profile(
        "cosyvoice",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.model.llm",
            "model.model.flow",
            "model.model.hift",
        ),
        loss_weights=(("loss", 1.0), ("generator_loss", 1.0)),
        source_entrypoints=("cosyvoice/bin/train.py", ),
        native_training=True,
    ),
    _profile(
        "f5tts",
        TrainingFamily.FLOW_MATCHING,
        module_paths=("model.ema_model", "model.model", "model"),
        label_names=("labels", "mel_spec", "mel_labels", "target"),
        source_entrypoints=("f5_tts/train/train.py", ),
        native_training=True,
    ),
    _profile(
        "gptsovits",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.t2s_model",
            "model.vits_model",
        ),
        loss_weights=(
            ("semantic_loss", 1.0),
            ("mel_loss", 1.0),
            ("generator_loss", 1.0),
            ("discriminator_loss", 1.0),
        ),
        source_entrypoints=(
            "GPT_SoVITS/s1_train.py",
            "GPT_SoVITS/s2_train.py",
        ),
        native_training=True,
    ),
    _profile(
        "melotts",
        TrainingFamily.ACOUSTIC,
        regression_loss="l1",
        source_entrypoints=("melo/train.py", ),
        native_training=True,
    ),
    _profile("openvoice", TrainingFamily.ACOUSTIC, regression_loss="l1"),
    _profile("outetts", TrainingFamily.CAUSAL_LM),
    _profile("parlertts", TrainingFamily.SEQ2SEQ),
    _profile(
        "styletts2",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.model",
            "model.generator",
            "model.discriminator",
        ),
        loss_weights=(
            ("mel_loss", 5.0),
            ("generator_loss", 1.0),
            ("discriminator_loss", 1.0),
            ("duration_loss", 1.0),
            ("diffusion_loss", 1.0),
        ),
        source_entrypoints=("styletts2/train_finetune.py", ),
        native_training=True,
    ),
    _profile(
        "mosstts",
        TrainingFamily.CAUSAL_LM,
        source_entrypoints=("moss_tts_local_v1_5/finetuning/sft.py", ),
        native_training=True,
    ),
    _profile("qwen3tts", TrainingFamily.CAUSAL_LM),
    _profile(
        "irodoritts",
        TrainingFamily.FLOW_MATCHING,
        label_names=("labels", "audio_labels", "mel_labels", "target"),
    ),
    _profile("zonos", TrainingFamily.CAUSAL_LM),
    _profile("zonos2", TrainingFamily.CAUSAL_LM),
    _profile(
        "voxcpm",
        TrainingFamily.FLOW_MATCHING,
        module_paths=("model.model", "model"),
        source_entrypoints=("voxcpm/training", ),
        native_training=True,
    ),
    _profile(
        "omnivoice",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.model",
            "model.audio_tokenizer",
            "model.flow",
        ),
        source_entrypoints=("omnivoice/training/trainer.py", ),
        native_training=True,
    ),
    _profile("higgstts", TrainingFamily.CAUSAL_LM),
    _profile(
        "xtts",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.gpt",
            "model.hifigan_decoder",
            "model",
        ),
        loss_weights=(
            ("text_ce", 1.0),
            ("mel_ce", 1.0),
            ("generator_loss", 1.0),
            ("discriminator_loss", 1.0),
        ),
        source_entrypoints=("TTS/tts/layers/xtts/trainer/gpt_trainer.py", ),
        native_training=True,
    ),
    _profile("vibevoice", TrainingFamily.SEQ2SEQ),
    _profile(
        "fishtts",
        TrainingFamily.COMPOSITE,
        component_paths=("model", "model.codec"),
        source_entrypoints=("fish_speech/train.py", ),
        native_training=True,
    ),
    _profile("csm", TrainingFamily.CAUSAL_LM),
    _profile("neutts", TrainingFamily.CAUSAL_LM),
    _profile("supertonic", TrainingFamily.ACOUSTIC, regression_loss="l1"),
    _profile("inflecttts", TrainingFamily.ACOUSTIC, regression_loss="l1"),
)

MODEL_TRAINING_SPECS: Mapping[str, ModelTrainingSpec] = MappingProxyType(
    {spec.model_type: spec
     for spec in _MODEL_TRAINING_SPECS})


def get_training_spec(model_type: str) -> ModelTrainingSpec:
    """Resolve aliases and return one mandatory training profile."""
    canonical = get_model_spec(model_type).model_type
    return MODEL_TRAINING_SPECS[canonical]


def list_training_specs() -> tuple[ModelTrainingSpec, ...]:
    """List training profiles without importing any ML framework."""
    return tuple(MODEL_TRAINING_SPECS.values())
