"""Declarative, framework-lazy training profiles for audio architectures."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any

from voicehub.errors import UnknownModelError
from voicehub.registry import get_model_spec, normalize_model_type
from voicehub.tasks import SpeechTask
from voicehub.training.contracts import (
    TrainingContext,
    TrainingPhaseKind,
    TrainingPhaseSpec,
    TrainingRecipeKind,
    TrainingSupport,
)


class TrainingFamily(str, Enum):
    """Built-in objective and optimization shapes.

    ``ModelTrainingSpec.family`` also accepts a non-empty string. This
    enum is therefore a convenience for the built-ins, not a closed
    extension point.
    """

    CAUSAL_LM = "causal-lm"
    SEQ2SEQ = "sequence-to-sequence"
    FLOW_MATCHING = "flow-matching"
    ACOUSTIC = "acoustic-regression"
    VITS = "vits"
    COMPOSITE = "composite"
    CTC = "ctc"
    SPEECH_SEQ2SEQ = "speech-sequence-to-sequence"
    RNNT = "rnnt"
    TDT = "tdt"
    AUDIO_CLASSIFICATION = "audio-classification"
    FRAME_CLASSIFICATION = "frame-classification"
    UPSTREAM_NATIVE = "upstream-native"

    def __str__(self) -> str:
        return self.value


def _strings(value: Iterable[str] | str, *, name: str) -> tuple[str, ...]:
    values = (value, ) if isinstance(value, str) else tuple(value)
    normalized = []
    for item in values:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{name} must contain non-empty strings.")
        normalized.append(item.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate values.")
    return tuple(normalized)


def _pairs(value: Mapping[str, Any] | Iterable[tuple[str, Any]], *, name: str):
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"{name} entries must be two-item pairs.")
        key, item_value = item
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings.")
        key = key.strip()
        if key in seen:
            raise ValueError(f"{name} contains duplicate key {key!r}.")
        seen.add(key)
        normalized.append((key, item_value))
    return tuple(normalized)


@dataclass(frozen=True)
class ModelTrainingSpec:
    """Everything a training adapter needs to expose a source runtime.

    The original, single-objective fields remain supported. When
    ``phases`` is omitted they are projected into one phase named
    ``default``. New backends can instead declare multiple independently
    scheduled component phases.
    """

    model_type: str
    family: TrainingFamily | str
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
    support: TrainingSupport = TrainingSupport.PREPROCESSED
    phases: tuple[TrainingPhaseSpec, ...] = ()
    default_phase: str | None = None
    fallback_objective: str | None = None
    recipe_kind: TrainingRecipeKind = TrainingRecipeKind.SINGLE_PHASE
    allow_module_discovery: bool = False
    training_default_model_name_or_path: str | None = None
    field_schemas: Mapping[str, Any] = field(default_factory=dict)
    task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH

    def __post_init__(self) -> None:
        if not isinstance(self.model_type, str) or not self.model_type.strip():
            raise ValueError("model_type must be a non-empty string.")
        object.__setattr__(
            self,
            "model_type",
            normalize_model_type(self.model_type),
        )
        object.__setattr__(
            self,
            "task",
            SpeechTask.coerce(self.task),
        )

        family = self.family
        if isinstance(family, str) and not isinstance(family, TrainingFamily):
            family = family.strip().lower()
            if not family:
                raise ValueError("family must be a TrainingFamily or non-empty string.")
            try:
                family = TrainingFamily(family)
            except ValueError:
                pass
        elif not isinstance(family, TrainingFamily):
            raise TypeError("family must be a TrainingFamily or non-empty string.")
        object.__setattr__(self, "family", family)

        for field_name in (
                "module_paths",
                "component_paths",
                "label_names",
                "prediction_keys",
                "loss_keys",
                "source_entrypoints",
        ):
            object.__setattr__(
                self,
                field_name,
                _strings(getattr(self, field_name), name=field_name),
            )
        if not self.module_paths:
            raise ValueError("module_paths must contain at least one candidate path.")

        weights = _pairs(self.loss_weights, name="loss_weights")
        normalized_weights = []
        for loss_name, weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError("loss_weights values must be real numbers.")
            normalized_weights.append((loss_name, float(weight)))
        object.__setattr__(self, "loss_weights", tuple(normalized_weights))

        if self.regression_loss not in ("l1", "mse"):
            raise ValueError("regression_loss must be either 'l1' or 'mse'.")
        if not isinstance(self.native_training, bool):
            raise TypeError("native_training must be a boolean.")
        if not isinstance(self.separate_optimizers, bool):
            raise TypeError("separate_optimizers must be a boolean.")
        if not isinstance(self.allow_module_discovery, bool):
            raise TypeError("allow_module_discovery must be a boolean.")
        training_default = self.training_default_model_name_or_path
        if training_default is not None:
            if not isinstance(training_default, str) or not training_default.strip():
                raise ValueError(
                    "training_default_model_name_or_path must be a non-empty "
                    "string or None.")
            object.__setattr__(
                self,
                "training_default_model_name_or_path",
                training_default.strip(),
            )
        if not isinstance(self.field_schemas, Mapping):
            raise TypeError("field_schemas must be a mapping of dotted paths to schemas.")
        normalized_schemas = {}
        for path, schema in self.field_schemas.items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError("field_schemas keys must be non-empty dotted paths.")
            if isinstance(schema, Mapping):
                schema = MappingProxyType(dict(schema))
            normalized_schemas[path.strip()] = schema
        object.__setattr__(
            self,
            "field_schemas",
            MappingProxyType(normalized_schemas),
        )
        object.__setattr__(
            self,
            "support",
            TrainingSupport.coerce(self.support),
        )
        object.__setattr__(
            self,
            "recipe_kind",
            TrainingRecipeKind.coerce(self.recipe_kind),
        )

        fallback = self.fallback_objective
        if fallback is None:
            fallback = self._legacy_fallback_objective()
        elif not isinstance(fallback, str) or not fallback.strip():
            raise ValueError("fallback_objective must be a non-empty string or None.")
        if fallback is not None:
            fallback = fallback.strip().lower().replace("-", "_")
        object.__setattr__(self, "fallback_objective", fallback)

        phases = tuple(self.phases)
        if any(not isinstance(phase, TrainingPhaseSpec) for phase in phases):
            raise TypeError("phases must contain TrainingPhaseSpec instances.")
        phase_names = [phase.name for phase in phases]
        if len(set(phase_names)) != len(phase_names):
            raise ValueError("Training phase names must be unique within a model profile.")

        default_phase = self.default_phase
        if default_phase is not None:
            if not isinstance(default_phase, str) or not default_phase.strip():
                raise ValueError("default_phase must be a non-empty phase name or None.")
            default_phase = default_phase.strip()
        if not phases:
            default_phase = default_phase or "default"
            phases = (
                TrainingPhaseSpec(
                    name=default_phase,
                    label_names=self.label_names,
                    prediction_keys=self.prediction_keys,
                    loss_keys=self.loss_keys,
                    loss_weights=self.loss_weights,
                    fallback_objective=fallback,
                ), )
        else:
            default_phase = default_phase or phases[0].name
            if default_phase not in phase_names:
                raise ValueError(f"default_phase {default_phase!r} is not present in phases.")
            if (len(phases) > 1 and self.recipe_kind is TrainingRecipeKind.SINGLE_PHASE):
                object.__setattr__(
                    self,
                    "recipe_kind",
                    TrainingRecipeKind.MULTI_PHASE,
                )
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "default_phase", default_phase)
        self._validate_phase_schedule(phases)

    @staticmethod
    def _validate_phase_schedule(phases: tuple[TrainingPhaseSpec, ...]) -> None:
        """Reject schedules that leave the Trainer unable to advance a step."""
        period = 1
        for phase in phases:
            period = math.lcm(period, phase.frequency)
            if period > 100_000:
                raise ValueError(
                    "The combined training phase schedule is too large to "
                    "validate; reduce phase frequencies.")
        uncovered = [step for step in range(period) if not any(phase.is_scheduled(step) for phase in phases)]
        if uncovered:
            preview = ", ".join(str(step) for step in uncovered[:8])
            raise ValueError(
                "Training phases must cover every recipe step. "
                f"No phase is scheduled at period position(s): {preview}.")

    def _legacy_fallback_objective(self) -> str | None:
        if self.family is TrainingFamily.CAUSAL_LM:
            return "causal_cross_entropy"
        if self.family in (
                TrainingFamily.SEQ2SEQ,
                TrainingFamily.SPEECH_SEQ2SEQ,
        ):
            return "cross_entropy"
        if self.family is TrainingFamily.ACOUSTIC:
            return self.regression_loss
        if self.family in (
                TrainingFamily.AUDIO_CLASSIFICATION,
                TrainingFamily.FRAME_CLASSIFICATION,
        ):
            return "classification"
        if self.family is TrainingFamily.COMPOSITE:
            return "auto"
        # Flow, CTC, RNN-T, TDT, and source-native objectives cannot be
        # reconstructed safely from an arbitrary prediction/label pair.
        return None

    @property
    def family_name(self) -> str:
        return self.family.value if isinstance(self.family, TrainingFamily) else self.family

    @property
    def supports_training(self) -> bool:
        """Whether the built-in family adapter can be used directly."""
        return self.is_turnkey

    @property
    def has_training_recipe(self) -> bool:
        """Whether a generic or source-native recipe is represented."""
        return self.support.is_trainable

    @property
    def requires_custom_adapter(self) -> bool:
        return self.support is TrainingSupport.CUSTOM

    @property
    def is_turnkey(self) -> bool:
        return self.support in (
            TrainingSupport.NATIVE,
            TrainingSupport.PREPROCESSED,
        )

    @property
    def phase_map(self) -> Mapping[str, TrainingPhaseSpec]:
        return MappingProxyType({phase.name: phase for phase in self.phases})

    def get_phase(self, name: str | None = None) -> TrainingPhaseSpec:
        """Resolve a phase by name, defaulting to ``default_phase``."""
        selected = self.default_phase if name is None else name
        if not isinstance(selected, str):
            raise TypeError("Training phase names must be strings.")
        try:
            return self.phase_map[selected]
        except KeyError as exc:
            available = ", ".join(self.phase_map)
            raise ValueError(
                f"Unknown training phase {selected!r} for {self.model_type!r}. "
                f"Available phases: {available}.") from exc

    @property
    def install_extra(self) -> str:
        """Return the one dependency extra used by every training profile."""
        return "training"


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


def _phase(name: str, **kwargs) -> TrainingPhaseSpec:
    return TrainingPhaseSpec(name=name, **kwargs)


def _profile(
    model_type: str,
    family: TrainingFamily,
    *,
    task: SpeechTask | str = SpeechTask.TEXT_TO_SPEECH,
    module_paths: tuple[str, ...] | None = None,
    component_paths: tuple[str, ...] = (),
    label_names: tuple[str, ...] | None = None,
    loss_weights: tuple[tuple[str, float], ...] = (),
    regression_loss: str = "mse",
    source_entrypoints: tuple[str, ...] = (),
    native_training: bool = False,
    separate_optimizers: bool | None = None,
    support: TrainingSupport = TrainingSupport.PREPROCESSED,
    phases: tuple[TrainingPhaseSpec, ...] = (),
    default_phase: str | None = None,
    fallback_objective: str | None = None,
    recipe_kind: TrainingRecipeKind = TrainingRecipeKind.SINGLE_PHASE,
    allow_module_discovery: bool = False,
    training_default_model_name_or_path: str | None = None,
    field_schemas: Mapping[str, Any] | None = None,
) -> ModelTrainingSpec:
    if module_paths is None:
        module_paths = (
            _COMMON_LM_PATHS if family is TrainingFamily.CAUSAL_LM else
            _COMMON_COMPOSITE_PATHS if family is TrainingFamily.COMPOSITE else _COMMON_ACOUSTIC_PATHS)
    return ModelTrainingSpec(
        model_type=model_type,
        family=family,
        task=task,
        module_paths=module_paths,
        component_paths=component_paths,
        label_names=label_names or ModelTrainingSpec.label_names,
        loss_weights=loss_weights,
        regression_loss=regression_loss,
        source_entrypoints=source_entrypoints,
        native_training=native_training,
        separate_optimizers=(
            family in (TrainingFamily.COMPOSITE,
                       TrainingFamily.VITS) if separate_optimizers is None else separate_optimizers),
        support=support,
        phases=phases,
        default_phase=default_phase,
        fallback_objective=fallback_objective,
        recipe_kind=recipe_kind,
        allow_module_discovery=allow_module_discovery,
        training_default_model_name_or_path=(training_default_model_name_or_path),
        field_schemas=field_schemas or {},
    )


_STYLE_GENERATOR_PATHS = (
    # The first path retains compatibility with simple runtime facades.
    "model.generator",
    "model.model.bert",
    "model.model.bert_encoder",
    "model.model.predictor",
    "model.model.decoder",
    "model.model.text_encoder",
    "model.model.predictor_encoder",
    "model.model.style_encoder",
    "model.model.diffusion",
    "model.model.text_aligner",
    "model.model.pitch_extractor",
)
_STYLE_DISCRIMINATOR_PATHS = (
    "model.discriminator",
    "model.model.mpd",
    "model.model.msd",
    "model.model.wd",
)

_BUILTIN_TRAINING_SPECS = (
    _profile("orpheustts", TrainingFamily.CAUSAL_LM),
    _profile(
        "dia",
        TrainingFamily.SEQ2SEQ,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("transformers.models.dia.modeling_dia."
                            "DiaForConditionalGeneration.forward", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    "labels",
                ),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "vui",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.vui.training.VuiTrainingAdapter", ),
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("audio_codes", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "codec_ce_loss"),
                required_inputs=("input_ids", "audio_codes"),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "chatterbox",
        TrainingFamily.FLOW_MATCHING,
        component_paths=("model.t3", "model.s3gen.flow"),
        separate_optimizers=True,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "language_model",
                component_paths=("model.t3", ),
                optimizer_names=("language_model", ),
                loss_keys=("loss", "text_loss", "speech_token_loss"),
            ),
            _phase(
                "flow",
                component_paths=("model.s3gen.flow", ),
                optimizer_names=("flow", ),
                label_names=("labels", "mel_spec", "mel_labels", "target"),
                loss_keys=("loss", "flow_loss", "diffusion_loss"),
            ),
        ),
        default_phase="language_model",
    ),
    _profile(
        "kokoro",
        TrainingFamily.ACOUSTIC,
        regression_loss="l1",
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "echo",
        TrainingFamily.FLOW_MATCHING,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.echo.training.EchoTrainingAdapter", ),
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "flow",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("target_latents", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "flow_loss"),
                required_inputs=(
                    "target_latents",
                    "text_input_ids",
                    "text_mask",
                    "speaker_latents",
                    "speaker_mask",
                ),
            ), ),
        default_phase="flow",
    ),
    _profile(
        "conversationtts",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model", ),
        component_paths=("model", ),
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=(
                    "loss",
                    "codebook0_loss",
                    "residual_loss",
                ),
                required_inputs=(
                    "tokens",
                    "labels",
                    "tokens_mask",
                ),
            ), ),
    ),
    _profile("llasa", TrainingFamily.CAUSAL_LM),
    _profile(
        "cosyvoice",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.model.llm",
            "model.model.flow",
            "model.model.hift.generator",
            "model.model.hift.discriminator",
        ),
        loss_weights=(("loss", 1.0), ("generator_loss", 1.0)),
        source_entrypoints=("cosyvoice/bin/train.py", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "language_model",
                component_paths=("model.model.llm", ),
                optimizer_names=("language_model", ),
                loss_keys=("loss", "text_loss", "speech_token_loss"),
            ),
            _phase(
                "flow",
                component_paths=("model.model.flow", ),
                optimizer_names=("flow", ),
                label_names=("labels", "speech_feat", "mel_spec", "target"),
                loss_keys=("loss", "flow_loss", "l1_loss"),
            ),
            _phase(
                "vocoder_generator",
                component_paths=("model.model.hift.generator", ),
                optimizer_names=("vocoder_generator", ),
                forward_component="model.model.hift",
                label_names=("labels", "speech", "audio_values", "target"),
                prediction_keys=("audio_values", "waveform", "predictions"),
                loss_keys=("loss", "generator_loss", "mel_loss"),
                required_inputs=("batch", "device"),
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=("model.model.hift.discriminator", ),
            ),
            _phase(
                "vocoder_discriminator",
                component_paths=("model.model.hift.discriminator", ),
                optimizer_names=("vocoder_discriminator", ),
                forward_component="model.model.hift",
                label_names=("labels", "speech", "audio_values", "target"),
                loss_keys=("loss", "discriminator_loss", "loss_disc"),
                required_inputs=("batch", "device"),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                detach_inputs=("generated_audio", "fake_audio", "y_hat"),
                frozen_component_paths=("model.model.hift.generator", ),
            ),
        ),
        default_phase="language_model",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "f5tts",
        TrainingFamily.FLOW_MATCHING,
        module_paths=("model.ema_model", "model.model", "model"),
        component_paths=("model.ema_model", ),
        label_names=("labels", "mel_spec", "mel_labels", "target"),
        source_entrypoints=("f5_tts/train/train.py", ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "flow",
                component_paths=("model.ema_model", ),
                optimizer_names=("model", ),
                label_names=("mel", "mel_spec", "input_values"),
                prediction_keys=("predictions", ),
                loss_keys=("loss", ),
                required_inputs=("inp", "text"),
            ), ),
    ),
    _profile(
        "gptsovits",
        TrainingFamily.COMPOSITE,
        component_paths=(
            "model.t2s_model",
            "model.t2s_model.model",
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
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "semantic",
                component_paths=("model.t2s_model.model", ),
                optimizer_names=("semantic", ),
                forward_component="model.t2s_model.model",
                label_names=("labels", "semantic_labels", "target"),
                loss_keys=("loss", "semantic_loss"),
            ),
            _phase(
                "acoustic_generator",
                component_paths=("model.vits_model", ),
                optimizer_names=("acoustic_generator", ),
                label_names=("labels", "mel_spec", "audio_values", "target"),
                loss_keys=("loss", "mel_loss", "generator_loss"),
                kind=TrainingPhaseKind.GENERATOR,
            ),
            _phase(
                "acoustic_discriminator",
                optimizer_names=("acoustic_discriminator", ),
                label_names=("labels", "audio_values", "target"),
                loss_keys=("loss", "discriminator_loss"),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                detach_inputs=("generated_audio", "fake_audio", "y_hat"),
            ),
        ),
        default_phase="semantic",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "melotts",
        TrainingFamily.VITS,
        component_paths=("model.model", ),
        regression_loss="l1",
        source_entrypoints=("melo/train.py", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "generator",
                component_paths=("model.model", ),
                optimizer_names=("generator", ),
                label_names=("labels", "spectrogram", "audio_values", "target"),
                prediction_keys=("audio_values", "waveform", "predictions", "logits"),
                loss_keys=("loss", "mel_loss", "generator_loss"),
                fallback_objective="l1",
                kind=TrainingPhaseKind.GENERATOR,
            ),
            _phase(
                "discriminator",
                optimizer_names=("discriminator", ),
                label_names=("labels", "audio_values", "target"),
                loss_keys=("loss", "discriminator_loss"),
                kind=TrainingPhaseKind.DISCRIMINATOR,
                detach_inputs=("generated_audio", "fake_audio", "y_hat"),
            ),
            _phase(
                "duration_discriminator",
                optimizer_names=("duration_discriminator", ),
                label_names=("labels", "duration_labels", "target"),
                loss_keys=("loss", "duration_discriminator_loss"),
                kind=TrainingPhaseKind.DURATION_DISCRIMINATOR,
                detach_inputs=("predicted_durations", "duration_predictions"),
            ),
        ),
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "openvoice",
        TrainingFamily.VITS,
        component_paths=("model.model", ),
        regression_loss="l1",
        support=TrainingSupport.INFERENCE_ONLY,
        phases=(
            _phase(
                "generator",
                component_paths=("model.model", ),
                optimizer_names=("generator", ),
                label_names=("labels", "spectrogram", "audio_values", "target"),
                prediction_keys=("audio_values", "waveform", "predictions", "logits"),
                loss_keys=("loss", "mel_loss", "generator_loss"),
                fallback_objective="l1",
                kind=TrainingPhaseKind.GENERATOR,
            ), ),
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "outetts",
        TrainingFamily.CAUSAL_LM,
        module_paths=(
            "model.model.model",
            "model.model.model.model",
        ),
        component_paths=("model.model.model", ),
    ),
    _profile("parlertts", TrainingFamily.SEQ2SEQ, support=TrainingSupport.NATIVE),
    _profile(
        "styletts2",
        TrainingFamily.VITS,
        component_paths=_STYLE_GENERATOR_PATHS + _STYLE_DISCRIMINATOR_PATHS,
        loss_weights=(
            ("mel_loss", 5.0),
            ("generator_loss", 1.0),
            ("discriminator_loss", 1.0),
            ("duration_loss", 1.0),
            ("diffusion_loss", 1.0),
        ),
        source_entrypoints=("styletts2/train_finetune.py", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "generator",
                component_paths=_STYLE_GENERATOR_PATHS,
                optimizer_names=("generator", ),
                label_names=("labels", "mel_spec", "audio_values", "target"),
                loss_keys=(
                    "loss",
                    "mel_loss",
                    "generator_loss",
                    "duration_loss",
                    "diffusion_loss",
                ),
                loss_weights=(
                    ("mel_loss", 5.0),
                    ("generator_loss", 1.0),
                    ("duration_loss", 1.0),
                    ("diffusion_loss", 1.0),
                ),
                fallback_objective="auto",
                kind=TrainingPhaseKind.GENERATOR,
                frozen_component_paths=_STYLE_DISCRIMINATOR_PATHS,
            ),
            _phase(
                "discriminator",
                component_paths=_STYLE_DISCRIMINATOR_PATHS,
                optimizer_names=("discriminator", ),
                label_names=("labels", "real_audio", "audio_values", "target"),
                loss_keys=("loss", "discriminator_loss"),
                input_aliases=(("input_values", "input"), ),
                fallback_objective="mse",
                kind=TrainingPhaseKind.DISCRIMINATOR,
                detach_inputs=("generated_audio", "fake_audio", "y_hat"),
                frozen_component_paths=_STYLE_GENERATOR_PATHS,
            ),
        ),
        default_phase="generator",
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "mosstts",
        TrainingFamily.CAUSAL_LM,
        source_entrypoints=("moss_tts_local_v1_5/finetuning/sft.py", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
    ),
    _profile(
        "qwen3tts",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model.model", ),
        component_paths=("model.model.talker", ),
        support=TrainingSupport.PREPROCESSED,
        native_training=True,
        training_default_model_name_or_path=("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        source_entrypoints=("Qwen3-TTS/finetuning/sft_12hz.py", ),
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model.model.talker", ),
                optimizer_names=("model", ),
                label_names=("codec_0_labels", ),
                prediction_keys=("logits", ),
                loss_keys=(
                    "loss",
                    "talker_loss",
                    "sub_talker_loss",
                ),
                required_inputs=(
                    "input_ids",
                    "codec_ids",
                    "ref_mels",
                    "text_embedding_mask",
                    "codec_embedding_mask",
                    "attention_mask",
                    "codec_0_labels",
                    "codec_mask",
                ),
            ), ),
    ),
    _profile(
        "irodoritts",
        TrainingFamily.FLOW_MATCHING,
        module_paths=("model.model", ),
        component_paths=("model.model", ),
        label_names=(
            "velocity_target",
            "labels",
            "audio_labels",
            "mel_labels",
            "target",
        ),
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "flow",
                component_paths=("model.model", ),
                optimizer_names=("model", ),
                label_names=("velocity_target", "labels"),
                prediction_keys=("velocity", "predictions", "logits"),
                loss_keys=("loss", "flow_loss"),
                required_inputs=("velocity_target", ),
                fallback_objective="velocity_mse",
            ), ),
    ),
    _profile(
        "zonos",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=("voicehub.models.zonos.training.ZonosTrainingAdapter", ),
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("audio_codes", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "codec_ce_loss"),
                required_inputs=("prefix_conditioning", "audio_codes"),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "zonos2",
        TrainingFamily.CAUSAL_LM,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "voxcpm",
        TrainingFamily.FLOW_MATCHING,
        module_paths=("model.tts_model", "model"),
        component_paths=("model.tts_model", ),
        source_entrypoints=("voxcpm/training", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "flow",
                component_paths=("model.tts_model", ),
                optimizer_names=("model", ),
                prediction_keys=("feat_pred", "predictions", "logits"),
                loss_keys=("loss", "loss/diff", "loss/stop"),
                loss_weights=(("loss/diff", 1.0), ("loss/stop", 1.0)),
            ), ),
    ),
    _profile(
        "omnivoice",
        TrainingFamily.COMPOSITE,
        component_paths=("model", "model.llm"),
        source_entrypoints=("omnivoice/training/trainer.py", ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", "audio_labels", "target"),
                loss_keys=("loss", "text_loss", "audio_loss"),
            ), ),
        field_schemas={
            "input_ids": {
                "sequence_dim": -1,
            },
            "labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
        },
    ),
    _profile(
        "higgstts",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model.model", ),
        component_paths=("model.model", ),
        support=TrainingSupport.PREPROCESSED,
        native_training=False,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model.model", ),
                optimizer_names=("model", ),
                label_names=("label_ids", "label_audio_ids"),
                prediction_keys=("logits", "audio_logits"),
                loss_keys=("loss", "text_loss", "audio_loss"),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "label_ids",
                    "label_audio_ids",
                ),
            ), ),
    ),
    _profile(
        "xtts",
        TrainingFamily.COMPOSITE,
        module_paths=("model.gpt", ),
        component_paths=("model.gpt", ),
        loss_weights=(
            ("text_ce", 1.0),
            ("mel_ce", 1.0),
        ),
        source_entrypoints=("TTS/tts/layers/xtts/trainer/gpt_trainer.py", ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        separate_optimizers=False,
        phases=(
            _phase(
                "language_model",
                component_paths=("model.gpt", ),
                optimizer_names=("language_model", ),
                label_names=("labels", "audio_codes", "target"),
                loss_keys=("loss", "text_ce", "mel_ce"),
            ), ),
        default_phase="language_model",
    ),
    _profile(
        "vibevoice",
        TrainingFamily.COMPOSITE,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "voicehub.models.vibevoice.training.VibeVoiceTrainingAdapter",
            "microsoft/VibeVoice/finetuning",
        ),
        native_training=True,
        support=TrainingSupport.PREPROCESSED,
        training_default_model_name_or_path="microsoft/VibeVoice-1.5B",
        phases=(
            _phase(
                "lm_diffusion",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("input_ids", "speeches_loss_input"),
                prediction_keys=("logits", ),
                loss_keys=("loss", "ce_loss", "diffusion_loss"),
                required_inputs=(
                    "input_ids",
                    "attention_mask",
                    "speech_tensors",
                    "speech_masks",
                    "speeches_loss_input",
                    "speech_semantic_tensors",
                    "acoustic_input_mask",
                    "acoustic_loss_mask",
                ),
            ), ),
        default_phase="lm_diffusion",
    ),
    _profile(
        "fishtts",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model", ),
        component_paths=("model", ),
        source_entrypoints=(
            "fish_speech/models/text2semantic/lit_module.py",
            "fish_speech/configs/text2semantic_finetune.yaml",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "semantic",
                component_paths=("model", ),
                optimizer_names=("semantic", ),
                label_names=("labels", ),
                loss_keys=("loss", "base_loss", "semantic_loss"),
            ), ),
        default_phase="semantic",
    ),
    _profile(
        "csm",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "transformers.CsmForConditionalGeneration",
            "transformers.CsmProcessor",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "codec_language_model",
                component_paths=("model", ),
                optimizer_names=("codec_language_model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", "backbone_loss", "depth_decoder_loss"),
            ), ),
        default_phase="codec_language_model",
    ),
    _profile(
        "neutts",
        TrainingFamily.CAUSAL_LM,
        module_paths=("model.backbone", ),
        component_paths=("model.backbone", ),
    ),
    _profile(
        "supertonic",
        TrainingFamily.ACOUSTIC,
        regression_loss="l1",
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "inflecttts",
        TrainingFamily.VITS,
        component_paths=("model.model", ),
        regression_loss="l1",
        support=TrainingSupport.INFERENCE_ONLY,
        phases=(
            _phase(
                "generator",
                component_paths=("model.model", ),
                optimizer_names=("generator", ),
                label_names=("labels", "spectrogram", "audio_values", "target"),
                prediction_keys=("audio_values", "waveform", "predictions", "logits"),
                loss_keys=("loss", "mel_loss", "generator_loss"),
                fallback_objective="l1",
                kind=TrainingPhaseKind.GENERATOR,
            ), ),
        recipe_kind=TrainingRecipeKind.ADVERSARIAL,
    ),
    _profile(
        "bark",
        TrainingFamily.COMPOSITE,
        module_paths=("training_model.semantic", ),
        component_paths=(
            "training_model.semantic",
            "training_model.coarse",
            "training_model.fine",
        ),
        label_names=("labels", ),
        source_entrypoints=(
            "transformers.BarkModel",
            "transformers.BarkSemanticModel",
            "transformers.BarkCoarseModel",
            "transformers.BarkFineModel",
        ),
        # Transformers exposes the three Bark submodel logits but rejects
        # labels. VoiceHub owns the stage-specific cross-entropy objectives,
        # so this is a verified pre-tokenized route, not a native-loss claim.
        native_training=False,
        separate_optimizers=True,
        support=TrainingSupport.PREPROCESSED,
        phases=(
            _phase(
                "semantic",
                component_paths=("training_model.semantic", ),
                optimizer_names=("semantic", ),
                forward_component="training_model.semantic",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ),
            _phase(
                "coarse",
                component_paths=("training_model.coarse", ),
                optimizer_names=("coarse", ),
                forward_component="training_model.coarse",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ),
            _phase(
                "fine",
                component_paths=("training_model.fine", ),
                optimizer_names=("fine", ),
                forward_component="training_model.fine",
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels", "codebook_idx"),
            ),
        ),
        default_phase="semantic",
        recipe_kind=TrainingRecipeKind.MULTI_PHASE,
        field_schemas={
            "semantic_input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "semantic_labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
            "coarse_input_ids": {
                "sequence_dim": -1,
                "padding_value": 0,
            },
            "coarse_labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
            "fine_input_ids": {
                "sequence_dim": -2,
                "padding_value": 0,
            },
            "fine_labels": {
                "sequence_dim": -1,
                "padding_value": -100,
            },
        },
    ),
    _profile(
        "speecht5",
        TrainingFamily.SEQ2SEQ,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "transformers.SpeechT5ForTextToSpeech",
            "transformers.SpeechT5Processor",
        ),
        native_training=True,
        separate_optimizers=False,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "spectrogram",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("spectrogram", ),
                loss_keys=("loss", ),
                required_inputs=("input_ids", "labels"),
            ), ),
        default_phase="spectrogram",
        field_schemas={
            "labels": {
                "sequence_dim": -2,
                "padding_value": -100.0,
            },
            "speaker_embeddings": {
                "sequence_dim": -1,
                "padding_value": 0.0,
            },
        },
    ),
    _profile(
        "vits",
        TrainingFamily.VITS,
        module_paths=("training_model", ),
        component_paths=("training_model", ),
        label_names=("audio_values", ),
        source_entrypoints=(
            "transformers.VitsModel",
            "transformers.VitsTokenizer",
            "voicehub.models.vits.training.VitsReconstructionTrainingAdapter",
        ),
        # The Transformers VITS forward pass is an inference synthesizer. It
        # does not expose the source posterior/KL/duration/GAN objectives.
        # Keep the reconstruction experiment behind a specialized adapter
        # boundary instead of presenting it as turnkey VITS fine-tuning.
        native_training=False,
        separate_optimizers=False,
        support=TrainingSupport.CUSTOM,
        phases=(
            _phase(
                "waveform_reconstruction",
                component_paths=("training_model", ),
                optimizer_names=("model", ),
                forward_component="training_model",
                label_names=("audio_values", ),
                prediction_keys=("waveform", "audio_values"),
                loss_keys=("loss", "waveform_loss", "spectral_loss"),
                required_inputs=("input_ids", "audio_values"),
                kind=TrainingPhaseKind.GENERATOR,
            ), ),
        default_phase="waveform_reconstruction",
        field_schemas={
            "audio_values": {
                "sequence_dim": -1,
                "padding_value": 0.0,
                "length_field": "audio_lengths",
            },
        },
    ),
)

_RAW_AUDIO_FIELD_SCHEMAS = {
    "audio": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "audio_lengths",
    },
    "input_values": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "input_lengths",
        "mask_field": "attention_mask",
    },
    "input_features": {
        "sequence_dim": -1,
        "padding_value": 0.0,
        "length_field": "feature_lengths",
    },
}


def _transformers_asr_preset_profile(
    model_type: str,
    family: TrainingFamily,
    entrypoint: str,
) -> ModelTrainingSpec:
    """Create one locked preset around the shared native ASR trainer."""
    return _profile(
        model_type,
        family,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(entrypoint, ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    )


_BUILTIN_AUDIO_INPUT_TRAINING_SPECS = (
    _profile(
        "asr_transformers",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "transformers.AutoModelForCTC",
            "transformers.AutoModelForSpeechSeq2Seq",
            "transformers.AutoModelForRNNT",
            "transformers.AutoModelForTDT",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "speech_recognition",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
            ), ),
        default_phase="speech_recognition",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _transformers_asr_preset_profile(
        "asr_wav2vec2",
        TrainingFamily.CTC,
        "transformers.AutoModelForCTC",
    ),
    _transformers_asr_preset_profile(
        "asr_hubert",
        TrainingFamily.CTC,
        "transformers.AutoModelForCTC",
    ),
    _transformers_asr_preset_profile(
        "asr_wavlm",
        TrainingFamily.CTC,
        "transformers.AutoModelForCTC",
    ),
    _transformers_asr_preset_profile(
        "asr_moonshine",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "transformers.AutoModelForSpeechSeq2Seq",
    ),
    _transformers_asr_preset_profile(
        "asr_seamless_m4t_v2",
        TrainingFamily.SPEECH_SEQ2SEQ,
        "transformers.AutoModelForSpeechSeq2Seq",
    ),
    _profile(
        "asr_faster_whisper",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "asr_whisperx",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "asr_openai_whisper",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "asr_nemo",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        source_entrypoints=(
            "nemo.collections.asr.models.ASRModel",
            "lightning.pytorch.Trainer",
        ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "asr_speechbrain",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        source_entrypoints=("speechbrain.Brain", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "asr_funasr",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        source_entrypoints=("funasr.train", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "asr_espnet",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        source_entrypoints=("espnet2.tasks.asr.ASRTask", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "asr_wenet",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
        source_entrypoints=("wenet.bin.train", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "vad_transformers",
        TrainingFamily.AUDIO_CLASSIFICATION,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        module_paths=("model", ),
        component_paths=("model", ),
        label_names=("labels", ),
        source_entrypoints=(
            "transformers.AutoModelForAudioClassification",
            "transformers.AutoModelForAudioFrameClassification",
        ),
        native_training=True,
        support=TrainingSupport.NATIVE,
        phases=(
            _phase(
                "voice_activity_detection",
                component_paths=("model", ),
                optimizer_names=("model", ),
                label_names=("labels", ),
                prediction_keys=("logits", ),
                loss_keys=("loss", ),
                fallback_objective="classification",
            ), ),
        default_phase="voice_activity_detection",
        field_schemas=_RAW_AUDIO_FIELD_SCHEMAS,
    ),
    _profile(
        "vad_silero",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "vad_webrtc",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "vad_pyannote",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        source_entrypoints=("pyannote.audio.tasks.Segmentation", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "vad_speechbrain",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        source_entrypoints=("speechbrain.Brain", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "vad_nemo",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        source_entrypoints=(
            "nemo.collections.asr.models.EncDecClassificationModel",
            "lightning.pytorch.Trainer",
        ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "vad_funasr",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        source_entrypoints=("funasr.train", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "vad_auditok",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "vad_sherpa_onnx",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        support=TrainingSupport.INFERENCE_ONLY,
    ),
    _profile(
        "vad_pyannote_segmentation",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        source_entrypoints=("pyannote.audio.tasks.Segmentation", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
    _profile(
        "vad_pyannote_brouhaha",
        TrainingFamily.UPSTREAM_NATIVE,
        task=SpeechTask.VOICE_ACTIVITY_DETECTION,
        source_entrypoints=("brouhaha.task.RegressiveActivityDetectionTask", ),
        native_training=True,
        support=TrainingSupport.CUSTOM,
    ),
)

_BUILTIN_TRAINING_SPECS += _BUILTIN_AUDIO_INPUT_TRAINING_SPECS

_TRAINING_SPEC_REGISTRY: dict[str, ModelTrainingSpec] = {
    spec.model_type: spec
    for spec in _BUILTIN_TRAINING_SPECS
}
_TTS_TRAINING_SPEC_REGISTRY: dict[str, ModelTrainingSpec] = {
    model_type: spec
    for model_type, spec in _TRAINING_SPEC_REGISTRY.items() if spec.task is SpeechTask.TEXT_TO_SPEECH
}
_TRAINING_ALIASES: dict[str, str] = {}
_TRAINING_REGISTRY_LOCK = RLock()
# Historical public view: keep this TTS-only even as ASR and VAD profiles are
# registered. New task-aware callers can use ``ALL_MODEL_TRAINING_SPECS`` or
# ``list_training_specs(task=None)``.
MODEL_TRAINING_SPECS: Mapping[str, ModelTrainingSpec] = MappingProxyType(_TTS_TRAINING_SPEC_REGISTRY)
ALL_MODEL_TRAINING_SPECS: Mapping[str, ModelTrainingSpec] = MappingProxyType(_TRAINING_SPEC_REGISTRY)


def _normalize_training_identifier(model_type: str) -> str:
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("Training model identifiers must be non-empty strings.")
    raw = model_type.strip().lower()
    with _TRAINING_REGISTRY_LOCK:
        if raw in _TRAINING_ALIASES:
            return _TRAINING_ALIASES[raw]
    canonical = normalize_model_type(raw)
    with _TRAINING_REGISTRY_LOCK:
        return _TRAINING_ALIASES.get(canonical, canonical)


def register_training_spec(
        spec: ModelTrainingSpec,
        *,
        exist_ok: bool = False,
        aliases: Iterable[str] = (),
) -> None:
    """Register or explicitly replace a training profile.

    The read-only ``MODEL_TRAINING_SPECS`` view updates immediately,
    allowing third-party families to participate without editing
    VoiceHub's enums.
    """
    if not isinstance(spec, ModelTrainingSpec):
        raise TypeError("Training profiles must be ModelTrainingSpec instances.")
    aliases = tuple(aliases)
    if any(not isinstance(alias, str) for alias in aliases):
        raise TypeError("Training aliases must be strings.")
    normalized_aliases = [alias.strip().lower() for alias in aliases]
    if len(set(normalized_aliases)) != len(normalized_aliases):
        raise ValueError("Training aliases must not contain duplicates.")
    try:
        inference_spec = get_model_spec(spec.model_type)
    except UnknownModelError:
        inference_spec = None
    if inference_spec is not None and inference_spec.task is not spec.task:
        raise ValueError(
            f"Training profile {spec.model_type!r} declares task "
            f"{spec.task.value!r}, but its inference backend is registered "
            f"for {inference_spec.task.value!r}.")

    with _TRAINING_REGISTRY_LOCK:
        if spec.model_type in _TRAINING_ALIASES:
            target = _TRAINING_ALIASES[spec.model_type]
            raise ValueError(
                f"Training model type {spec.model_type!r} collides with an "
                f"alias for {target!r}.")
        if spec.model_type in _TRAINING_SPEC_REGISTRY and not exist_ok:
            raise ValueError(f"A training profile is already registered for "
                             f"{spec.model_type!r}.")
        validated_aliases = tuple(
            _validate_training_alias(
                alias,
                spec.model_type,
                exist_ok=exist_ok,
            ) for alias in aliases)
        _TRAINING_SPEC_REGISTRY[spec.model_type] = spec
        if spec.task is SpeechTask.TEXT_TO_SPEECH:
            _TTS_TRAINING_SPEC_REGISTRY[spec.model_type] = spec
        else:
            _TTS_TRAINING_SPEC_REGISTRY.pop(spec.model_type, None)
        for alias in validated_aliases:
            _TRAINING_ALIASES[alias] = spec.model_type


def _validate_training_alias(
    alias: str,
    canonical: str,
    *,
    exist_ok: bool,
) -> str:
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("Training aliases must be non-empty strings.")
    normalized = alias.strip().lower()
    if normalized == canonical:
        raise ValueError(f"Training alias {alias!r} is identical to its canonical model "
                         "type.")
    inference_target = normalize_model_type(normalized)
    if normalized in _TRAINING_SPEC_REGISTRY:
        raise ValueError(f"Training alias {alias!r} collides with a registered model type.")
    try:
        inference_spec = get_model_spec(normalized)
    except UnknownModelError:
        inference_spec = None
    if inference_spec is not None and inference_spec.model_type == normalized:
        raise ValueError(f"Training alias {alias!r} collides with a registered inference "
                         "model type.")
    if inference_target != normalized:
        existing_target = _TRAINING_ALIASES.get(normalized, inference_target)
        if existing_target != canonical or not exist_ok:
            raise ValueError(
                f"Training alias {alias!r} collides with an inference alias "
                f"for {inference_target!r}.")
    existing = _TRAINING_ALIASES.get(normalized)
    if existing is not None and (existing != canonical or not exist_ok):
        raise ValueError(f"Training alias {alias!r} is already registered for {existing!r}.")
    return normalized


def register_training_alias(
    alias: str,
    model_type: str,
    *,
    exist_ok: bool = False,
) -> None:
    """Register a training-only alias for a canonical profile."""
    with _TRAINING_REGISTRY_LOCK:
        canonical = _normalize_training_identifier(model_type)
        if canonical not in _TRAINING_SPEC_REGISTRY:
            raise KeyError(f"No training profile is registered for {model_type!r}.")
        normalized = _validate_training_alias(
            alias,
            canonical,
            exist_ok=exist_ok,
        )
        _TRAINING_ALIASES[normalized] = canonical


def unregister_training_alias(
    alias: str,
    *,
    missing_ok: bool = False,
) -> str | None:
    """Remove an alias and return its former canonical target."""
    if not isinstance(alias, str) or not alias.strip():
        raise ValueError("Training aliases must be non-empty strings.")
    normalized = alias.strip().lower()
    with _TRAINING_REGISTRY_LOCK:
        try:
            return _TRAINING_ALIASES.pop(normalized)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No training alias is registered for {alias!r}.") from None


def unregister_training_spec(
    model_type: str,
    *,
    missing_ok: bool = False,
) -> ModelTrainingSpec | None:
    """Remove and return a dynamically registered (or built-in) profile."""
    with _TRAINING_REGISTRY_LOCK:
        canonical = _normalize_training_identifier(model_type)
        try:
            removed = _TRAINING_SPEC_REGISTRY.pop(canonical)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No training profile is registered for {model_type!r}.") from None
        stale_aliases = [alias for alias, target in _TRAINING_ALIASES.items() if target == canonical]
        for alias in stale_aliases:
            del _TRAINING_ALIASES[alias]
        _TTS_TRAINING_SPEC_REGISTRY.pop(canonical, None)
        return removed


def get_training_spec(model_type: str) -> ModelTrainingSpec:
    """Resolve inference aliases and return one registered training profile."""
    canonical = _normalize_training_identifier(model_type)
    with _TRAINING_REGISTRY_LOCK:
        spec = _TRAINING_SPEC_REGISTRY.get(canonical)
    if spec is not None:
        return spec
    # Preserve the inference registry's informative error for known public
    # lookup paths while still permitting training-only future profiles.
    try:
        canonical = get_model_spec(model_type).model_type
    except UnknownModelError:
        raise KeyError(f"No training profile is registered for {model_type!r}.") from None
    with _TRAINING_REGISTRY_LOCK:
        try:
            return _TRAINING_SPEC_REGISTRY[canonical]
        except KeyError:
            raise KeyError(f"No training profile is registered for {model_type!r}.") from None


def list_training_specs(
    *,
    task: SpeechTask | str | None = SpeechTask.TEXT_TO_SPEECH,
    support: TrainingSupport | str | None = None,
) -> tuple[ModelTrainingSpec, ...]:
    """List profiles by task and, optionally, support boundary.

    Omitting ``task`` preserves the historical TTS-only result. Pass
    ``task=None`` to inspect every registered speech task.
    """
    resolved_task = None if task is None else SpeechTask.coerce(task)
    with _TRAINING_REGISTRY_LOCK:
        if resolved_task is None:
            specs = tuple(_TRAINING_SPEC_REGISTRY.values())
        elif resolved_task is SpeechTask.TEXT_TO_SPEECH:
            specs = tuple(_TTS_TRAINING_SPEC_REGISTRY.values())
        else:
            specs = tuple(spec for spec in _TRAINING_SPEC_REGISTRY.values() if spec.task is resolved_task)
    if support is None:
        return specs
    support = TrainingSupport.coerce(support)
    return tuple(spec for spec in specs if spec.support is support)
