"""Automatic selection of a model-family training adapter."""

from __future__ import annotations

from voicehub.training.adapters import (
    AcousticTrainingAdapter,
    BaseTrainingAdapter,
    CausalLMTrainingAdapter,
    CompositeTrainingAdapter,
    FlowMatchingTrainingAdapter,
    Seq2SeqTrainingAdapter,
)
from voicehub.training.specs import MODEL_TRAINING_SPECS, ModelTrainingSpec, TrainingFamily, get_training_spec

_FAMILY_ADAPTERS = {
    TrainingFamily.CAUSAL_LM: CausalLMTrainingAdapter,
    TrainingFamily.SEQ2SEQ: Seq2SeqTrainingAdapter,
    TrainingFamily.FLOW_MATCHING: FlowMatchingTrainingAdapter,
    TrainingFamily.ACOUSTIC: AcousticTrainingAdapter,
    TrainingFamily.COMPOSITE: CompositeTrainingAdapter,
}


class AutoTrainingAdapter:
    """Resolve the mandatory training adapter paired with a VoiceHub model."""

    _model_overrides: dict[str, type[BaseTrainingAdapter]] = {}

    def __init__(self):
        raise OSError("AutoTrainingAdapter must be created with `from_model()`.")

    @classmethod
    def from_model(
        cls,
        model,
        *,
        spec: ModelTrainingSpec | None = None,
    ) -> BaseTrainingAdapter:
        """Create an unloaded adapter without importing PyTorch."""
        if spec is None:
            config = getattr(model, "config", None)
            model_type = getattr(config, "model_type", None)
            if not model_type:
                raise ValueError("AutoTrainingAdapter requires a model with `config.model_type`.")
            spec = get_training_spec(model_type)
        adapter_class = cls._model_overrides.get(
            spec.model_type,
            _FAMILY_ADAPTERS[spec.family],
        )
        return adapter_class(model, spec)

    @classmethod
    def register(
        cls,
        model_type: str,
        adapter_class: type[BaseTrainingAdapter],
        *,
        exist_ok: bool = False,
    ) -> None:
        """Register a specialized adapter for one architecture."""
        spec = get_training_spec(model_type)
        if not issubclass(adapter_class, BaseTrainingAdapter):
            raise TypeError("Training adapters must inherit `BaseTrainingAdapter`.")
        if spec.model_type in cls._model_overrides and not exist_ok:
            raise ValueError(f"An adapter is already registered for {spec.model_type!r}.")
        cls._model_overrides[spec.model_type] = adapter_class

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        """Return every model type with a training profile."""
        return tuple(MODEL_TRAINING_SPECS)
