"""Automatic selection and extension of model-family training adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from voicehub.registry import normalize_model_type
from voicehub.training.adapters import (
    AcousticTrainingAdapter,
    BaseTrainingAdapter,
    CausalLMTrainingAdapter,
    CompositeTrainingAdapter,
    FlowMatchingTrainingAdapter,
    Seq2SeqTrainingAdapter,
    VITSTrainingAdapter,
)
from voicehub.training.recipes import BUILTIN_MODEL_ADAPTERS
from voicehub.training.specs import MODEL_TRAINING_SPECS, ModelTrainingSpec, TrainingFamily, get_training_spec

AdapterFactory = Callable[[Any, ModelTrainingSpec], BaseTrainingAdapter]


def _family_key(family: TrainingFamily | str) -> str:
    if isinstance(family, TrainingFamily):
        return family.value
    if not isinstance(family, str) or not family.strip():
        raise ValueError("Training adapter families must be non-empty strings.")
    return family.strip().lower()


def _validate_adapter_factory(factory) -> None:
    if isinstance(factory, type):
        if not issubclass(factory, BaseTrainingAdapter):
            raise TypeError("Training adapter classes must inherit BaseTrainingAdapter.")
        return
    if not callable(factory):
        raise TypeError("Training adapter factories must be adapter classes or callables.")


_FAMILY_ADAPTERS: dict[str, AdapterFactory] = {
    TrainingFamily.CAUSAL_LM.value: CausalLMTrainingAdapter,
    TrainingFamily.SEQ2SEQ.value: Seq2SeqTrainingAdapter,
    TrainingFamily.FLOW_MATCHING.value: FlowMatchingTrainingAdapter,
    TrainingFamily.ACOUSTIC.value: AcousticTrainingAdapter,
    TrainingFamily.VITS.value: VITSTrainingAdapter,
    TrainingFamily.COMPOSITE.value: CompositeTrainingAdapter,
}


class AutoTrainingAdapter:
    """Resolve the adapter paired with a VoiceHub model or future family."""

    _model_overrides: dict[str, AdapterFactory] = {}

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
                raise ValueError("AutoTrainingAdapter requires a model with "
                                 "`config.model_type`.")
            spec = get_training_spec(model_type)
        if not isinstance(spec, ModelTrainingSpec):
            raise TypeError("spec must be a ModelTrainingSpec.")

        is_specialized = (
            spec.model_type in cls._model_overrides or spec.model_type in BUILTIN_MODEL_ADAPTERS)
        factory = cls._model_overrides.get(spec.model_type)
        if factory is None:
            factory = BUILTIN_MODEL_ADAPTERS.get(spec.model_type)
        if factory is None:
            family_key = _family_key(spec.family)
            try:
                factory = _FAMILY_ADAPTERS[family_key]
            except KeyError as exc:
                raise ValueError(
                    f"No training adapter factory is registered for family "
                    f"{family_key!r}. Register one with "
                    "AutoTrainingAdapter.register_family().") from exc
        adapter = factory(model, spec)
        if not isinstance(adapter, BaseTrainingAdapter):
            raise TypeError(
                f"Adapter factory for {spec.model_type!r} returned "
                f"{type(adapter).__name__}, not BaseTrainingAdapter.")
        adapter._registered_specialization = is_specialized
        return adapter

    @classmethod
    def register(
        cls,
        model_type: str,
        adapter_class: AdapterFactory,
        *,
        exist_ok: bool = False,
    ) -> None:
        """Register a specialized class or factory for one architecture."""
        spec = get_training_spec(model_type)
        _validate_adapter_factory(adapter_class)
        if spec.model_type in cls._model_overrides and not exist_ok:
            raise ValueError(f"An adapter is already registered for {spec.model_type!r}.")
        cls._model_overrides[spec.model_type] = adapter_class

    @classmethod
    def register_model_adapter(
        cls,
        model_type: str,
        factory: AdapterFactory,
        *,
        exist_ok: bool = False,
    ) -> None:
        """Explicitly named alias for :meth:`register`."""
        cls.register(model_type, factory, exist_ok=exist_ok)

    @classmethod
    def unregister(
        cls,
        model_type: str,
        *,
        missing_ok: bool = False,
    ) -> AdapterFactory | None:
        """Remove and return a per-model adapter override."""
        try:
            canonical = get_training_spec(model_type).model_type
        except KeyError:
            canonical = normalize_model_type(model_type)
        try:
            return cls._model_overrides.pop(canonical)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No adapter override is registered for {canonical!r}.") from None

    @classmethod
    def unregister_model_adapter(
        cls,
        model_type: str,
        *,
        missing_ok: bool = False,
    ) -> AdapterFactory | None:
        return cls.unregister(model_type, missing_ok=missing_ok)

    @classmethod
    def register_family(
        cls,
        family: TrainingFamily | str,
        factory: AdapterFactory,
        *,
        exist_ok: bool = False,
    ) -> None:
        """Register an adapter factory for a built-in or future family."""
        key = _family_key(family)
        _validate_adapter_factory(factory)
        if key in _FAMILY_ADAPTERS and not exist_ok:
            raise ValueError(f"An adapter factory is already registered for family {key!r}.")
        _FAMILY_ADAPTERS[key] = factory

    @classmethod
    def unregister_family(
        cls,
        family: TrainingFamily | str,
        *,
        missing_ok: bool = False,
    ) -> AdapterFactory | None:
        """Remove and return one family adapter factory."""
        key = _family_key(family)
        try:
            return _FAMILY_ADAPTERS.pop(key)
        except KeyError:
            if missing_ok:
                return None
            raise KeyError(f"No adapter factory is registered for family {key!r}.") from None

    @classmethod
    def available_families(cls) -> tuple[str, ...]:
        return tuple(_FAMILY_ADAPTERS)

    @classmethod
    def available_models(cls) -> tuple[str, ...]:
        """Return every model type with a registered training profile."""
        return tuple(MODEL_TRAINING_SPECS)
