"""Pluggable runtime strategies for TTS inference.

An inference strategy owns optional runtime transformations such as
graph compilation, quantization, accelerator placement, or an external
serving engine. Model wrappers retain responsibility for loading
checkpoints and implementing their family-specific generation contract.

Strategies are registered as zero-argument factories so optional
optimization libraries remain lazy imports.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, TypeVar

ModelT = TypeVar("ModelT")


class InferenceStrategy:
    """Lifecycle hooks for an inference optimization runtime.

    Implementations should keep :meth:`validate` side-effect free.
    VoiceHub can then reject an incompatible strategy before allocating
    model weights. :meth:`prepare` may return a wrapped or replaced
    runtime, while :meth:`restore_for_training` must return the model
    representation expected by the wrapper's training path.
    """

    name = "base"

    def validate(self, wrapper: Any) -> None:
        """Validate compatibility with a model wrapper before it is loaded."""

    def prepare(self, model: ModelT, *, wrapper: Any) -> ModelT:
        """Prepare and return a runtime for inference."""
        return model

    def restore_for_training(self, model: ModelT, *, wrapper: Any) -> ModelT:
        """Undo inference-only transformations before a training transition."""
        return model


class EagerInferenceStrategy(InferenceStrategy):
    """Default no-op strategy using each model's native eager runtime."""

    name = "eager"


InferenceStrategyFactory = Callable[[], InferenceStrategy]

_BUILTIN_STRATEGY_NAME = EagerInferenceStrategy.name
_INFERENCE_STRATEGIES: dict[str, InferenceStrategyFactory] = {
    _BUILTIN_STRATEGY_NAME: EagerInferenceStrategy,
}
_REGISTRY_LOCK = RLock()


def _normalize_strategy_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError("Inference strategy names must be strings.")
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Inference strategy names cannot be empty.")
    return normalized


def register_inference_strategy(
    name: str,
    factory: InferenceStrategyFactory | type[InferenceStrategy],
    *,
    exist_ok: bool = False,
) -> None:
    """Register a zero-argument strategy factory.

    Factories are called only when :func:`get_inference_strategy`
    resolves the strategy, allowing implementations to import optional
    runtimes lazily. ``exist_ok=True`` intentionally replaces an
    existing custom registration. The built-in ``"eager"`` strategy
    cannot be replaced.
    """
    normalized = _normalize_strategy_name(name)
    if not callable(factory):
        raise TypeError("Inference strategy factories must be callable.")
    if isinstance(factory, type) and not issubclass(factory, InferenceStrategy):
        raise TypeError("Inference strategy classes must inherit InferenceStrategy.")

    with _REGISTRY_LOCK:
        if normalized == _BUILTIN_STRATEGY_NAME:
            if exist_ok and factory is EagerInferenceStrategy:
                return
            raise ValueError("The built-in 'eager' strategy cannot be replaced.")
        if normalized in _INFERENCE_STRATEGIES and not exist_ok:
            raise ValueError(f"An inference strategy named {normalized!r} is already registered.")
        _INFERENCE_STRATEGIES[normalized] = factory


def unregister_inference_strategy(name: str) -> None:
    """Remove a custom strategy registration.

    The built-in eager strategy cannot be removed because it is
    VoiceHub's default inference runtime.
    """
    normalized = _normalize_strategy_name(name)
    if normalized == _BUILTIN_STRATEGY_NAME:
        raise ValueError("The built-in 'eager' strategy cannot be unregistered.")

    with _REGISTRY_LOCK:
        try:
            del _INFERENCE_STRATEGIES[normalized]
        except KeyError as error:
            raise KeyError(f"No inference strategy named {normalized!r} is registered.") from error


def get_inference_strategy(strategy: str | InferenceStrategy | None = None, ) -> InferenceStrategy:
    """Resolve a strategy name or validate an existing strategy instance."""
    if strategy is None:
        strategy = _BUILTIN_STRATEGY_NAME
    if isinstance(strategy, InferenceStrategy):
        return strategy
    if not isinstance(strategy, str):
        raise TypeError("`inference_strategy` must be a name or InferenceStrategy instance.")

    normalized = _normalize_strategy_name(strategy)
    with _REGISTRY_LOCK:
        try:
            factory = _INFERENCE_STRATEGIES[normalized]
        except KeyError as error:
            available = ", ".join(sorted(_INFERENCE_STRATEGIES))
            raise KeyError(
                f"Unknown inference strategy {normalized!r}. Available strategies: {available}.") from error

    resolved = factory()
    if not isinstance(resolved, InferenceStrategy):
        raise TypeError(
            f"Inference strategy factory {normalized!r} returned "
            f"{type(resolved).__name__}, not an InferenceStrategy instance.")
    return resolved


def list_inference_strategies() -> tuple[str, ...]:
    """Return registered strategy names in deterministic order."""
    with _REGISTRY_LOCK:
        return tuple(sorted(_INFERENCE_STRATEGIES))
