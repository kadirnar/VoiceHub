"""Optimization pass contracts for native inference and training graphs."""

from importlib import import_module

from voicehub.optimization.capabilities import (
    OptimizationCapabilities,
    OptimizationContext,
    OptimizationMode,
    bind_registered_architecture,
    normalize_optimization_kind,
)
from voicehub.optimization.passes import (
    OPTIMIZATION_PASSES,
    AppliedPass,
    OptimizationApplicationError,
    OptimizationCompatibilityError,
    OptimizationError,
    OptimizationPass,
    OptimizationPassManager,
    OptimizationPassRegistry,
    OptimizationResult,
    PassResult,
)

_LORA_EXPORTS = frozenset({
    "LoRAConfig",
    "LoRAInjection",
    "LoRALinear",
    "inject_lora",
})


def __getattr__(name: str):
    if name in _LORA_EXPORTS:
        module = import_module("voicehub.optimization.lora")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LORA_EXPORTS)


__all__ = [
    "OPTIMIZATION_PASSES",
    "AppliedPass",
    "LoRAConfig",
    "LoRAInjection",
    "LoRALinear",
    "OptimizationApplicationError",
    "OptimizationCapabilities",
    "OptimizationCompatibilityError",
    "OptimizationContext",
    "OptimizationError",
    "OptimizationMode",
    "OptimizationPass",
    "OptimizationPassManager",
    "OptimizationPassRegistry",
    "OptimizationResult",
    "PassResult",
    "bind_registered_architecture",
    "inject_lora",
    "normalize_optimization_kind",
]
