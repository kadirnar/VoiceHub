"""Optimization pass contracts for native inference and training graphs."""

from importlib import import_module

from voicehub.optimization.capabilities import (
    OptimizationCapabilities,
    OptimizationContext,
    OptimizationMode,
    bind_registered_architecture,
    normalize_optimization_dtype,
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
    canonical_json_string,
    canonical_json_tree,
    snapshot_optimization_pass_declaration,
)
from voicehub.optimization.protocols import (
    OptimizationCompileTarget,
    OptimizationCompileTargetProvider,
    OptimizationModuleRoot,
    OptimizationModuleRootProvider,
    OptimizationRuntimeProtocol,
)

_LORA_EXPORTS = frozenset({
    "LoRAConfig",
    "LoRAInjection",
    "LoRALinear",
    "inject_lora",
})
_ACCELERATOR_EXPORTS = frozenset({
    "AcceleratorOptimizationError",
    "AcceleratorRestorationError",
    "AcceleratorStateDictError",
    "CustomKernelPass",
    "FlashAttention4Pass",
})
_TORCH_COMPILE_EXPORTS = frozenset({
    "TorchCompileCapabilityReport",
    "TorchCompileConfig",
    "TorchCompileError",
    "TorchCompilePass",
    "TorchCompileRequirement",
    "TorchCompileRuntimeError",
    "TorchCompileUnavailableError",
    "inspect_torch_compile",
})
_TTS_EXPORTS = frozenset({
    "TTSAttentionImplementation",
    "TTSCompilePolicy",
    "TTSKernelBackend",
    "TTSOptimizationCompatibilityError",
    "TTSOptimizationConfig",
    "TTSOptimizationDecision",
    "TTSOptimizationError",
    "TTSOptimizationPlan",
    "TTSOptimizationResult",
    "TTSOptimizationSupport",
    "coerce_tts_optimization_config",
    "get_tts_optimization_config",
    "get_tts_optimization_support",
    "list_tts_optimization_support",
    "resolve_tts_optimization",
    "tts_optimization_config_from_options",
    "validate_tts_optimization_config",
})
_VITS_EXPORTS = frozenset({
    "VITSArchitectureKind",
    "VITSModelOptimizationSupport",
    "get_vits_model_optimization_support",
    "list_vits_model_optimization_support",
})


def _create_flash_attention4_pass():
    module = import_module("voicehub.optimization.accelerators")
    return module.FlashAttention4Pass()


def _create_custom_kernel_pass():
    module = import_module("voicehub.optimization.accelerators")
    return module.CustomKernelPass()


def _create_torch_compile_pass():
    module = import_module("voicehub.optimization.torch_compile")
    return module.TorchCompilePass()


OPTIMIZATION_PASSES.register(
    "compile",
    _create_torch_compile_pass,
    exist_ok=True,
)
OPTIMIZATION_PASSES.register(
    "flash-attention-4",
    _create_flash_attention4_pass,
    exist_ok=True,
)
OPTIMIZATION_PASSES.register(
    "custom-kernels",
    _create_custom_kernel_pass,
    exist_ok=True,
)


def __getattr__(name: str):
    if name in _ACCELERATOR_EXPORTS:
        module = import_module("voicehub.optimization.accelerators")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LORA_EXPORTS:
        module = import_module("voicehub.optimization.lora")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _TORCH_COMPILE_EXPORTS:
        module = import_module("voicehub.optimization.torch_compile")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _TTS_EXPORTS:
        module = import_module("voicehub.optimization.tts")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _VITS_EXPORTS:
        module = import_module("voicehub.optimization.vits")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | _ACCELERATOR_EXPORTS
        | _LORA_EXPORTS
        | _TORCH_COMPILE_EXPORTS
        | _TTS_EXPORTS
        | _VITS_EXPORTS)


__all__ = [
    "OPTIMIZATION_PASSES",
    "AcceleratorOptimizationError",
    "AcceleratorRestorationError",
    "AcceleratorStateDictError",
    "AppliedPass",
    "CustomKernelPass",
    "FlashAttention4Pass",
    "LoRAConfig",
    "LoRAInjection",
    "LoRALinear",
    "OptimizationApplicationError",
    "OptimizationCapabilities",
    "OptimizationCompatibilityError",
    "OptimizationCompileTarget",
    "OptimizationCompileTargetProvider",
    "OptimizationContext",
    "OptimizationError",
    "OptimizationMode",
    "OptimizationModuleRoot",
    "OptimizationModuleRootProvider",
    "OptimizationPass",
    "OptimizationPassManager",
    "OptimizationPassRegistry",
    "OptimizationResult",
    "OptimizationRuntimeProtocol",
    "PassResult",
    "TorchCompileCapabilityReport",
    "TorchCompileConfig",
    "TorchCompileError",
    "TorchCompilePass",
    "TorchCompileRequirement",
    "TorchCompileRuntimeError",
    "TorchCompileUnavailableError",
    "TTSAttentionImplementation",
    "TTSCompilePolicy",
    "TTSKernelBackend",
    "TTSOptimizationCompatibilityError",
    "TTSOptimizationConfig",
    "TTSOptimizationDecision",
    "TTSOptimizationError",
    "TTSOptimizationPlan",
    "TTSOptimizationResult",
    "TTSOptimizationSupport",
    "VITSArchitectureKind",
    "VITSModelOptimizationSupport",
    "bind_registered_architecture",
    "canonical_json_string",
    "canonical_json_tree",
    "coerce_tts_optimization_config",
    "get_tts_optimization_config",
    "get_tts_optimization_support",
    "get_vits_model_optimization_support",
    "inject_lora",
    "inspect_torch_compile",
    "list_tts_optimization_support",
    "list_vits_model_optimization_support",
    "normalize_optimization_dtype",
    "normalize_optimization_kind",
    "resolve_tts_optimization",
    "snapshot_optimization_pass_declaration",
    "tts_optimization_config_from_options",
    "validate_tts_optimization_config",
]
