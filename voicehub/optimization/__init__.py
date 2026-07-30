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
_CODEC_EXPORTS = frozenset({
    "CodecCUDAGraphCaptureError",
    "CodecCUDAGraphRunner",
    "CodecCompileComponent",
    "CodecCompilePolicy",
    "CodecCompileTargetKind",
    "CodecKernelBackend",
    "CodecKernelPass",
    "CodecNumericalPolicy",
    "CodecOptimizationCompatibilityError",
    "CodecOptimizationConfig",
    "CodecOptimizationDecision",
    "CodecOptimizationError",
    "CodecOptimizationFidelity",
    "CodecOptimizationPlan",
    "CodecOptimizationPolicy",
    "CodecOptimizationResult",
    "FixedShapeCodecCUDAGraph",
    "capture_codec_cuda_graph",
    "codec_component_view",
    "coerce_codec_optimization_config",
    "discover_codec_compile_targets",
    "optimize_codec",
    "resolve_codec_optimization",
})
_DIFFUSION_EXPORTS = frozenset({
    "DIFFUSION_FAMILY_FEATURE",
    "DIFFUSION_KIND_FEATURE_PREFIX",
    "DIFFUSION_OPERATION_FEATURE_PREFIX",
    "DIFFUSION_SAMPLING_FEATURE_PREFIX",
    "DiffusionArchitectureKind",
    "DiffusionModelOptimizationSupport",
    "DiffusionOperation",
    "diffusion_kind_feature",
    "diffusion_operation_feature",
    "get_diffusion_model_optimization_support",
    "list_diffusion_model_optimization_support",
})
_DIFFUSION_CACHE_EXPORTS = frozenset({
    "DiffusionBlockResidualCache",
    "DiffusionCacheCompatibilityError",
    "DiffusionCacheConfig",
    "DiffusionCacheError",
    "DiffusionCacheMethod",
    "DiffusionCacheMixin",
    "DiffusionCachePass",
    "DiffusionCachePolicy",
    "DiffusionCachePredictor",
    "coerce_diffusion_cache_config",
})
_DIFFUSION_SAMPLING_EXPORTS = frozenset({
    "DiffusionGuidanceStrategy",
    "DiffusionPredictionCacheMethod",
    "DiffusionSamplingCompatibilityError",
    "DiffusionSamplingConfig",
    "DiffusionSamplingController",
    "DiffusionSamplingError",
    "DiffusionSamplingMixin",
    "DiffusionSamplingPass",
    "DiffusionSamplingPolicy",
    "DiffusionScheduleStrategy",
    "DiffusionSolverStrategy",
    "DiffusionStepContext",
    "coerce_diffusion_sampling_config",
})
_DIFFUSION_SOLVER_EXPORTS = frozenset({
    "STORK2FlowSolver",
    "STORKFlowConfig",
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


def _create_codec_kernel_pass():
    module = import_module("voicehub.optimization.codec_accelerators")
    return module.CodecKernelPass()


def _create_torch_compile_pass():
    module = import_module("voicehub.optimization.torch_compile")
    return module.TorchCompilePass()


def _create_diffusion_cache_pass():
    module = import_module("voicehub.optimization.diffusion_cache")
    return module.DiffusionCachePass()


def _create_diffusion_sampling_pass():
    module = import_module("voicehub.optimization.diffusion_sampling")
    return module.DiffusionSamplingPass()


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
OPTIMIZATION_PASSES.register(
    "codec-kernels",
    _create_codec_kernel_pass,
    exist_ok=True,
)
OPTIMIZATION_PASSES.register(
    "diffusion-cache",
    _create_diffusion_cache_pass,
    exist_ok=True,
)
OPTIMIZATION_PASSES.register(
    "diffusion-sampling",
    _create_diffusion_sampling_pass,
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
    if name in _CODEC_EXPORTS:
        module = import_module("voicehub.optimization.codecs")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _DIFFUSION_EXPORTS:
        module = import_module("voicehub.optimization.diffusion")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _DIFFUSION_CACHE_EXPORTS:
        module = import_module("voicehub.optimization.diffusion_cache")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _DIFFUSION_SAMPLING_EXPORTS:
        module = import_module("voicehub.optimization.diffusion_sampling")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _DIFFUSION_SOLVER_EXPORTS:
        module = import_module("voicehub.optimization.diffusion_solvers")
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
        | _CODEC_EXPORTS
        | _DIFFUSION_EXPORTS
        | _DIFFUSION_CACHE_EXPORTS
        | _DIFFUSION_SAMPLING_EXPORTS
        | _DIFFUSION_SOLVER_EXPORTS
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
    "CodecCUDAGraphCaptureError",
    "CodecCUDAGraphRunner",
    "CodecCompileComponent",
    "CodecCompilePolicy",
    "CodecCompileTargetKind",
    "CodecKernelBackend",
    "CodecKernelPass",
    "CodecNumericalPolicy",
    "CodecOptimizationCompatibilityError",
    "CodecOptimizationConfig",
    "CodecOptimizationDecision",
    "CodecOptimizationError",
    "CodecOptimizationFidelity",
    "CodecOptimizationPlan",
    "CodecOptimizationPolicy",
    "CodecOptimizationResult",
    "CustomKernelPass",
    "DIFFUSION_FAMILY_FEATURE",
    "DIFFUSION_KIND_FEATURE_PREFIX",
    "DIFFUSION_OPERATION_FEATURE_PREFIX",
    "DIFFUSION_SAMPLING_FEATURE_PREFIX",
    "DiffusionArchitectureKind",
    "DiffusionBlockResidualCache",
    "DiffusionCacheCompatibilityError",
    "DiffusionCacheConfig",
    "DiffusionCacheError",
    "DiffusionCacheMethod",
    "DiffusionCacheMixin",
    "DiffusionCachePass",
    "DiffusionCachePolicy",
    "DiffusionCachePredictor",
    "DiffusionModelOptimizationSupport",
    "DiffusionOperation",
    "DiffusionGuidanceStrategy",
    "DiffusionPredictionCacheMethod",
    "DiffusionSamplingCompatibilityError",
    "DiffusionSamplingConfig",
    "DiffusionSamplingController",
    "DiffusionSamplingError",
    "DiffusionSamplingMixin",
    "DiffusionSamplingPass",
    "DiffusionSamplingPolicy",
    "DiffusionScheduleStrategy",
    "DiffusionSolverStrategy",
    "DiffusionStepContext",
    "FlashAttention4Pass",
    "FixedShapeCodecCUDAGraph",
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
    "STORK2FlowSolver",
    "STORKFlowConfig",
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
    "capture_codec_cuda_graph",
    "codec_component_view",
    "coerce_codec_optimization_config",
    "coerce_diffusion_cache_config",
    "coerce_diffusion_sampling_config",
    "coerce_tts_optimization_config",
    "diffusion_kind_feature",
    "diffusion_operation_feature",
    "discover_codec_compile_targets",
    "get_diffusion_model_optimization_support",
    "get_tts_optimization_config",
    "get_tts_optimization_support",
    "get_vits_model_optimization_support",
    "inject_lora",
    "inspect_torch_compile",
    "list_tts_optimization_support",
    "list_diffusion_model_optimization_support",
    "list_vits_model_optimization_support",
    "normalize_optimization_dtype",
    "normalize_optimization_kind",
    "optimize_codec",
    "resolve_codec_optimization",
    "resolve_tts_optimization",
    "snapshot_optimization_pass_declaration",
    "tts_optimization_config_from_options",
    "validate_tts_optimization_config",
]
