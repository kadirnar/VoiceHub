"""Trait-driven optimization inventory for diffusion and flow TTS graphs.

The inventory describes active public execution graphs.  It deliberately
does not infer support from model names or from vendored source trees:
architectures opt in with a family feature, one normalized kind, and the
diffusion operations that their registered runtime actually executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from voicehub.tasks import SpeechTask

DIFFUSION_FAMILY_FEATURE = "diffusion-family"
DIFFUSION_KIND_FEATURE_PREFIX = "diffusion-kind-"
DIFFUSION_OPERATION_FEATURE_PREFIX = "diffusion-operation-"


class DiffusionArchitectureKind(str, Enum):
    """The diffusion or flow formulation used by an active TTS graph."""

    CONDITIONAL_FLOW_MATCHING = "conditional-flow-matching"
    RECTIFIED_FLOW = "rectified-flow"
    FLOW_MATCHING = "flow-matching"
    STYLE_DIFFUSION = "style-diffusion"
    DENOISING_DIFFUSION = "denoising-diffusion"

    @classmethod
    def coerce(
        cls,
        value: DiffusionArchitectureKind | str,
    ) -> DiffusionArchitectureKind:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Diffusion architecture kinds must be strings.")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown diffusion architecture kind {value!r}; expected: "
                f"{choices}.") from error


class DiffusionOperation(str, Enum):
    """Optimization-relevant operations present in an active sampler."""

    DENOISER = "denoiser"
    CLASSIFIER_FREE_GUIDANCE = "classifier-free-guidance"
    EULER_SOLVER = "euler-solver"
    MIDPOINT_SOLVER = "midpoint-solver"
    ADPM2_SOLVER = "adpm2-solver"
    DPM_SOLVER_PLUS_PLUS = "dpm-solver-plus-plus"
    ITERATIVE_ESTIMATOR = "iterative-estimator"

    @classmethod
    def coerce(
        cls,
        value: DiffusionOperation | str,
    ) -> DiffusionOperation:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Diffusion operations must be strings.")
        normalized = value.strip().lower().replace("_", "-")
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown diffusion operation {value!r}; expected: "
                             f"{choices}.") from error


def diffusion_kind_feature(kind: DiffusionArchitectureKind | str, ) -> str:
    """Return the normalized feature token for one formulation."""
    return f"{DIFFUSION_KIND_FEATURE_PREFIX}{DiffusionArchitectureKind.coerce(kind).value}"


def diffusion_operation_feature(operation: DiffusionOperation | str, ) -> str:
    """Return the normalized feature token for one active operation."""
    return (f"{DIFFUSION_OPERATION_FEATURE_PREFIX}"
            f"{DiffusionOperation.coerce(operation).value}")


@dataclass(frozen=True, slots=True)
class DiffusionModelOptimizationSupport:
    """One public TTS model's declared diffusion optimization surface."""

    model_type: str
    architecture: str
    kind: DiffusionArchitectureKind
    operations: tuple[DiffusionOperation, ...]
    training: bool
    distributed_training: bool
    optimization_passes: tuple[str, ...]

    @property
    def compile_supported(self) -> bool:
        """Whether the architecture declares ``torch.compile`` support."""
        return "compile" in self.optimization_passes

    @property
    def diffusion_cache_supported(self) -> bool:
        """Whether an architecture-owned approximate block cache is exposed."""
        return "diffusion-cache" in self.optimization_passes

    def supports_optimization_pass(self, optimization_pass: str) -> bool:
        """Return whether an optimization pass is declared compatible."""
        if not isinstance(optimization_pass, str):
            raise TypeError("Optimization pass names must be strings.")
        normalized = optimization_pass.strip().lower().replace("_", "-")
        if not normalized:
            raise ValueError("Optimization pass names must be non-empty.")
        return normalized in self.optimization_passes

    def to_dict(self) -> dict[str, object]:
        return {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "kind": self.kind.value,
            "operations": [item.value for item in self.operations],
            "training": self.training,
            "distributed_training": self.distributed_training,
            "compile_supported": self.compile_supported,
            "diffusion_cache_supported": self.diffusion_cache_supported,
            "optimization_passes": list(self.optimization_passes),
        }


def _architecture_kind(architecture) -> DiffusionArchitectureKind:
    raw_kind = architecture.metadata.get("diffusion_architecture_kind")
    if not isinstance(raw_kind, str):
        raise TypeError(
            f"Diffusion-family architecture {architecture.architecture_id!r} "
            "does not declare metadata.diffusion_architecture_kind.")
    kind = DiffusionArchitectureKind.coerce(raw_kind)
    expected_feature = diffusion_kind_feature(kind)
    declared_features = tuple(
        feature for feature in architecture.capabilities.features
        if feature.startswith(DIFFUSION_KIND_FEATURE_PREFIX))
    if declared_features != (expected_feature, ):
        raise ValueError(
            f"Diffusion-family architecture {architecture.architecture_id!r} "
            "must declare exactly the kind feature "
            f"{expected_feature!r}; found {declared_features!r}.")
    return kind


def _architecture_operations(architecture) -> tuple[DiffusionOperation, ...]:
    raw_operations = architecture.metadata.get("diffusion_operations")
    if isinstance(raw_operations, str):
        raw_operations = (raw_operations, )
    else:
        try:
            raw_operations = tuple(raw_operations)
        except TypeError as error:
            raise TypeError(
                f"Diffusion-family architecture {architecture.architecture_id!r} "
                "must declare metadata.diffusion_operations as an iterable "
                "of operation names.") from error
    if not raw_operations:
        raise ValueError(
            f"Diffusion-family architecture {architecture.architecture_id!r} "
            "must declare at least one diffusion operation.")
    operations = tuple(DiffusionOperation.coerce(operation) for operation in raw_operations)
    if len(operations) != len(set(operations)):
        raise ValueError(
            f"Diffusion-family architecture {architecture.architecture_id!r} "
            "declares duplicate diffusion operations.")

    expected_features = tuple(diffusion_operation_feature(operation) for operation in operations)
    declared_features = tuple(
        feature for feature in architecture.capabilities.features
        if feature.startswith(DIFFUSION_OPERATION_FEATURE_PREFIX))
    if declared_features != expected_features:
        raise ValueError(
            f"Diffusion-family architecture {architecture.architecture_id!r} "
            "must keep metadata.diffusion_operations aligned with operation "
            f"features; expected {expected_features!r}, found "
            f"{declared_features!r}.")
    return operations


def list_diffusion_model_optimization_support() -> tuple[DiffusionModelOptimizationSupport, ...]:
    """List active public diffusion/flow TTS models by architecture traits."""
    from voicehub.architectures import get_architecture_spec
    from voicehub.registry import list_model_specs

    output = []
    for model in list_model_specs(task=SpeechTask.TEXT_TO_SPEECH):
        if model.architecture is None:
            continue
        architecture = get_architecture_spec(model.architecture)
        capabilities = architecture.capabilities
        if not capabilities.has_feature(DIFFUSION_FAMILY_FEATURE):
            continue
        output.append(
            DiffusionModelOptimizationSupport(
                model_type=model.model_type,
                architecture=architecture.architecture_id,
                kind=_architecture_kind(architecture),
                operations=_architecture_operations(architecture),
                training=capabilities.training,
                distributed_training=capabilities.distributed_training,
                optimization_passes=capabilities.optimization_passes,
            ))
    return tuple(sorted(output, key=lambda item: item.model_type))


def get_diffusion_model_optimization_support(model_type: str, ) -> DiffusionModelOptimizationSupport:
    """Return one registered diffusion/flow TTS model or fail explicitly."""
    if not isinstance(model_type, str) or not model_type.strip():
        raise ValueError("`model_type` must be a non-empty string.")
    from voicehub.registry import get_model_spec

    canonical = get_model_spec(model_type).model_type
    for support in list_diffusion_model_optimization_support():
        if support.model_type == canonical:
            return support
    raise ValueError(f"Model {canonical!r} is not a registered diffusion-family TTS model.")


__all__ = [
    "DIFFUSION_FAMILY_FEATURE",
    "DIFFUSION_KIND_FEATURE_PREFIX",
    "DIFFUSION_OPERATION_FEATURE_PREFIX",
    "DiffusionArchitectureKind",
    "DiffusionModelOptimizationSupport",
    "DiffusionOperation",
    "diffusion_kind_feature",
    "diffusion_operation_feature",
    "get_diffusion_model_optimization_support",
    "list_diffusion_model_optimization_support",
]
