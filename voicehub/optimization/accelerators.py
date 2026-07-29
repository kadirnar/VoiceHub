"""Reversible accelerator-selector passes for native VoiceHub modules.

These passes configure execution policies already owned by architecture
submodules. They do not replace modules, add parameters, import Triton,
load FlashAttention-4, or compile CUDA extensions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from torch import nn

from voicehub.kernels.activations import ACTIVATION_CUDA_EXTENSION_NAME
from voicehub.kernels.cuda_extensions import CUDA_EXTENSIONS
from voicehub.kernels.registry import KernelBackend
from voicehub.neural.backends.flash_attention4 import FlashAttention4Policy
from voicehub.optimization.capabilities import OptimizationCapabilities, OptimizationContext, OptimizationMode
from voicehub.optimization.passes import OptimizationCompatibilityError, OptimizationError, OptimizationPass, PassResult
from voicehub.optimization.protocols import OptimizationModuleRoot


class AcceleratorOptimizationError(OptimizationError):
    """Base failure for accelerator selector application or restoration."""


class AcceleratorStateDictError(AcceleratorOptimizationError):
    """A selector unexpectedly changed canonical checkpoint keys."""


class AcceleratorRestorationError(AcceleratorOptimizationError):
    """One or more module selectors could not be restored."""


@dataclass(frozen=True, slots=True)
class _SelectorTarget:
    module: nn.Module
    label: str
    setter_name: str
    state_attribute: str


@dataclass(frozen=True, slots=True)
class _SelectorPatch:
    target: _SelectorTarget
    previous: Any


_SELECTOR_CAPABILITIES = OptimizationCapabilities(
    modes=(OptimizationMode.INFERENCE, OptimizationMode.TRAINING),
    devices=("cpu", "cuda", "mps"),
    dtypes=("float32", "float16", "bfloat16"),
    streaming_safe=True,
    distributed_safe=True,
    persistent=True,
    reversible=True,
    changes_parameter_names=False,
    changes_topology=False,
)


def _adapter_component_roots(model: Any) -> tuple[tuple[str, nn.Module], ...]:
    candidates: list[tuple[str, Any]] = []
    primary = getattr(model, "primary_model", None)
    if primary is not None:
        candidates.append(("primary_model", primary))
    components = getattr(model, "_components", ())
    if isinstance(components, (tuple, list)):
        for entry in components:
            if (isinstance(entry, (tuple, list)) and len(entry) == 2 and isinstance(entry[0], str) and
                    entry[0]):
                candidates.append((f"component:{entry[0]}", entry[1]))
    return tuple((label, candidate) for label, candidate in candidates if isinstance(candidate, nn.Module))


def _module_roots(model: Any) -> tuple[tuple[str, nn.Module], ...]:
    provider = getattr(model, "optimization_module_roots", None)
    if callable(provider):
        declared = provider()
        entries = (tuple(declared.items()) if isinstance(declared, Mapping) else tuple(declared))
        roots = []
        labels = set()
        modules = set()
        for entry in entries:
            if isinstance(entry, OptimizationModuleRoot):
                label = entry.label
                module = entry.module
            elif (not isinstance(entry, (tuple, list)) or len(entry) != 2):
                raise TypeError("optimization_module_roots() entries must be "
                                "(label, module) pairs.")
            else:
                label, module = entry
            if not isinstance(label, str) or not label:
                raise ValueError("Optimization module-root labels must be non-empty "
                                 "strings.")
            if not isinstance(module, nn.Module):
                raise TypeError(f"Optimization module root {label!r} must be an "
                                "nn.Module.")
            if label in labels or id(module) in modules:
                raise ValueError("Optimization module roots cannot contain duplicate "
                                 "labels or modules.")
            labels.add(label)
            modules.add(id(module))
            roots.append((label, module))
        return tuple(roots)
    if isinstance(model, nn.Module):
        return (("model", model), )
    return _adapter_component_roots(model)


def _selector_targets(
    model: Any,
    *,
    setter_name: str,
    state_attribute: str,
) -> tuple[_SelectorTarget, ...]:
    targets = []
    seen: set[int] = set()
    for root_label, root in _module_roots(model):
        for path, module in root.named_modules():
            identity = id(module)
            if identity in seen:
                continue
            seen.add(identity)
            setter = getattr(module, setter_name, None)
            if not callable(setter):
                continue
            label = root_label if not path else f"{root_label}.{path}"
            targets.append(
                _SelectorTarget(
                    module=module,
                    label=label,
                    setter_name=setter_name,
                    state_attribute=state_attribute,
                ))
    return tuple(targets)


def _state_dict_key_tree(value: Any, *, path: str) -> tuple[Any, ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    output = []
    for key, item in value.items():
        nested = (_state_dict_key_tree(item, path=f"{path}[{key!r}]") if isinstance(item, Mapping) else None)
        output.append((key, nested))
    return tuple(output)


def _state_dict_keys(model: Any) -> tuple[Any, ...]:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError(f"{type(model).__name__} has no checkpoint state_dict().")
    return _state_dict_key_tree(
        state_dict(),
        path=f"{type(model).__name__}.state_dict()",
    )


def _key_count(tree: tuple[Any, ...]) -> int:
    count = 0
    for _key, nested in tree:
        count += 1
        if nested is not None:
            count += _key_count(nested)
    return count


def _selector_token(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        return value
    return type(value).__name__


def _cuda_extension_loaded() -> bool:
    """Check process state without loading or compiling the extension."""
    return CUDA_EXTENSIONS.is_loaded(ACTIVATION_CUDA_EXTENSION_NAME)


class _SelectorPass(OptimizationPass):
    capabilities = _SELECTOR_CAPABILITIES
    setter_name: str
    state_attribute: str

    @property
    def selection(self) -> Any:
        raise NotImplementedError

    @property
    def selection_token(self) -> str:
        return _selector_token(self.selection)

    def _coerce_previous(self, value: Any) -> Any:
        raise NotImplementedError

    def _selection_issues(self, context: OptimizationContext) -> tuple[str, ...]:
        del context
        return ()

    def _extra_metadata(self) -> Mapping[str, Any]:
        return {}

    def _targets_or_error(
        self,
        model: Any,
        *,
        compatibility: bool,
    ) -> tuple[_SelectorTarget, ...]:
        targets = _selector_targets(
            model,
            setter_name=self.setter_name,
            state_attribute=self.state_attribute,
        )
        issue = None
        if not targets:
            issue = (f"{type(model).__name__} has no submodule exposing "
                     f"{self.setter_name}()")
        else:
            missing = tuple(
                target.label for target in targets if not hasattr(target.module, target.state_attribute))
            if missing:
                issue = (
                    f"reversible selector state {self.state_attribute!r} is "
                    f"missing from targets {missing!r}")
        if issue is not None:
            if compatibility:
                raise OptimizationCompatibilityError(
                    f"Optimization pass {self.qualified_id!r} is incompatible: {issue}.")
            raise AcceleratorOptimizationError(
                f"Optimization pass {self.qualified_id!r} cannot be applied: {issue}.")

        invalid = []
        for target in targets:
            previous = getattr(target.module, target.state_attribute)
            try:
                self._coerce_previous(previous)
            except (TypeError, ValueError):
                invalid.append(target.label)
        if invalid:
            issue = (f"targets {tuple(invalid)!r} expose invalid "
                     f"{self.state_attribute!r} values")
            if compatibility:
                raise OptimizationCompatibilityError(
                    f"Optimization pass {self.qualified_id!r} is incompatible: {issue}.")
            raise AcceleratorOptimizationError(
                f"Optimization pass {self.qualified_id!r} cannot be applied: {issue}.")
        return targets

    def validate(self, model: Any, context: OptimizationContext) -> None:
        super().validate(model, context)
        self._targets_or_error(model, compatibility=True)
        if not callable(getattr(model, "state_dict", None)):
            raise OptimizationCompatibilityError(
                f"Optimization pass {self.qualified_id!r} requires "
                f"{type(model).__name__}.state_dict() to verify checkpoint keys.")
        issues = self._selection_issues(context)
        if issues:
            raise OptimizationCompatibilityError(
                f"Optimization pass {self.qualified_id!r} is incompatible: "
                f"{'; '.join(issues)}.")

    @staticmethod
    def _restore_patches(patches: tuple[_SelectorPatch, ...]) -> None:
        errors = []
        for patch in patches:
            try:
                setter = getattr(
                    patch.target.module,
                    patch.target.setter_name,
                )
                setter(patch.previous)
            except BaseException as error:
                errors.append((patch.target.label, error))
        if errors:
            details = "; ".join(f"{label}: {type(error).__name__}: {error}" for label, error in errors)
            raise AcceleratorRestorationError(f"Could not restore accelerator selectors ({details}).")

    def apply(self, model: Any, context: OptimizationContext) -> PassResult:
        targets = self._targets_or_error(model, compatibility=False)
        issues = self._selection_issues(context)
        if issues:
            raise AcceleratorOptimizationError(
                f"Optimization pass {self.qualified_id!r} cannot be applied: "
                f"{'; '.join(issues)}.")
        before_keys = _state_dict_keys(model)
        patches = tuple(
            _SelectorPatch(
                target=target,
                previous=getattr(target.module, target.state_attribute),
            ) for target in targets)
        try:
            for target in targets:
                setter = getattr(target.module, target.setter_name)
                setter(self.selection)
            after_keys = _state_dict_keys(model)
        except BaseException as error:
            try:
                self._restore_patches(patches)
            except BaseException as restore_error:
                raise AcceleratorRestorationError(
                    "Accelerator selector application failed and its "
                    f"rollback also failed: {restore_error}") from error
            raise
        if before_keys != after_keys:
            self._restore_patches(patches)
            restored_keys = _state_dict_keys(model)
            suffix = (
                ""
                if restored_keys == before_keys else " Restoration also failed to recover the original keys.")
            raise AcceleratorStateDictError(
                "Accelerator selector application changed canonical "
                f"state_dict keys.{suffix}")

        previous = sorted({_selector_token(patch.previous) for patch in patches})
        metadata = {
            "outcome":
            "configured",
            "selection":
            self.selection_token,
            "target_count":
            len(targets),
            "changed_target_count":
            sum(_selector_token(patch.previous) != self.selection_token for patch in patches),
            "targets": [target.label for target in targets],
            "previous_selections":
            previous,
            "state_dict_key_count":
            _key_count(before_keys),
            "state_dict_safe":
            True,
            **dict(self._extra_metadata()),
        }
        return PassResult(
            model=model,
            state={
                "model": model,
                "patches": patches,
                "state_dict_keys": before_keys,
            },
            metadata=metadata,
        )

    def restore(
        self,
        model: Any,
        state: Mapping[str, Any],
        context: OptimizationContext,
    ) -> Any:
        del context
        patches = state.get("patches")
        if not isinstance(patches, tuple) or any(not isinstance(patch, _SelectorPatch) for patch in patches):
            raise AcceleratorRestorationError("Accelerator selector restoration state is invalid.")
        eager_model = state.get("model", model)
        expected_keys = state.get("state_dict_keys")
        if not isinstance(expected_keys, tuple):
            raise AcceleratorRestorationError("Accelerator selector restoration is missing state_dict keys.")
        self._restore_patches(patches)
        if _state_dict_keys(eager_model) != expected_keys:
            raise AcceleratorStateDictError(
                "Restoring accelerator selectors changed canonical state_dict keys.")
        return eager_model


class FlashAttention4Pass(_SelectorPass):
    """Configure every native FlashAttention-4-aware attention module."""

    pass_id = "flash-attention-4"
    pass_version = "1"
    optimization_kind = "attention-backend"
    setter_name = "set_flash_attention4_policy"
    state_attribute = "flash_attention4_policy"

    def __init__(
        self,
        *,
        policy: FlashAttention4Policy | str = FlashAttention4Policy.AUTO,
    ) -> None:
        self.policy = FlashAttention4Policy.coerce(policy)

    @property
    def selection(self) -> FlashAttention4Policy:
        return self.policy

    def _coerce_previous(self, value: Any) -> FlashAttention4Policy:
        return FlashAttention4Policy.coerce(value)

    def _selection_issues(self, context: OptimizationContext) -> tuple[str, ...]:
        if self.policy is not FlashAttention4Policy.REQUIRED:
            return ()
        issues = []
        if context.device.partition(":")[0] != "cuda":
            issues.append("required FlashAttention-4 needs a CUDA context")
        if context.dtype not in {"float16", "bfloat16"}:
            issues.append("required FlashAttention-4 needs float16 or bfloat16")
        return tuple(issues)

    def manifest_configuration(self) -> Mapping[str, Any]:
        return {"policy": self.policy.value}


class CustomKernelPass(_SelectorPass):
    """Configure every native module exposing VoiceHub custom kernels."""

    pass_id = "custom-kernels"
    pass_version = "1"
    optimization_kind = "custom-kernels"
    setter_name = "set_kernel_backend"
    state_attribute = "kernel_backend"

    def __init__(
        self,
        *,
        backend: KernelBackend | str = KernelBackend.AUTO,
    ) -> None:
        self.backend = KernelBackend.coerce(backend)

    @property
    def selection(self) -> KernelBackend:
        return self.backend

    def _coerce_previous(self, value: Any) -> KernelBackend:
        return KernelBackend.coerce(value)

    def _selection_issues(self, context: OptimizationContext) -> tuple[str, ...]:
        issues = []
        device_family = context.device.partition(":")[0]
        if (self.backend in {
                KernelBackend.TRITON,
                KernelBackend.CUDA_EXTENSION,
        } and device_family != "cuda"):
            issues.append(f"{self.backend.value} custom kernels need a CUDA context")
        if self.backend is KernelBackend.CUDA_EXTENSION:
            if not _cuda_extension_loaded():
                issues.append(
                    f"CUDA extension {ACTIVATION_CUDA_EXTENSION_NAME!r} is not already loaded; "
                    "load it explicitly before applying this pass")
        return tuple(issues)

    def _extra_metadata(self) -> Mapping[str, Any]:
        return {
            "cuda_extension": ACTIVATION_CUDA_EXTENSION_NAME,
            "cuda_extension_loaded": _cuda_extension_loaded(),
        }

    def manifest_configuration(self) -> Mapping[str, Any]:
        return {"backend": self.backend.value}


__all__ = [
    "AcceleratorOptimizationError",
    "AcceleratorRestorationError",
    "AcceleratorStateDictError",
    "CustomKernelPass",
    "FlashAttention4Pass",
]
