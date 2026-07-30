"""Typed runtime hooks for multi-stage speech optimization graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


def _label(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{name}` must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class OptimizationModuleRoot:
    """One runtime-owned module tree inspected by selector passes."""

    label: str
    module: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label",
            _label(self.label, name="label"),
        )
        if self.module is None:
            raise ValueError("`module` must not be None.")


@dataclass(frozen=True, slots=True)
class OptimizationCompileTarget:
    """One method boundary that synthesis or training actually invokes."""

    label: str
    owner: Any
    attribute: str
    component: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "label",
            _label(self.label, name="label"),
        )
        object.__setattr__(
            self,
            "attribute",
            _label(self.attribute, name="attribute"),
        )
        if self.component is not None:
            object.__setattr__(
                self,
                "component",
                _label(self.component, name="component"),
            )
        if self.owner is None:
            raise ValueError("`owner` must not be None.")
        if not callable(getattr(self.owner, self.attribute, None)):
            raise TypeError(f"{type(self.owner).__name__}.{self.attribute} must be "
                            "callable.")


@runtime_checkable
class OptimizationModuleRootProvider(Protocol):
    """Contract for runtimes that expose more than one module tree."""

    def optimization_module_roots(self, ) -> Iterable[OptimizationModuleRoot]:
        ...


@runtime_checkable
class OptimizationCompileTargetProvider(Protocol):
    """Contract for runtimes whose executed boundary is not plain forward."""

    def optimization_compile_targets(
        self,
        mode: str,
    ) -> Iterable[OptimizationCompileTarget]:
        """Return executed boundaries, or an empty iterable if unsupported."""
        ...


@runtime_checkable
class OptimizationRuntimeProtocol(
        OptimizationModuleRootProvider,
        OptimizationCompileTargetProvider,
        Protocol,
):
    """Complete checkpoint-aware contract for composite speech runtimes."""

    def parameters(self) -> Iterable[Any]:
        ...

    def state_dict(self) -> Mapping[str, Any]:
        ...


__all__ = [
    "OptimizationCompileTarget",
    "OptimizationCompileTargetProvider",
    "OptimizationModuleRoot",
    "OptimizationModuleRootProvider",
    "OptimizationRuntimeProtocol",
]
