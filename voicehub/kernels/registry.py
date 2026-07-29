"""Lazy registry and dispatch contracts for optional tensor kernels."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any, Callable


class KernelBackend(str, Enum):
    """Built-in implementation families understood by VoiceHub."""

    AUTO = "auto"
    TORCH = "torch"
    TRITON = "triton"
    CUDA_EXTENSION = "cuda_extension"

    @classmethod
    def coerce(cls, value: KernelBackend | str) -> KernelBackend:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("Kernel backends must be strings or KernelBackend values.")
        normalized = value.strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as error:
            available = ", ".join(backend.value for backend in cls)
            raise ValueError(f"Unknown kernel backend {value!r}; expected one of: {available}.") from error


@dataclass(frozen=True)
class KernelSupport:
    """Result of checking one implementation against concrete arguments."""

    available: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.available, bool):
            raise TypeError("Kernel support `available` must be a boolean.")
        if not isinstance(self.reason, str):
            raise TypeError("Kernel support `reason` must be a string.")
        if not self.available and not self.reason.strip():
            object.__setattr__(self, "reason", "implementation is unavailable")

    def __bool__(self) -> bool:
        return self.available


KernelCallable = Callable[..., Any]
KernelSupportCheck = Callable[..., KernelSupport | bool]


class KernelError(RuntimeError):
    """Base error for kernel registration or dispatch."""


class KernelRegistrationError(ValueError, KernelError):
    """A kernel declaration conflicts with the registry contract."""


class KernelDispatchError(KernelError):
    """No registered implementation supports the supplied arguments."""


def _normalize_operation(operation: str) -> str:
    if not isinstance(operation, str):
        raise TypeError("Kernel operation names must be strings.")
    normalized = operation.strip().lower()
    if not normalized:
        raise ValueError("Kernel operation names cannot be empty.")
    if any(character.isspace() for character in normalized):
        raise ValueError("Kernel operation names cannot contain whitespace.")
    return normalized


@dataclass(frozen=True)
class RegisteredKernel:
    """One implementation candidate for a logical tensor operation."""

    operation: str
    backend: KernelBackend
    implementation: KernelCallable
    priority: int = 0
    support_check: KernelSupportCheck | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _normalize_operation(self.operation))
        backend = KernelBackend.coerce(self.backend)
        if backend is KernelBackend.AUTO:
            raise KernelRegistrationError("`auto` is a dispatch policy, not a kernel backend.")
        object.__setattr__(self, "backend", backend)
        if not callable(self.implementation):
            raise TypeError("Kernel implementations must be callable.")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("Kernel priorities must be integers.")
        if self.support_check is not None and not callable(self.support_check):
            raise TypeError("Kernel support checks must be callable or None.")

    def supports(self, *args: Any, **kwargs: Any) -> KernelSupport:
        if self.support_check is None:
            return KernelSupport(True)
        result = self.support_check(*args, **kwargs)
        if isinstance(result, KernelSupport):
            return result
        if isinstance(result, bool):
            return KernelSupport(
                result,
                "" if result else "capability predicate returned false",
            )
        raise TypeError(
            f"Support check for {self.operation!r}/{self.backend.value!r} "
            "must return bool or KernelSupport.")


class KernelRegistry:
    """Thread-safe registry resolving accelerator kernels without eager
    imports.

    Each operation can have one candidate per backend. Automatic
    dispatch evaluates candidates by descending priority and uses the
    first supported implementation. An implementation failure is
    surfaced rather than silently retried with another backend, so
    numerical or programming errors are never hidden by the fallback
    path.
    """

    def __init__(self) -> None:
        self._implementations: dict[str, dict[KernelBackend, RegisteredKernel]] = {}
        self._lock = RLock()

    def register(
        self,
        operation: str,
        backend: KernelBackend | str,
        implementation: KernelCallable,
        *,
        priority: int = 0,
        support_check: KernelSupportCheck | None = None,
        replace: bool = False,
    ) -> RegisteredKernel:
        candidate = RegisteredKernel(
            operation=operation,
            backend=KernelBackend.coerce(backend),
            implementation=implementation,
            priority=priority,
            support_check=support_check,
        )
        if not isinstance(replace, bool):
            raise TypeError("`replace` must be a boolean.")
        with self._lock:
            operation_candidates = self._implementations.setdefault(
                candidate.operation,
                {},
            )
            if candidate.backend in operation_candidates and not replace:
                raise KernelRegistrationError(
                    f"Kernel {candidate.operation!r} already has a "
                    f"{candidate.backend.value!r} implementation.")
            operation_candidates[candidate.backend] = candidate
        return candidate

    def unregister(
        self,
        operation: str,
        backend: KernelBackend | str,
        *,
        missing_ok: bool = False,
    ) -> None:
        normalized_operation = _normalize_operation(operation)
        normalized_backend = KernelBackend.coerce(backend)
        if normalized_backend is KernelBackend.AUTO:
            raise ValueError("`auto` does not identify a registered implementation.")
        if not isinstance(missing_ok, bool):
            raise TypeError("`missing_ok` must be a boolean.")
        with self._lock:
            operation_candidates = self._implementations.get(normalized_operation)
            if operation_candidates is None or normalized_backend not in operation_candidates:
                if missing_ok:
                    return
                raise KeyError(
                    f"No {normalized_backend.value!r} kernel is registered for "
                    f"{normalized_operation!r}.")
            del operation_candidates[normalized_backend]
            if not operation_candidates:
                del self._implementations[normalized_operation]

    def implementations(self, operation: str) -> tuple[RegisteredKernel, ...]:
        normalized = _normalize_operation(operation)
        with self._lock:
            candidates = tuple(self._implementations.get(normalized, {}).values())
        return tuple(
            sorted(
                candidates,
                key=lambda candidate: (-candidate.priority, candidate.backend.value),
            ))

    def list_operations(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._implementations))

    def resolve(
        self,
        operation: str,
        *args: Any,
        backend: KernelBackend | str = KernelBackend.AUTO,
        **kwargs: Any,
    ) -> RegisteredKernel:
        normalized_operation = _normalize_operation(operation)
        requested_backend = KernelBackend.coerce(backend)
        candidates = self.implementations(normalized_operation)
        if requested_backend is not KernelBackend.AUTO:
            candidates = tuple(
                candidate for candidate in candidates if candidate.backend is requested_backend)

        if not candidates:
            qualifier = (
                ""
                if requested_backend is KernelBackend.AUTO else f" for backend {requested_backend.value!r}")
            raise KernelDispatchError(
                f"No kernel implementation is registered for "
                f"{normalized_operation!r}{qualifier}.")

        unavailable = []
        for candidate in candidates:
            try:
                support = candidate.supports(*args, **kwargs)
            except Exception as error:
                support = KernelSupport(
                    False,
                    f"capability check failed with {type(error).__name__}: {error}",
                )
            if support.available:
                return candidate
            unavailable.append(f"{candidate.backend.value}: {support.reason}")

        attempted = "; ".join(unavailable)
        raise KernelDispatchError(
            f"No supported kernel implementation was found for "
            f"{normalized_operation!r} ({attempted}).")

    def dispatch(
        self,
        operation: str,
        *args: Any,
        backend: KernelBackend | str = KernelBackend.AUTO,
        **kwargs: Any,
    ) -> Any:
        candidate = self.resolve(
            operation,
            *args,
            backend=backend,
            **kwargs,
        )
        return candidate.implementation(*args, **kwargs)


KERNEL_REGISTRY = KernelRegistry()


def register_kernel(
    operation: str,
    backend: KernelBackend | str,
    implementation: KernelCallable,
    *,
    priority: int = 0,
    support_check: KernelSupportCheck | None = None,
    replace: bool = False,
) -> RegisteredKernel:
    """Register an implementation in VoiceHub's process-wide registry."""
    return KERNEL_REGISTRY.register(
        operation,
        backend,
        implementation,
        priority=priority,
        support_check=support_check,
        replace=replace,
    )


def resolve_kernel(
    operation: str,
    *args: Any,
    backend: KernelBackend | str = KernelBackend.AUTO,
    **kwargs: Any,
) -> RegisteredKernel:
    """Resolve a process-wide kernel without executing it."""
    return KERNEL_REGISTRY.resolve(
        operation,
        *args,
        backend=backend,
        **kwargs,
    )


def dispatch_kernel(
    operation: str,
    *args: Any,
    backend: KernelBackend | str = KernelBackend.AUTO,
    **kwargs: Any,
) -> Any:
    """Resolve and execute a process-wide kernel implementation."""
    return KERNEL_REGISTRY.dispatch(
        operation,
        *args,
        backend=backend,
        **kwargs,
    )
