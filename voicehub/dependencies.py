"""Helpers for optional, backend-specific dependencies."""

from importlib import import_module
from types import ModuleType
from typing import Any

from voicehub.errors import OptionalDependencyError


def normalize_import_path(path: str, *, name: str = "Import path") -> str:
    """Validate and return one canonical ``module:attribute`` path."""
    if not isinstance(path, str):
        raise TypeError(f"{name} must be a string.")
    module_name, separator, attribute = path.strip().partition(":")
    if (not separator or not module_name or not attribute or
            any(not segment.isidentifier() for segment in module_name.split(".")) or
            any(not segment.isidentifier() for segment in attribute.split("."))):
        raise ValueError(f"{name} must use the 'module:attribute' import-path form.")
    return f"{module_name}:{attribute}"


def resolve_import_path(path: str) -> Any:
    """Resolve one lazy ``module:attribute`` path.

    This helper is the auditable dynamic-import boundary for declarative
    VoiceHub registries. Callers remain responsible for validating the
    resolved object's domain-specific protocol before using it.
    """
    normalized = normalize_import_path(path)
    module_name, _, attribute = normalized.partition(":")
    target: Any = import_module(module_name)
    for segment in attribute.split("."):
        target = getattr(target, segment)
    return target


def import_optional(
    module_name: str,
    *,
    model_type: str,
    install_extra: str | None = None,
    setup_url: str | None = None,
) -> ModuleType:
    """Import a lazily loaded dependency with actionable installation guidance.

    Inference dependencies ship with VoiceHub's default installation.
    The ``training`` extra is deliberately the only user-facing runtime
    add-on. Keeping the hint optional also makes this helper suitable
    for future optimization backends without inventing a model-specific
    installation surface.
    """
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        if install_extra:
            installation = f'Install them with `pip install "voicehub[{install_extra}]"`'
        else:
            installation = "Reinstall the complete runtime with `pip install --upgrade voicehub`"
        message = (
            f"{model_type!r} requires dependencies that are not installed. "
            f"{installation} and retry.")
        if setup_url:
            message += f" This backend also needs its official source checkout: {setup_url}"
        raise OptionalDependencyError(message) from exc
