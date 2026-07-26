"""Helpers for optional, backend-specific dependencies."""

from importlib import import_module
from types import ModuleType

from voicehub.errors import OptionalDependencyError


def import_optional(
    module_name: str,
    *,
    model_type: str,
    install_extra: str,
    setup_url: str | None = None,
) -> ModuleType:
    """Import an optional backend module with an actionable error message."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        message = (
            f"{model_type!r} requires optional dependencies that are not installed. "
            f'Install them with `pip install "voicehub[{install_extra}]"` and retry.')
        if setup_url:
            message += f" This backend also needs its official source checkout: {setup_url}"
        raise OptionalDependencyError(message) from exc
