"""Helpers for optional, backend-specific dependencies."""

from importlib import import_module
from types import ModuleType

from voicehub.errors import OptionalDependencyError


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
