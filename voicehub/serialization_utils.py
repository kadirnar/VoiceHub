"""Serialization helpers shared by configuration objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SERIALIZED_SECRET_FIELDS = frozenset({
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "credential",
    "credentials",
    "hf_token",
    "huggingface_token",
    "password",
    "secret",
    "token",
    "use_auth_token",
})


def _secret_paths(
        value: Any,
        *,
        path: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return paths to credential-shaped fields without reading values."""
    matches: list[str] = []
    is_non_string_sequence = (isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)))
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_name = str(key)
            normalized = key_name.strip().lower().replace("-", "_")
            nested_path = (*path, key_name)
            if normalized in _SERIALIZED_SECRET_FIELDS:
                matches.append(".".join(nested_path))
                continue
            matches.extend(_secret_paths(nested, path=nested_path))
    elif is_non_string_sequence:
        for index, nested in enumerate(value):
            matches.extend(_secret_paths(
                nested,
                path=(*path, f"[{index}]"),
            ))
    return tuple(matches)


def reject_serialized_secrets(
    value: Any,
    *,
    owner: str,
) -> None:
    """Reject credentials before they can be persisted in a public artifact."""
    paths = _secret_paths(value)
    if not paths:
        return
    fields = ", ".join(paths)
    raise ValueError(
        f"{owner} cannot store runtime secrets ({fields}). Pass credentials "
        "to the model constructor or from_pretrained() call instead.")


def serialize_paths(value: Any) -> Any:
    """Recursively convert path-like configuration values to strings."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {serialize_paths(key): serialize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [serialize_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(serialize_paths(item) for item in value)
    return value
