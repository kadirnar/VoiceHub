"""Serialization helpers shared by configuration objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
