"""Strict JSON decoding shared by untrusted VoiceHub artifact boundaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class _StrictJSONError(ValueError):
    """Internal error for JSON values outside VoiceHub's artifact contract."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJSONError(f"Duplicate JSON object key {key!r}.")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise _StrictJSONError(f"Unsupported non-finite JSON constant {value!r}.")


def _json_path_for_key(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def _validate_finite_json_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json_numbers(
                item,
                path=_json_path_for_key(path, key),
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json_numbers(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise _StrictJSONError(f"{path} contains a non-finite JSON number.")


def parse_json_value(
    document: str | bytes | bytearray,
    *,
    source: str | Path,
) -> Any:
    """Parse finite JSON without accepting ambiguous duplicate object keys."""
    try:
        value = json.loads(
            document,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        _validate_finite_json_numbers(value)
    except _StrictJSONError as error:
        raise ValueError(f"Invalid JSON artifact {source}: {error}") from error
    return value


def parse_json_object(
    document: str | bytes | bytearray,
    *,
    source: str | Path,
) -> dict[str, Any]:
    """Parse one finite JSON object without ambiguous duplicate keys."""
    value = parse_json_value(document, source=source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return value
