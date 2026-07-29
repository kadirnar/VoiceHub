"""Strict, dependency-free configuration loading for native Vocos.

Released Vocos repositories use a tiny YAML mapping.  Pulling in a general
YAML object loader for three component declarations is unnecessary and makes
configuration loading harder to audit.  This module accepts only the scalar,
inline-list, and nested-mapping subset used by those repositories.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INTEGER = re.compile(r"^[+-]?[0-9]+$")
_FLOAT = re.compile(
    r"^[+-]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+|[0-9]+[eE][+-]?[0-9]+"
    r"|[0-9]+\.[0-9]*[eE][+-]?[0-9]+|[0-9]*\.[0-9]+[eE][+-]?[0-9]+)$"
)
_FORBIDDEN_YAML_TOKENS = ("&", "*", "!", "<<", "%YAML", "---", "...")


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    bracket_depth = 0
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth -= 1
        elif character == "#" and bracket_depth == 0:
            return line[:index]
    if quote is not None or bracket_depth != 0:
        raise ValueError("Vocos configuration contains an unterminated value.")
    return line


def _split_inline_list(value: str) -> tuple[str, ...]:
    inner = value[1:-1].strip()
    if not inner:
        return ()
    items: list[str] = []
    start = 0
    quote: str | None = None
    escaped = False
    for index, character in enumerate(inner):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in "[]{}":
            raise ValueError("Nested YAML collections are not supported.")
        elif character == ",":
            item = inner[start:index].strip()
            if not item:
                raise ValueError("Vocos configuration contains an empty list item.")
            items.append(item)
            start = index + 1
    item = inner[start:].strip()
    if not item:
        raise ValueError("Vocos configuration contains an empty list item.")
    items.append(item)
    return tuple(items)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        raise ValueError("Vocos configuration contains an empty scalar.")
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("["):
        if not value.endswith("]"):
            raise ValueError("Vocos inline list is not closed.")
        return [_parse_scalar(item) for item in _split_inline_list(value)]
    if value[0] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError("Vocos configuration contains an invalid string.") from error
        if not isinstance(parsed, str):
            raise ValueError("Quoted Vocos values must be strings.")
        return parsed
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        return float(value)
    if any(token in value for token in _FORBIDDEN_YAML_TOKENS):
        raise ValueError("Vocos configuration uses a forbidden YAML feature.")
    return value


def parse_vocos_yaml(text: str) -> dict[str, Any]:
    """Parse the reviewed Vocos YAML subset into plain Python values."""
    if not isinstance(text, str):
        raise TypeError("Vocos configuration text must be a string.")
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    saw_value = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if "\t" in raw_line:
            raise ValueError(
                f"Vocos configuration line {line_number} contains a tab."
            )
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip(" ")
        indentation = len(line) - len(stripped)
        if any(
            stripped == token or stripped.startswith(f"{token} ")
            for token in _FORBIDDEN_YAML_TOKENS
        ):
            raise ValueError(
                f"Vocos configuration line {line_number} uses a forbidden "
                "YAML feature."
            )
        if ":" not in stripped:
            raise ValueError(
                f"Vocos configuration line {line_number} is not a mapping entry."
            )
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        if not _KEY.fullmatch(key):
            raise ValueError(
                f"Vocos configuration line {line_number} has an invalid key."
            )
        while stack[-1][0] >= indentation:
            stack.pop()
        parent = stack[-1][1]
        if key in parent:
            raise ValueError(
                f"Vocos configuration line {line_number} repeats key {key!r}."
            )
        raw_value = raw_value.strip()
        if raw_value:
            parent[key] = _parse_scalar(raw_value)
            saw_value = True
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indentation, child))

    if not saw_value:
        raise ValueError("Vocos configuration is empty.")
    return root


def load_vocos_config(path: str | Path) -> dict[str, Any]:
    """Read one bounded UTF-8 configuration file."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Vocos configuration was not found: {source}.")
    if source.stat().st_size > 256 * 1024:
        raise ValueError("Vocos configuration exceeds the 256 KiB safety limit.")
    return parse_vocos_yaml(source.read_text(encoding="utf-8"))


def require_component_config(
    config: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    """Validate one ``class_path``/``init_args`` component declaration."""
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Vocos configuration requires a {name!r} mapping.")
    extra = set(value) - {"class_path", "init_args"}
    if extra:
        raise ValueError(
            f"Vocos {name!r} configuration has unknown keys: "
            + ", ".join(sorted(str(key) for key in extra))
        )
    class_path = value.get("class_path")
    if not isinstance(class_path, str) or not class_path:
        raise ValueError(f"Vocos {name!r} requires a non-empty `class_path`.")
    init_args = value.get("init_args", {})
    if not isinstance(init_args, Mapping):
        raise ValueError(f"Vocos {name!r} `init_args` must be a mapping.")
    return {
        "class_path": class_path,
        "init_args": dict(init_args),
    }


__all__ = [
    "load_vocos_config",
    "parse_vocos_yaml",
    "require_component_config",
]
