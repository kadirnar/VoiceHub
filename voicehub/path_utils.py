"""Path classification helpers shared by local and Hub loaders."""

from __future__ import annotations

from pathlib import Path


def is_explicit_local_path(value: str | Path) -> bool:
    """Return whether a model source unambiguously denotes a local path."""
    if isinstance(value, Path):
        return True
    raw_value = str(value)
    return (Path(raw_value).expanduser().is_absolute() or raw_value.startswith(("./", "../", "~")))


def normalize_model_source(value: str | Path) -> str:
    """Normalize an explicit local source while preserving Hub identifiers."""
    if not isinstance(value, (str, Path)):
        raise TypeError("A model source must be a string or pathlib.Path.")
    source = Path(value).expanduser()
    if not is_explicit_local_path(value):
        return str(value)
    if not source.exists():
        raise FileNotFoundError(f"Local model path was not found: {source}.")
    return str(source.resolve())
