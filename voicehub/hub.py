"""Small Hugging Face Hub-compatible file resolution helpers."""

from __future__ import annotations

import json
import os
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any

from voicehub.hub_transport import download_hugging_face_file
from voicehub.json_utils import parse_json_object
from voicehub.path_utils import is_explicit_local_path

DEFAULT_MAX_JSON_BYTES = 64 * 1024 * 1024


def resolve_pretrained_file(
    pretrained_model_name_or_path: str | Path,
    filename: str,
    *,
    subfolder: str = "",
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local or Hub file without importing model runtimes."""
    source = Path(pretrained_model_name_or_path).expanduser()
    relative_file = Path(subfolder) / filename if subfolder else Path(filename)

    if source.is_dir():
        resolved = source / relative_file
        if not resolved.is_file():
            raise FileNotFoundError(f"Could not find {relative_file} in {source}.")
        return resolved
    if source.is_file():
        if subfolder or source.name != filename:
            raise FileNotFoundError(f"{source} is a file, but {relative_file} was requested.")
        return source
    if is_explicit_local_path(pretrained_model_name_or_path):
        raise FileNotFoundError(f"Local pretrained path was not found: {source}.")

    return download_hugging_face_file(
        repo_id=str(pretrained_model_name_or_path),
        filename=filename,
        subfolder=subfolder,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )


def read_json_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_JSON_BYTES,
) -> dict[str, Any]:
    """Read one bounded finite UTF-8 JSON object without ambiguity."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, Integral):
        raise TypeError("`max_bytes` must be an integer.")
    if max_bytes <= 0:
        raise ValueError("`max_bytes` must be greater than zero.")

    source = Path(path)
    size = source.stat().st_size
    if size > max_bytes:
        raise ValueError(f"JSON artifact {source} is {size} bytes; the configured limit is {max_bytes}.")
    with source.open("rb") as stream:
        document = stream.read(max_bytes + 1)
    if len(document) > max_bytes:
        raise ValueError(f"JSON artifact {source} exceeds the configured {max_bytes}-byte limit.")
    return parse_json_object(document, source=source)


def write_json_file(path: str | Path, value: dict[str, Any]) -> None:
    """Atomically write stable, human-readable UTF-8 JSON."""
    output_path = Path(path)
    encoded = (json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
