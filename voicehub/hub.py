"""Small Hugging Face Hub-compatible file resolution helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from voicehub.errors import OptionalDependencyError


def resolve_pretrained_file(
    pretrained_model_name_or_path: str,
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

    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            "Loading remote checkpoints requires `huggingface-hub`. "
            "Install it with `pip install huggingface-hub`.") from exc

    resolved = hf_hub_download(
        repo_id=pretrained_model_name_or_path,
        filename=filename,
        subfolder=subfolder or None,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    return Path(resolved)


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def write_json_file(path: str | Path, value: dict[str, Any]) -> None:
    """Write stable, human-readable UTF-8 JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
