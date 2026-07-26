from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REFERENCE_AUDIO_FIELDS = ("ref_audio", "reference_audio", "reference")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_jsonl_paths(spec: str | Sequence[str]) -> List[Path]:
    if isinstance(spec, (list, tuple)):
        raw_tokens = [str(item).strip() for item in spec if str(item).strip()]
    else:
        raw_tokens = [item.strip() for item in str(spec).split(",") if item.strip()]

    paths: List[Path] = []
    for token in raw_tokens:
        if any(ch in token for ch in "*?[]"):
            matches = [Path(match) for match in sorted(glob.glob(token))]
            paths.extend(match for match in matches if match.suffix == ".jsonl")
            continue

        path = Path(token)
        if path.is_dir():
            paths.extend(sorted(child for child in path.iterdir() if child.suffix == ".jsonl"))
            continue

        paths.append(path)

    deduped: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)

    if not deduped:
        raise ValueError(f"No JSONL files found for input spec: {spec}")
    return deduped


def load_jsonl_spec(spec: str | Sequence[str]) -> tuple[List[Path], List[Dict[str, Any]]]:
    paths = resolve_jsonl_paths(spec)
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return paths, records


def resolve_shard_spec(
    num_shards: Optional[int],
    shard_rank: Optional[int],
    *,
    default_num_shards: Optional[int] = None,
    default_shard_rank: Optional[int] = None,
) -> tuple[int, int]:
    world_size = int(num_shards or default_num_shards or os.environ.get("WORLD_SIZE", 1))
    rank = int(
        shard_rank
        if shard_rank is not None
        else default_shard_rank
        if default_shard_rank is not None
        else os.environ.get("RANK", 0)
    )
    if world_size < 1:
        raise ValueError(f"`num_shards` must be >= 1, got {world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"`shard_rank` must satisfy 0 <= rank < {world_size}, got {rank}.")
    return world_size, rank


def select_rank_shard(records: Sequence[Dict[str, Any]], num_shards: int, shard_rank: int) -> List[Dict[str, Any]]:
    return [record for index, record in enumerate(records) if index % num_shards == shard_rank]


def shard_output_path(path: str | Path, shard_rank: int, num_shards: int) -> Path:
    output_path = Path(path)
    suffix = "".join(output_path.suffixes) or ".jsonl"
    stem = output_path.name[: -len(suffix)] if suffix else output_path.name
    shard_name = f"{stem}.rank{shard_rank:05d}-of-{num_shards:05d}{suffix}"
    return output_path.with_name(shard_name)


def normalize_audio_path_list(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
    base_dir: Optional[Path] = None,
) -> Optional[List[Optional[str]]]:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [_resolve_path(value, base_dir)]
    if isinstance(value, list):
        resolved: List[Optional[str]] = []
        for item in value:
            if item is None:
                if not allow_none:
                    raise ValueError(f"`{field_name}` must not contain null.")
                resolved.append(None)
                continue
            if not isinstance(item, str):
                raise ValueError(f"`{field_name}` must be a string or list of strings.")
            resolved.append(_resolve_path(item, base_dir))
        return resolved
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


def resolve_record_audio_paths(record: Dict[str, Any], *, base_dir: Path) -> Dict[str, Any]:
    resolved = dict(record)
    if "audio" in resolved and isinstance(resolved["audio"], str) and resolved["audio"]:
        resolved["audio"] = _resolve_path(resolved["audio"], base_dir)

    for field_name in REFERENCE_AUDIO_FIELDS:
        if field_name not in resolved:
            continue
        allow_none = field_name == "reference"
        normalized = normalize_audio_path_list(
            resolved.get(field_name),
            field_name,
            allow_none=allow_none,
            base_dir=base_dir,
        )
        if isinstance(record.get(field_name), str) and isinstance(normalized, list):
            resolved[field_name] = normalized[0] if normalized else None
        else:
            resolved[field_name] = normalized
    return resolved


def format_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(0, int(seconds))))


def _resolve_path(path: str, base_dir: Optional[Path]) -> str:
    stripped = str(path).strip()
    if _looks_like_uri(stripped):
        return stripped
    path_obj = Path(stripped).expanduser()
    if path_obj.is_absolute() or base_dir is None:
        return str(path_obj.resolve())
    return str((base_dir / path_obj).resolve())


def _looks_like_uri(path: str) -> bool:
    return "://" in path or path.startswith("hf://")
