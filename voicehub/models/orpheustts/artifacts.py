"""Coherent native artifact resolution for Orpheus and its SNAC codec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

ORPHEUS_REFERENCE_REVISION = "4206a56e5a68cf6cf96900a8a78acd3370c02eb6"
SNAC_SAFE_REVISION = "c29a77c025506947a7ff15a678787b66b4c2ff47"
_CONFIG_NAME = "config.json"
_TOKENIZER_NAME = "tokenizer.json"
_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_SINGLE_CHECKPOINT_NAME = "model.safetensors"
_SHARDED_CHECKPOINT_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class OrpheusArtifacts:
    """Immutable language-model and tokenizer files for one runtime."""

    source: str
    revision: str | None
    config: Path
    tokenizer: Path
    checkpoint: Path
    tokenizer_config: Path | None = None

    @property
    def root(self) -> Path:
        return self.config.parent

    @property
    def is_sharded(self) -> bool:
        return self.checkpoint.name.endswith(".safetensors.index.json")


@dataclass(frozen=True, slots=True)
class SNACArtifacts:
    """Immutable configuration and safe checkpoint for the SNAC graph."""

    source: str
    revision: str | None
    config: Path
    checkpoint: Path


def _safe_filename(value: object, *, name: str, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        suffix = " or None" if optional else ""
        raise ValueError(f"`{name}` must be a non-empty filename{suffix}.")
    normalized = value.strip()
    path = PurePosixPath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
        raise ValueError(f"`{name}` must be one safe artifact-root filename.")
    return normalized


def _required_local(root: Path, filename: str, *, owner: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native {owner} requires {filename!r} in {root}.")
    return path


def _optional_local(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _optional_remote(
    repo_id: str,
    filename: str,
    *,
    cache_dir: str | None,
    revision: str,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            repo_id,
            filename,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _safe_shard_names(index_path: Path, *, owner: str) -> tuple[str, ...]:
    document = read_json_file(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{owner} Safetensors index must contain a non-empty `weight_map`.")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"{owner} index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError(f"{owner} index contains an invalid shard name.")
        path = PurePosixPath(shard_name)
        if (path.is_absolute() or len(path.parts) != 1 or ".." in path.parts or
                not shard_name.endswith(".safetensors")):
            raise ValueError(f"Unsafe {owner} checkpoint shard {shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _validate_safe_checkpoint(path: Path, *, owner: str) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError(f"Native {owner} accepts Safetensors checkpoints only.")


def _resolve_checkpoint_local(
    root: Path,
    *,
    checkpoint_filename: str | None,
    owner: str,
) -> Path:
    checkpoint = (
        _required_local(root, checkpoint_filename, owner=owner)
        if checkpoint_filename is not None else _optional_local(root, _SINGLE_CHECKPOINT_NAME))
    if checkpoint is None:
        checkpoint = _required_local(
            root,
            _SHARDED_CHECKPOINT_NAME,
            owner=owner,
        )
    _validate_safe_checkpoint(checkpoint, owner=owner)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard_name in _safe_shard_names(checkpoint, owner=owner):
            _required_local(root, shard_name, owner=owner)
    return checkpoint


def _resolve_checkpoint_remote(
    repo_id: str,
    *,
    checkpoint_filename: str | None,
    cache_dir: str | None,
    revision: str,
    token: str | bool | None,
    local_files_only: bool,
    owner: str,
) -> Path:
    candidates = ((checkpoint_filename, ) if checkpoint_filename is not None else
                  (_SINGLE_CHECKPOINT_NAME, _SHARDED_CHECKPOINT_NAME))
    checkpoint = None
    for candidate in candidates:
        checkpoint = _optional_remote(
            repo_id,
            candidate,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            local_files_only=local_files_only,
        )
        if checkpoint is not None:
            break
    if checkpoint is None:
        choices = ", ".join(repr(name) for name in candidates)
        raise FileNotFoundError(f"Native {owner} could not find a checkpoint among: {choices}.")
    _validate_safe_checkpoint(checkpoint, owner=owner)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard_name in _safe_shard_names(checkpoint, owner=owner):
            resolve_pretrained_file(
                repo_id,
                shard_name,
                cache_dir=cache_dir,
                revision=revision,
                token=token,
                local_files_only=local_files_only,
            )
    return checkpoint


def resolve_orpheus_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    tokenizer_filename: str = _TOKENIZER_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> OrpheusArtifacts:
    """Resolve a local directory/file or one immutable Hub snapshot."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Orpheus `source` must be a non-empty path or Hub ID.")
    checkpoint_filename = _safe_filename(
        checkpoint_filename,
        name="checkpoint_filename",
        optional=True,
    )
    tokenizer_filename = _safe_filename(
        tokenizer_filename,
        name="tokenizer_filename",
        optional=False,
    )
    source_path = Path(source).expanduser()
    if source_path.exists():
        if source_path.is_file():
            _validate_safe_checkpoint(source_path, owner="Orpheus")
            checkpoint = source_path.resolve()
            root = checkpoint.parent
            if checkpoint.name.endswith(".safetensors.index.json"):
                for shard_name in _safe_shard_names(
                        checkpoint,
                        owner="Orpheus",
                ):
                    _required_local(root, shard_name, owner="Orpheus")
        else:
            root = source_path.resolve()
            checkpoint = _resolve_checkpoint_local(
                root,
                checkpoint_filename=checkpoint_filename,
                owner="Orpheus",
            )
        return OrpheusArtifacts(
            source=str(source_path.resolve()),
            revision=None,
            config=_required_local(root, _CONFIG_NAME, owner="Orpheus"),
            tokenizer=_required_local(root, tokenizer_filename, owner="Orpheus"),
            tokenizer_config=_optional_local(root, _TOKENIZER_CONFIG_NAME),
            checkpoint=checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Orpheus model path was not found: {source_path}.")

    repo_id = str(source)
    requested_revision = revision or "main"
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG_NAME,
        cache_dir=cache_dir,
        revision=requested_revision,
        token=token,
        local_files_only=local_files_only,
    )
    resolved_revision = (
        get_cached_hugging_face_commit(
            repo_id,
            _CONFIG_NAME,
            cache_dir=cache_dir,
            revision=requested_revision,
        ) or requested_revision)
    tokenizer = resolve_pretrained_file(
        repo_id,
        tokenizer_filename,
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
    )
    checkpoint = _resolve_checkpoint_remote(
        repo_id,
        checkpoint_filename=checkpoint_filename,
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
        owner="Orpheus",
    )
    return OrpheusArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        tokenizer=tokenizer,
        tokenizer_config=_optional_remote(
            repo_id,
            _TOKENIZER_CONFIG_NAME,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        ),
        checkpoint=checkpoint,
    )


def resolve_snac_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str = _SINGLE_CHECKPOINT_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> SNACArtifacts:
    """Resolve SNAC config and Safetensors without a Hub SDK."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("SNAC `source` must be a non-empty path or Hub ID.")
    checkpoint_filename = _safe_filename(
        checkpoint_filename,
        name="checkpoint_filename",
        optional=False,
    )
    source_path = Path(source).expanduser()
    if source_path.exists():
        root = source_path.resolve()
        if not root.is_dir():
            raise ValueError("A local SNAC source must be a checkpoint directory.")
        checkpoint = _required_local(root, checkpoint_filename, owner="SNAC")
        _validate_safe_checkpoint(checkpoint, owner="SNAC")
        return SNACArtifacts(
            source=str(root),
            revision=None,
            config=_required_local(root, _CONFIG_NAME, owner="SNAC"),
            checkpoint=checkpoint,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"SNAC model path was not found: {source_path}.")

    repo_id = str(source)
    requested_revision = revision or "main"
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG_NAME,
        cache_dir=cache_dir,
        revision=requested_revision,
        token=token,
        local_files_only=local_files_only,
    )
    resolved_revision = (
        get_cached_hugging_face_commit(
            repo_id,
            _CONFIG_NAME,
            cache_dir=cache_dir,
            revision=requested_revision,
        ) or requested_revision)
    checkpoint = resolve_pretrained_file(
        repo_id,
        checkpoint_filename,
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
    )
    _validate_safe_checkpoint(checkpoint, owner="SNAC")
    return SNACArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        checkpoint=checkpoint,
    )


__all__ = [
    "ORPHEUS_REFERENCE_REVISION",
    "SNAC_SAFE_REVISION",
    "OrpheusArtifacts",
    "SNACArtifacts",
    "resolve_orpheus_artifacts",
    "resolve_snac_artifacts",
]
