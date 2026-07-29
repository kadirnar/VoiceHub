"""Coherent artifact resolution for VoiceHub-native VITS and MMS-TTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

_CONFIG_NAME = "config.json"
_VOCABULARY_NAME = "vocab.json"
_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_SINGLE_CHECKPOINT_NAME = "model.safetensors"
_SHARDED_CHECKPOINT_NAME = "model.safetensors.index.json"


@dataclass(frozen=True, slots=True)
class VitsArtifacts:
    """Immutable files forming one native VITS runtime."""

    source: str
    revision: str | None
    config: Path
    vocabulary: Path
    tokenizer_config: Path
    checkpoint: Path

    @property
    def is_sharded(self) -> bool:
        """Whether ``checkpoint`` is a Safetensors index."""
        return self.checkpoint.name.endswith(".safetensors.index.json")


def _required_local(root: Path, filename: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native VITS requires {filename!r} in {root}.")
    return path


def _optional_local(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path if path.is_file() else None


def _resolve_optional_remote(
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


def _safe_shard_names(index_path: Path) -> tuple[str, ...]:
    document = read_json_file(index_path)
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("VITS Safetensors index must contain a non-empty `weight_map`.")
    names: set[str] = set()
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError("VITS Safetensors index contains an invalid tensor name.")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError("VITS Safetensors index contains an invalid shard name.")
        shard_path = PurePosixPath(shard_name)
        if (shard_path.is_absolute() or len(shard_path.parts) != 1 or ".." in shard_path.parts):
            raise ValueError(f"Unsafe VITS checkpoint shard path {shard_name!r}.")
        if not shard_name.endswith(".safetensors"):
            raise ValueError(f"VITS checkpoint shard is not Safetensors: {shard_name!r}.")
        names.add(shard_name)
    return tuple(sorted(names))


def _validate_checkpoint(path: Path) -> None:
    if (path.suffix != ".safetensors" and not path.name.endswith(".safetensors.index.json")):
        raise ValueError("Native VITS accepts a Safetensors checkpoint or index.")


def _resolve_local(
    source: Path,
    *,
    checkpoint_filename: str | None,
    vocabulary_filename: str,
    tokenizer_config_filename: str,
) -> VitsArtifacts:
    checkpoint_override: Path | None = None
    if source.is_file():
        _validate_checkpoint(source)
        checkpoint_override = source
        root = source.parent
    else:
        root = source

    checkpoint = checkpoint_override
    if checkpoint is None and checkpoint_filename is not None:
        checkpoint = _required_local(root, checkpoint_filename)
    if checkpoint is None:
        checkpoint = _optional_local(root, _SINGLE_CHECKPOINT_NAME)
    if checkpoint is None:
        checkpoint = _required_local(root, _SHARDED_CHECKPOINT_NAME)
    _validate_checkpoint(checkpoint)
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard_name in _safe_shard_names(checkpoint):
            _required_local(root, shard_name)

    return VitsArtifacts(
        source=str(source),
        revision=None,
        config=_required_local(root, _CONFIG_NAME),
        vocabulary=_required_local(root, vocabulary_filename),
        tokenizer_config=_required_local(root, tokenizer_config_filename),
        checkpoint=checkpoint,
    )


def resolve_vits_artifacts(
    source: str | Path,
    *,
    checkpoint_filename: str | None = None,
    vocabulary_filename: str = _VOCABULARY_NAME,
    tokenizer_config_filename: str = _TOKENIZER_CONFIG_NAME,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> VitsArtifacts:
    """Resolve one local directory/file or immutable Hub artifact set."""
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("VITS `source` must be a non-empty path or Hub ID.")

    source_path = Path(source).expanduser()
    if source_path.exists():
        return _resolve_local(
            source_path.resolve(),
            checkpoint_filename=checkpoint_filename,
            vocabulary_filename=vocabulary_filename,
            tokenizer_config_filename=tokenizer_config_filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"VITS model path was not found: {source_path}.")

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
    pinned_revision = get_cached_hugging_face_commit(
        repo_id,
        _CONFIG_NAME,
        cache_dir=cache_dir,
        revision=requested_revision,
    )
    resolved_revision = pinned_revision or requested_revision
    vocabulary = resolve_pretrained_file(
        repo_id,
        vocabulary_filename,
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
    )
    tokenizer_config = resolve_pretrained_file(
        repo_id,
        tokenizer_config_filename,
        cache_dir=cache_dir,
        revision=resolved_revision,
        token=token,
        local_files_only=local_files_only,
    )

    candidates = ((checkpoint_filename, ) if checkpoint_filename is not None else
                  (_SINGLE_CHECKPOINT_NAME, _SHARDED_CHECKPOINT_NAME))
    checkpoint = None
    for candidate in candidates:
        checkpoint = _resolve_optional_remote(
            repo_id,
            candidate,
            cache_dir=cache_dir,
            revision=resolved_revision,
            token=token,
            local_files_only=local_files_only,
        )
        if checkpoint is not None:
            break
    if checkpoint is None:
        choices = ", ".join(repr(name) for name in candidates)
        raise FileNotFoundError(f"Native VITS could not find a checkpoint among: {choices}.")
    if checkpoint.name.endswith(".safetensors.index.json"):
        for shard_name in _safe_shard_names(checkpoint):
            resolve_pretrained_file(
                repo_id,
                shard_name,
                cache_dir=cache_dir,
                revision=resolved_revision,
                token=token,
                local_files_only=local_files_only,
            )

    return VitsArtifacts(
        source=repo_id,
        revision=resolved_revision,
        config=config,
        vocabulary=vocabulary,
        tokenizer_config=tokenizer_config,
        checkpoint=checkpoint,
    )


__all__ = ["VitsArtifacts", "resolve_vits_artifacts"]
