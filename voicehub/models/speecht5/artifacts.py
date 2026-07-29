"""Pinned, coherent artifact resolution for native SpeechT5."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.models.speecht5.metadata import (
    SPEECHT5_ASSET_SHA256,
    SPEECHT5_CHECKPOINT_FILENAME,
    SPEECHT5_CHECKPOINT_SHA256,
    SPEECHT5_CHECKPOINT_SIZE,
    SPEECHT5_HIFIGAN_CHECKPOINT_FILENAME,
    SPEECHT5_HIFIGAN_CHECKPOINT_SHA256,
    SPEECHT5_HIFIGAN_CHECKPOINT_SIZE,
    SPEECHT5_HIFIGAN_CONFIG_SHA256,
    SPEECHT5_HIFIGAN_REPOSITORY,
    SPEECHT5_HIFIGAN_REVISION,
    SPEECHT5_REPOSITORY,
    SPEECHT5_REVISION,
)
from voicehub.path_utils import is_explicit_local_path

_CONFIG = "config.json"
_PREPROCESSOR = "preprocessor_config.json"
_TOKENIZER_MODEL = "spm_char.model"
_TOKENIZER_CONFIG = "tokenizer_config.json"
_SPECIAL_TOKENS = "special_tokens_map.json"
_ADDED_TOKENS = "added_tokens.json"
_SAFE_CHECKPOINT = "model.safetensors"


@dataclass(frozen=True, slots=True)
class SpeechT5Artifacts:
    """Text-to-spectrogram checkpoint and processor from one snapshot."""

    source: str
    revision: str | None
    config: Path
    preprocessor_config: Path
    tokenizer_model: Path
    checkpoint: Path
    tokenizer_config: Path | None = None
    special_tokens_map: Path | None = None
    added_tokens: Path | None = None
    official: bool = False

    @property
    def root(self) -> Path:
        return self.config.parent


@dataclass(frozen=True, slots=True)
class SpeechT5HifiGanArtifacts:
    """HiFi-GAN config and checkpoint from one snapshot."""

    source: str
    revision: str | None
    config: Path
    checkpoint: Path
    official: bool = False

    @property
    def root(self) -> Path:
        return self.config.parent


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str,
    expected_size: int | None = None,
) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise OSError(f"{label} has size {path.stat().st_size}; expected "
                      f"{expected_size} bytes.")
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise OSError(f"{label} has SHA-256 {actual}; expected {expected_sha256}.")


def _required(root: Path, filename: str, *, owner: str) -> Path:
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Native {owner} requires {filename!r} in {root}.")
    return path.resolve()


def _optional(root: Path, filename: str) -> Path | None:
    path = root / filename
    return path.resolve() if path.is_file() else None


def _checkpoint_candidates(use_safetensors: bool | None) -> tuple[str, ...]:
    if use_safetensors is True:
        return (_SAFE_CHECKPOINT, )
    if use_safetensors is False:
        return (SPEECHT5_CHECKPOINT_FILENAME, )
    return (_SAFE_CHECKPOINT, SPEECHT5_CHECKPOINT_FILENAME)


def _local_checkpoint(
    root: Path,
    *,
    direct: Path | None,
    use_safetensors: bool | None,
    owner: str,
) -> Path:
    if direct is not None:
        if direct.suffix.lower() not in {".safetensors", ".bin"}:
            raise ValueError(f"Native {owner} accepts .safetensors or restricted .bin "
                             "checkpoints.")
        if use_safetensors is True and direct.suffix.lower() != ".safetensors":
            raise ValueError(f"`use_safetensors=True` rejects the {direct.suffix} checkpoint.")
        if use_safetensors is False and direct.suffix.lower() == ".safetensors":
            raise ValueError("`use_safetensors=False` rejects a Safetensors checkpoint.")
        return direct.resolve()
    for filename in _checkpoint_candidates(use_safetensors):
        candidate = root / filename
        if candidate.is_file():
            return candidate.resolve()
    names = ", ".join(repr(name) for name in _checkpoint_candidates(use_safetensors))
    raise FileNotFoundError(f"Native {owner} found none of the requested checkpoints: {names}.")


def _remote_optional(
    source: str,
    filename: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
) -> Path | None:
    try:
        return resolve_pretrained_file(
            source,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
    except FileNotFoundError:
        return None


def _remote_checkpoint(
    source: str,
    *,
    revision: str,
    cache_dir: str | None,
    token: str | bool | None,
    local_files_only: bool,
    use_safetensors: bool | None,
    owner: str,
) -> Path:
    for filename in _checkpoint_candidates(use_safetensors):
        candidate = _remote_optional(
            source,
            filename,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        if candidate is not None:
            return candidate
    names = ", ".join(repr(name) for name in _checkpoint_candidates(use_safetensors))
    raise FileNotFoundError(f"Native {owner} found none of the requested checkpoints: {names}.")


def _resolve_revision(
    source: str,
    config: Path,
    requested: str,
    *,
    cache_dir: str | None,
) -> str:
    return (
        get_cached_hugging_face_commit(
            source,
            config.name,
            cache_dir=cache_dir,
            revision=requested,
        ) or requested)


def _validate_controls(
    *,
    local_files_only: bool,
    use_safetensors: bool | None,
    verify_official_integrity: bool,
) -> None:
    if not isinstance(local_files_only, bool):
        raise TypeError("`local_files_only` must be a boolean.")
    if use_safetensors is not None and not isinstance(use_safetensors, bool):
        raise TypeError("`use_safetensors` must be a boolean or None.")
    if not isinstance(verify_official_integrity, bool):
        raise TypeError("`verify_official_integrity` must be a boolean.")


def resolve_speecht5_artifacts(
    source: str | Path = SPEECHT5_REPOSITORY,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    use_safetensors: bool | None = None,
    verify_official_integrity: bool = True,
) -> SpeechT5Artifacts:
    """Resolve one complete text, frontend, and checkpoint snapshot."""
    _validate_controls(
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
        verify_official_integrity=verify_official_integrity,
    )
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("SpeechT5 `source` must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        direct = local if local.is_file() else None
        root = local.parent if direct is not None else local
        return SpeechT5Artifacts(
            source=str(local.resolve()),
            revision=None,
            config=_required(root, _CONFIG, owner="SpeechT5"),
            preprocessor_config=_required(
                root,
                _PREPROCESSOR,
                owner="SpeechT5",
            ),
            tokenizer_model=_required(root, _TOKENIZER_MODEL, owner="SpeechT5"),
            tokenizer_config=_optional(root, _TOKENIZER_CONFIG),
            special_tokens_map=_optional(root, _SPECIAL_TOKENS),
            added_tokens=_optional(root, _ADDED_TOKENS),
            checkpoint=_local_checkpoint(
                root,
                direct=direct,
                use_safetensors=use_safetensors,
                owner="SpeechT5",
            ),
            official=False,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"SpeechT5 model path was not found: {local}.")

    repo_id = str(source)
    requested = revision or (SPEECHT5_REVISION if repo_id == SPEECHT5_REPOSITORY else "main")
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved = _resolve_revision(
        repo_id,
        config,
        requested,
        cache_dir=cache_dir,
    )
    required = {
        "preprocessor_config": _PREPROCESSOR,
        "tokenizer_model": _TOKENIZER_MODEL,
    }
    files = {
        name:
        resolve_pretrained_file(
            repo_id,
            filename,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for name, filename in required.items()
    }
    optional = {
        name:
        _remote_optional(
            repo_id,
            filename,
            revision=resolved,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        for name, filename in {
            "tokenizer_config": _TOKENIZER_CONFIG,
            "special_tokens_map": _SPECIAL_TOKENS,
            "added_tokens": _ADDED_TOKENS,
        }.items()
    }
    checkpoint = _remote_checkpoint(
        repo_id,
        revision=resolved,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
        owner="SpeechT5",
    )
    official = repo_id == SPEECHT5_REPOSITORY and resolved == SPEECHT5_REVISION
    if verify_official_integrity and official:
        asset_paths = {
            _CONFIG: config,
            _PREPROCESSOR: files["preprocessor_config"],
            _TOKENIZER_MODEL: files["tokenizer_model"],
            **{
                filename: optional[name]
                for name, filename in {
                    "tokenizer_config": _TOKENIZER_CONFIG,
                    "special_tokens_map": _SPECIAL_TOKENS,
                    "added_tokens": _ADDED_TOKENS,
                }.items() if optional[name] is not None
            },
        }
        for filename, path in asset_paths.items():
            _verify_file(
                path,
                label=f"Official SpeechT5 {filename}",
                expected_sha256=SPEECHT5_ASSET_SHA256[filename],
            )
        if checkpoint.name == SPEECHT5_CHECKPOINT_FILENAME:
            _verify_file(
                checkpoint,
                label="Official SpeechT5 checkpoint",
                expected_sha256=SPEECHT5_CHECKPOINT_SHA256,
                expected_size=SPEECHT5_CHECKPOINT_SIZE,
            )
    return SpeechT5Artifacts(
        source=repo_id,
        revision=resolved,
        config=config,
        checkpoint=checkpoint,
        official=official,
        **files,
        **optional,
    )


def resolve_speecht5_hifigan_artifacts(
    source: str | Path = SPEECHT5_HIFIGAN_REPOSITORY,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    use_safetensors: bool | None = None,
    verify_official_integrity: bool = True,
) -> SpeechT5HifiGanArtifacts:
    """Resolve one coherent native SpeechT5 HiFi-GAN bundle."""
    _validate_controls(
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
        verify_official_integrity=verify_official_integrity,
    )
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("SpeechT5 HiFi-GAN `source` must be a non-empty path or Hub ID.")
    local = Path(source).expanduser()
    if local.exists():
        direct = local if local.is_file() else None
        root = local.parent if direct is not None else local
        return SpeechT5HifiGanArtifacts(
            source=str(local.resolve()),
            revision=None,
            config=_required(root, _CONFIG, owner="SpeechT5 HiFi-GAN"),
            checkpoint=_local_checkpoint(
                root,
                direct=direct,
                use_safetensors=use_safetensors,
                owner="SpeechT5 HiFi-GAN",
            ),
            official=False,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"SpeechT5 HiFi-GAN path was not found: {local}.")

    repo_id = str(source)
    requested = revision or (SPEECHT5_HIFIGAN_REVISION if repo_id == SPEECHT5_HIFIGAN_REPOSITORY else "main")
    config = resolve_pretrained_file(
        repo_id,
        _CONFIG,
        revision=requested,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    resolved = _resolve_revision(
        repo_id,
        config,
        requested,
        cache_dir=cache_dir,
    )
    checkpoint = _remote_checkpoint(
        repo_id,
        revision=resolved,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
        use_safetensors=use_safetensors,
        owner="SpeechT5 HiFi-GAN",
    )
    official = (repo_id == SPEECHT5_HIFIGAN_REPOSITORY and resolved == SPEECHT5_HIFIGAN_REVISION)
    if verify_official_integrity and official:
        _verify_file(
            config,
            label="Official SpeechT5 HiFi-GAN config",
            expected_sha256=SPEECHT5_HIFIGAN_CONFIG_SHA256,
        )
        if checkpoint.name == SPEECHT5_HIFIGAN_CHECKPOINT_FILENAME:
            _verify_file(
                checkpoint,
                label="Official SpeechT5 HiFi-GAN checkpoint",
                expected_sha256=SPEECHT5_HIFIGAN_CHECKPOINT_SHA256,
                expected_size=SPEECHT5_HIFIGAN_CHECKPOINT_SIZE,
            )
    return SpeechT5HifiGanArtifacts(
        source=repo_id,
        revision=resolved,
        config=config,
        checkpoint=checkpoint,
        official=official,
    )


__all__ = [
    "SpeechT5Artifacts",
    "SpeechT5HifiGanArtifacts",
    "resolve_speecht5_artifacts",
    "resolve_speecht5_hifigan_artifacts",
]
