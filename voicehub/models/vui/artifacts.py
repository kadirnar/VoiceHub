"""Pinned, coherent artifact resolution for the native Vui runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from voicehub.hub import resolve_pretrained_file
from voicehub.hub_transport import get_cached_hugging_face_commit
from voicehub.path_utils import is_explicit_local_path

VUI_REPO_ID = "fluxions/vui"
VUI_REVISION = "8dc2bd9993a8118b6e2b71f3d9d92d1deb80e5f7"
VUI_MODEL_FILENAME = "vui-abraham-100m.pt"
VUI_MODEL_SIZE = 198_204_301
VUI_MODEL_SHA256 = ("28353f13788c353160efbfc4fa5f5db56844746d3de9a92531dfee704cc394ff")
VUI_CODEC_FILENAME = "fluac-22hz-22khz.pt"
VUI_CODEC_SIZE = 306_573_425
VUI_CODEC_SHA256 = ("04d1ee6567b5eaade6720bf7cc0241fbbd3c0aaeca00ac37cd1656afa08f3c96")

_OFFICIAL_MODEL_FILES = frozenset({
    "vui-100m-base.pt",
    "vui-cohost-100m.pt",
    VUI_MODEL_FILENAME,
})


@dataclass(frozen=True, slots=True)
class VuiArtifacts:
    """Every checkpoint required by one Vui runtime."""

    source: str
    revision: str | None
    model_checkpoint: Path
    codec_checkpoint: Path
    official: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_official_file(
    path: Path,
    *,
    label: str,
    expected_size: int,
    expected_sha256: str,
) -> None:
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise OSError(f"{label} has size {actual_size}; expected {expected_size} bytes.")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise OSError(f"{label} has SHA-256 {actual_sha256}; expected "
                      f"{expected_sha256}.")


def _require_local_file(root: Path, filename: str, *, label: str) -> Path:
    resolved = root / filename
    if not resolved.is_file():
        raise FileNotFoundError(f"Native Vui requires {label} {filename!r} in {root}.")
    return resolved.resolve()


def _resolve_local(
    source: Path,
    *,
    model_filename: str,
    codec_filename: str,
) -> VuiArtifacts:
    if source.is_file():
        if source.name != model_filename:
            raise ValueError(
                "A direct Vui checkpoint file must match "
                f"`model_filename={model_filename!r}`.")
        root = source.parent
        model_checkpoint = source.resolve()
    else:
        root = source
        model_checkpoint = _require_local_file(
            root,
            model_filename,
            label="model checkpoint",
        )
    codec_checkpoint = _require_local_file(
        root,
        codec_filename,
        label="Fluac codec checkpoint",
    )
    return VuiArtifacts(
        source=str(source.resolve()),
        revision=None,
        model_checkpoint=model_checkpoint,
        codec_checkpoint=codec_checkpoint,
        official=False,
    )


def resolve_vui_artifacts(
    source: str | Path = VUI_MODEL_FILENAME,
    *,
    model_filename: str | None = None,
    codec_filename: str = VUI_CODEC_FILENAME,
    revision: str | None = None,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
    verify_official_integrity: bool = True,
) -> VuiArtifacts:
    """Resolve the acoustic model and codec from one immutable snapshot.

    The three historical checkpoint filenames remain supported as short
    aliases. They resolve to the pinned official repository revision
    rather than the moving ``main`` branch. Local directories must
    contain both artifacts so a runtime can never mix unrelated
    snapshots.
    """
    if not isinstance(source, (str, Path)) or not str(source).strip():
        raise ValueError("Vui source must be a non-empty path or Hub ID.")
    if not isinstance(codec_filename, str) or not codec_filename.strip():
        raise ValueError("Vui codec_filename must be a non-empty filename.")

    local = Path(source).expanduser()
    selected_model = model_filename
    if selected_model is None:
        selected_model = (
            local.name if str(source) in _OFFICIAL_MODEL_FILES or local.is_file() else VUI_MODEL_FILENAME)
    if not isinstance(selected_model, str) or not selected_model.strip():
        raise ValueError("Vui model_filename must be a non-empty filename.")
    if Path(selected_model).name != selected_model:
        raise ValueError("Vui model_filename must not contain directories.")
    if Path(codec_filename).name != codec_filename:
        raise ValueError("Vui codec_filename must not contain directories.")

    if local.exists():
        return _resolve_local(
            local,
            model_filename=selected_model,
            codec_filename=codec_filename,
        )
    if is_explicit_local_path(source):
        raise FileNotFoundError(f"Vui model path was not found: {local}.")

    is_alias = str(source) in _OFFICIAL_MODEL_FILES
    repo_id = VUI_REPO_ID if is_alias else str(source)
    official = repo_id == VUI_REPO_ID
    requested_revision = revision or (VUI_REVISION if official else "main")
    model_checkpoint = resolve_pretrained_file(
        repo_id,
        selected_model,
        revision=requested_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    cached_commit = get_cached_hugging_face_commit(
        repo_id,
        selected_model,
        revision=requested_revision,
        cache_dir=cache_dir,
    )
    resolved_revision = cached_commit or requested_revision
    codec_checkpoint = resolve_pretrained_file(
        repo_id,
        codec_filename,
        revision=resolved_revision,
        cache_dir=cache_dir,
        token=token,
        local_files_only=local_files_only,
    )
    if (verify_official_integrity and official and selected_model == VUI_MODEL_FILENAME and
            codec_filename == VUI_CODEC_FILENAME):
        _verify_official_file(
            model_checkpoint,
            label="Official Vui model checkpoint",
            expected_size=VUI_MODEL_SIZE,
            expected_sha256=VUI_MODEL_SHA256,
        )
        _verify_official_file(
            codec_checkpoint,
            label="Official Fluac codec checkpoint",
            expected_size=VUI_CODEC_SIZE,
            expected_sha256=VUI_CODEC_SHA256,
        )
    return VuiArtifacts(
        source=repo_id,
        revision=resolved_revision,
        model_checkpoint=model_checkpoint,
        codec_checkpoint=codec_checkpoint,
        official=official,
    )


__all__ = [
    "VUI_CODEC_FILENAME",
    "VUI_CODEC_SHA256",
    "VUI_CODEC_SIZE",
    "VUI_MODEL_FILENAME",
    "VUI_MODEL_SHA256",
    "VUI_MODEL_SIZE",
    "VUI_REPO_ID",
    "VUI_REVISION",
    "VuiArtifacts",
    "resolve_vui_artifacts",
]
