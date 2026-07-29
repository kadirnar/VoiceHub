"""Versioned, integrity-checked manifests for native VoiceHub artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voicehub.checkpointing.errors import CheckpointFormatError, CheckpointIntegrityError

MANIFEST_NAME = "voicehub_manifest.json"
CURRENT_FORMAT_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"`{field_name}` must contain lowercase letters, digits, dots, "
            "underscores, or hyphens and begin with a letter or digit.")
    return value


def _text(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{field_name}` must be a non-empty string or None.")
    return value.strip()


def _safe_relative_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Artifact file path must be relative and safe: {value!r}.")
    return path.as_posix()


@dataclass(frozen=True)
class ArtifactFile:
    """Size and SHA-256 digest for one artifact file."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _safe_relative_path(self.path))
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("Artifact file `size` must be a non-negative integer.")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("Artifact file `sha256` must be a lowercase SHA-256 digest.")

    @classmethod
    def from_path(
        cls,
        root: str | Path,
        relative_path: str | Path,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> ArtifactFile:
        root_path = Path(root).expanduser().resolve()
        normalized = _safe_relative_path(relative_path)
        source = (root_path / normalized).resolve()
        if source.parent != root_path and root_path not in source.parents:
            raise ValueError(f"Artifact file escapes its root: {relative_path!r}.")
        if not source.is_file():
            raise FileNotFoundError(f"Artifact file was not found: {source}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
        return cls(
            path=normalized,
            size=source.stat().st_size,
            sha256=digest.hexdigest(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ArtifactFile:
        if not isinstance(value, dict):
            raise CheckpointFormatError("Manifest file record must be an object.")
        try:
            return cls(
                path=value["path"],
                size=value["size"],
                sha256=value["sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointFormatError(f"Invalid manifest file record: {error}.") from error


@dataclass(frozen=True)
class VoiceHubManifest:
    """Portable description of a native model, its assets, and provenance."""

    architecture: str
    architecture_version: str
    checkpoint_format: str
    adapter_version: str
    tensor_namespace: str = "voicehub"
    format_version: int = CURRENT_FORMAT_VERSION
    source: str | None = None
    source_revision: str | None = None
    source_license: str | None = None
    weight_license: str | None = None
    processor_assets: tuple[str, ...] = ()
    training_recipe: str | None = None
    files: tuple[ArtifactFile, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (isinstance(self.format_version, bool) or not isinstance(self.format_version, int) or
                self.format_version <= 0):
            raise ValueError("Manifest `format_version` must be a positive integer.")
        object.__setattr__(
            self,
            "architecture",
            _identifier(self.architecture, field_name="architecture"),
        )
        object.__setattr__(
            self,
            "checkpoint_format",
            _identifier(self.checkpoint_format, field_name="checkpoint_format"),
        )
        object.__setattr__(
            self,
            "tensor_namespace",
            _identifier(self.tensor_namespace, field_name="tensor_namespace"),
        )
        for name in ("architecture_version", "adapter_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Manifest `{name}` must be a non-empty string.")
            object.__setattr__(self, name, value.strip())
        for name in (
                "source",
                "source_revision",
                "source_license",
                "weight_license",
                "training_recipe",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )
        assets = tuple(_safe_relative_path(path) for path in self.processor_assets)
        if len(assets) != len(set(assets)):
            raise ValueError("Manifest `processor_assets` cannot contain duplicates.")
        object.__setattr__(self, "processor_assets", assets)
        normalized_files = tuple(self.files)
        if any(not isinstance(item, ArtifactFile) for item in normalized_files):
            raise TypeError("Manifest `files` must contain ArtifactFile instances.")
        paths = tuple(item.path for item in normalized_files)
        if len(paths) != len(set(paths)):
            raise ValueError("Manifest `files` cannot contain duplicate paths.")
        object.__setattr__(
            self,
            "files",
            tuple(sorted(normalized_files, key=lambda item: item.path)),
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("Manifest `metadata` must be a mapping.")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise TypeError("Manifest `metadata` must contain finite JSON values.") from error
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "architecture": self.architecture,
            "architecture_version": self.architecture_version,
            "checkpoint": {
                "format": self.checkpoint_format,
                "adapter_version": self.adapter_version,
                "tensor_namespace": self.tensor_namespace,
            },
            "source": {
                key: value
                for key, value in {
                    "identifier": self.source,
                    "revision": self.source_revision,
                    "source_license": self.source_license,
                    "weight_license": self.weight_license,
                }.items() if value is not None
            },
            "processor_assets": list(self.processor_assets),
            "training_recipe": self.training_recipe,
            "files": [record.to_dict() for record in self.files],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Any) -> VoiceHubManifest:
        if not isinstance(value, dict):
            raise CheckpointFormatError("VoiceHub manifest must be a JSON object.")
        checkpoint = value.get("checkpoint")
        source = value.get("source", {})
        if not isinstance(checkpoint, dict) or not isinstance(source, dict):
            raise CheckpointFormatError("Manifest `checkpoint` and `source` must be JSON objects.")
        files = value.get("files", [])
        if not isinstance(files, list):
            raise CheckpointFormatError("Manifest `files` must be a JSON array.")
        known = {
            "format_version",
            "architecture",
            "architecture_version",
            "checkpoint",
            "source",
            "processor_assets",
            "training_recipe",
            "files",
            "metadata",
        }
        unknown = sorted(set(value) - known)
        if unknown:
            raise CheckpointFormatError(f"Manifest contains unknown top-level fields: {unknown!r}.")
        try:
            manifest = cls(
                format_version=value["format_version"],
                architecture=value["architecture"],
                architecture_version=value["architecture_version"],
                checkpoint_format=checkpoint["format"],
                adapter_version=checkpoint["adapter_version"],
                tensor_namespace=checkpoint.get("tensor_namespace", "voicehub"),
                source=source.get("identifier"),
                source_revision=source.get("revision"),
                source_license=source.get("source_license"),
                weight_license=source.get("weight_license"),
                processor_assets=tuple(value.get("processor_assets", ())),
                training_recipe=value.get("training_recipe"),
                files=tuple(ArtifactFile.from_dict(item) for item in files),
                metadata=value.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CheckpointFormatError(f"Invalid VoiceHub manifest: {error}.") from error
        if manifest.format_version > CURRENT_FORMAT_VERSION:
            raise CheckpointFormatError(
                f"Manifest format {manifest.format_version} is newer than "
                f"supported format {CURRENT_FORMAT_VERSION}.")
        return manifest

    @classmethod
    def load(cls, path_or_directory: str | Path) -> VoiceHubManifest:
        source = Path(path_or_directory).expanduser()
        path = source / MANIFEST_NAME if source.is_dir() else source
        if not path.is_file():
            raise FileNotFoundError(f"VoiceHub manifest was not found: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CheckpointFormatError(f"Could not parse VoiceHub manifest {path}: {error}.") from error
        return cls.from_dict(value)

    def save(self, directory: str | Path) -> Path:
        root = Path(directory).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        path = root / MANIFEST_NAME
        encoded = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ) + "\n")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=root,
                    prefix=f".{MANIFEST_NAME}.",
                    suffix=".tmp",
                    delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return path

    def verify(self, directory: str | Path) -> None:
        """Verify all recorded files and processor assets under
        ``directory``."""
        root = Path(directory).expanduser().resolve()
        file_records = {record.path: record for record in self.files}
        missing_assets = sorted(set(self.processor_assets) - set(file_records))
        if missing_assets:
            raise CheckpointIntegrityError(
                "Processor assets are not covered by manifest file records: "
                f"{missing_assets!r}.")
        for record in self.files:
            path = (root / record.path).resolve()
            if path.parent != root and root not in path.parents:
                raise CheckpointIntegrityError(f"Manifest file escapes artifact directory: {record.path!r}.")
            if not path.is_file():
                raise CheckpointIntegrityError(f"Manifest file is missing: {record.path!r}.")
            if path.stat().st_size != record.size:
                raise CheckpointIntegrityError(
                    f"Manifest size mismatch for {record.path!r}: expected "
                    f"{record.size}, found {path.stat().st_size}.")
            actual = ArtifactFile.from_path(root, record.path).sha256
            if actual != record.sha256:
                raise CheckpointIntegrityError(f"Manifest SHA-256 mismatch for {record.path!r}.")


def build_manifest_files(
    directory: str | Path,
    paths: Iterable[str | Path],
) -> tuple[ArtifactFile, ...]:
    """Hash an explicit set of artifact-relative paths deterministically."""
    normalized = tuple(_safe_relative_path(path) for path in paths)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Artifact paths cannot contain duplicates.")
    return tuple(ArtifactFile.from_path(directory, path) for path in sorted(normalized))
