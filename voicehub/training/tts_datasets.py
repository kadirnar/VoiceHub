"""Portable manifests and datasets for TTS training."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Callable, Iterable, Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.training.data_contracts import TTSDataArchitecture, _field_present, get_tts_dataset_spec
from voicehub.training.dataset_base import SpeechDataset

_COMMON_ALIASES = MappingProxyType({
    "transcript": "text",
    "sentence": "text",
    "audio_path": "audio",
    "audio_filepath": "audio",
    "wav_path": "audio",
    "reference_audio_path": "reference_audio",
    "speaker_audio": "reference_audio",
})
_MODEL_ALIASES = MappingProxyType({
    "qwen3tts":
    MappingProxyType({
        "reference_audio": "ref_audio",
        "reference_audio_path": "ref_audio",
        "speaker_audio": "ref_audio",
    }),
})
_PATH_FIELDS = frozenset({
    "audio",
    "audios",
    "ref_audio",
    "prompt_audio",
    "speaker_audio",
    "reference_audio",
    "source_audio",
    "target_audio",
    "source_reference_audio",
    "target_reference_audio",
    "target_waveform",
})


class TTSDataset(SpeechDataset):
    """Portable TTS records loaded from mappings, JSON/JSONL, CSV, or TSV.

    ``TTSDataset`` normalizes common manifest aliases, resolves local
    audio paths relative to one root, validates architecture-level
    record shapes, supports deterministic group-disjoint splits, and
    exposes a stable resume fingerprint.  It never tokenizes text or
    converts audio into model targets.
    """

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        model_type: str | None = None,
        architecture: TTSDataArchitecture | str | None = None,
        root: str | PathLike[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        validate: bool = True,
        validate_files: bool = False,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ):
        if model_type is None and architecture is None:
            raise ValueError("TTSDataset requires model_type or architecture.")
        self.spec = get_tts_dataset_spec(
            model_type,
            architecture=architecture,
        )
        self.model_type = self.spec.model_type
        self.architecture = self.spec.architecture
        self.root = (None if root is None else Path(root).expanduser().resolve())
        self.aliases = self._normalize_aliases(aliases)
        self.validate = bool(validate)
        self.validate_files = bool(validate_files)
        if transform_fingerprint is not None and (not isinstance(transform_fingerprint, str) or
                                                  not transform_fingerprint.strip()):
            raise ValueError("transform_fingerprint must be a non-empty string or None.")
        self.transform_fingerprint = (
            None if transform_fingerprint is None else transform_fingerprint.strip())
        self.variant_names: tuple[str, ...] = ()

        normalized = []
        variants = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"TTS record {index} must be a mapping, received "
                    f"{type(record).__name__}.")
            value = self._normalize_record(record, index=index)
            if self.validate:
                variants.append(self.spec.match_variant(value, index=index))
            else:
                variants.append("unchecked")
            normalized.append(value)
        if not normalized:
            raise ValueError("TTSDataset requires at least one record.")
        self.variant_names = tuple(variants)
        super().__init__(normalized, transform=transform)

    @classmethod
    def from_manifest(
        cls,
        path: str | PathLike[str],
        *,
        model_type: str | None = None,
        architecture: TTSDataArchitecture | str | None = None,
        root: str | PathLike[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        validate: bool = True,
        validate_files: bool = False,
        delimiter: str | None = None,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ) -> TTSDataset:
        """Load a JSON, JSON Lines, CSV, or TSV source manifest."""
        manifest = Path(path).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"TTS manifest does not exist: {manifest}.")
        records = _read_manifest(manifest, delimiter=delimiter)
        return cls(
            records,
            model_type=model_type,
            architecture=architecture,
            root=(manifest.parent if root is None else root),
            aliases=aliases,
            validate=validate,
            validate_files=validate_files,
            transform=transform,
            transform_fingerprint=transform_fingerprint,
        )

    @classmethod
    def from_ljspeech(
        cls,
        root: str | PathLike[str],
        *,
        model_type: str | None = None,
        architecture: TTSDataArchitecture | str | None = None,
        metadata_filename: str = "metadata.csv",
        audio_directory: str = "wavs",
        audio_extension: str = ".wav",
        use_normalized_text: bool = True,
        validate: bool = True,
        validate_files: bool = False,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ) -> TTSDataset:
        """Load the common ``id|text|normalized_text`` LJSpeech layout."""
        directory = Path(root).expanduser().resolve()
        metadata = directory / metadata_filename
        if not metadata.is_file():
            raise FileNotFoundError(f"LJSpeech metadata does not exist: {metadata}.")
        extension = audio_extension if audio_extension.startswith(".") else "." + audio_extension
        records = []
        with metadata.open(encoding="utf-8", newline="") as stream:
            for line_number, row in enumerate(csv.reader(stream, delimiter="|"), start=1):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) < 2:
                    raise ValueError(f"{metadata}:{line_number} must contain at least id and text.")
                text_index = 2 if use_normalized_text and len(row) > 2 else 1
                records.append({
                    "id": row[0].strip(),
                    "text": row[text_index].strip(),
                    "audio": str(Path(audio_directory) / f"{row[0].strip()}{extension}"),
                })
        return cls(
            records,
            model_type=model_type,
            architecture=architecture,
            root=directory,
            validate=validate,
            validate_files=validate_files,
            transform=transform,
            transform_fingerprint=transform_fingerprint,
        )

    @classmethod
    def coerce(
        cls,
        records_or_manifest: TTSDataset | Iterable[Mapping[str, Any]] | str | PathLike[str],
        *,
        model_type: str | None = None,
        architecture: TTSDataArchitecture | str | None = None,
        root: str | PathLike[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        validate: bool = True,
        validate_files: bool = False,
        transform_fingerprint: str | None = None,
    ) -> TTSDataset:
        """Normalize an existing dataset, record iterable, or manifest path."""
        if isinstance(records_or_manifest, cls):
            requested_spec = (
                records_or_manifest.spec if model_type is None and architecture is None else
                get_tts_dataset_spec(model_type, architecture=architecture))
            unchanged_target = (
                records_or_manifest.model_type == requested_spec.model_type and
                records_or_manifest.architecture is requested_spec.architecture)
            stricter_validation = (
                validate and not records_or_manifest.validate or
                validate_files and not records_or_manifest.validate_files)
            changed_options = (root is not None or aliases is not None or transform_fingerprint is not None)
            if unchanged_target and not stricter_validation and not changed_options:
                return records_or_manifest
            return cls(
                records_or_manifest._records,
                model_type=requested_spec.model_type,
                architecture=(None if requested_spec.model_type is not None else requested_spec.architecture),
                root=(records_or_manifest.root if root is None else root),
                aliases=aliases,
                validate=validate,
                validate_files=validate_files,
                transform=records_or_manifest.transform,
                transform_fingerprint=(
                    records_or_manifest.transform_fingerprint
                    if transform_fingerprint is None else transform_fingerprint),
            )
        if isinstance(records_or_manifest, (str, PathLike)):
            return cls.from_manifest(
                records_or_manifest,
                model_type=model_type,
                architecture=architecture,
                root=root,
                aliases=aliases,
                validate=validate,
                validate_files=validate_files,
                transform_fingerprint=transform_fingerprint,
            )
        return cls(
            records_or_manifest,
            model_type=model_type,
            architecture=architecture,
            root=root,
            aliases=aliases,
            validate=validate,
            validate_files=validate_files,
            transform_fingerprint=transform_fingerprint,
        )

    def train_test_split(
        self,
        *,
        validation_fraction: float = 0.1,
        seed: int = 42,
        group_by: str | None = None,
    ) -> tuple[TTSDataset, TTSDataset]:
        """Create deterministic train/validation datasets.

        When ``group_by`` is provided, every value of that field is
        assigned wholly to one split so speakers, sessions, or
        recordings cannot leak across train and validation data.
        """
        if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or
                not 0.0 < float(validation_fraction) < 1.0):
            raise ValueError("validation_fraction must be between 0 and 1.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if len(self) < 2:
            raise ValueError("At least two TTS records are required for a split.")

        randomizer = random.Random(seed)
        if group_by is None:
            indices = list(range(len(self)))
            randomizer.shuffle(indices)
            validation_count = max(
                1,
                min(len(indices) - 1, round(len(indices) * float(validation_fraction))),
            )
            validation_indices = set(indices[:validation_count])
        else:
            if not isinstance(group_by, str) or not group_by.strip():
                raise ValueError("group_by must be a non-empty field name or None.")
            groups = {}
            for index, record in enumerate(self._records):
                if not _field_present(record, group_by):
                    raise ValueError(f"TTS record {index} is missing split group field {group_by!r}.")
                groups.setdefault(str(record[group_by]), []).append(index)
            names = sorted(groups)
            if len(names) < 2:
                raise ValueError(f"At least two {group_by!r} groups are required.")
            randomizer.shuffle(names)
            validation_group_count = max(
                1,
                min(len(names) - 1, round(len(names) * float(validation_fraction))),
            )
            validation_indices = {index for name in names[:validation_group_count] for index in groups[name]}

        train_records = [
            record for index, record in enumerate(self._records) if index not in validation_indices
        ]
        validation_records = [
            record for index, record in enumerate(self._records) if index in validation_indices
        ]
        options = {
            "model_type": self.model_type,
            "architecture": self.architecture,
            "root": self.root,
            "aliases": {},
            "validate": self.validate,
            "validate_files": False,
            "transform": self.transform,
            "transform_fingerprint": self.transform_fingerprint,
        }
        return (
            type(self)(train_records, **options),
            type(self)(validation_records, **options),
        )

    def to_jsonl(
        self,
        path: str | PathLike[str],
        *,
        relative_to: str | PathLike[str] | None = None,
    ) -> Path:
        """Write a portable JSON Lines manifest and return its path."""
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        relative_root = (
            destination.parent if relative_to is None else Path(relative_to).expanduser().resolve())
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for index, record in enumerate(self._records):
                serializable = self._portable_record(
                    record,
                    relative_to=relative_root,
                )
                try:
                    line = json.dumps(
                        serializable,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"TTS record {index} contains values that cannot be written "
                        "to JSON Lines. Store tensors/features separately and put "
                        "their paths in the manifest.") from exc
                stream.write(line + "\n")
        return destination

    def resume_fingerprint(self) -> dict[str, Any]:
        """Return a stable manifest-content identity for exact resume."""
        if self.transform is not None and self.transform_fingerprint is None:
            raise ValueError(
                "TTSDataset transforms require an explicit "
                "transform_fingerprint for exact resume.")
        canonical = json.dumps(
            {
                "records": [_fingerprint_value(record) for record in self._records],
                "transform_fingerprint": self.transform_fingerprint,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "model_type": self.model_type,
            "architecture": self.architecture.value,
            "length": len(self),
            "transform_fingerprint": self.transform_fingerprint,
            "content_sha256": hashlib.sha256(canonical).hexdigest(),
        }

    def _normalize_aliases(
        self,
        aliases: Mapping[str, str] | None,
    ) -> Mapping[str, str]:
        merged = dict(_COMMON_ALIASES)
        merged.update(_MODEL_ALIASES.get(self.model_type or "", {}))
        if aliases is not None:
            if not isinstance(aliases, Mapping):
                raise TypeError("aliases must be a mapping or None.")
            merged.update(aliases)
        normalized = {}
        for source, target in merged.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError("TTS dataset alias names must be non-empty strings.")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("TTS dataset alias targets must be non-empty strings.")
            source = source.strip()
            target = target.strip()
            if source != target:
                normalized[source] = target
        # Keep dataset instances pickle-safe for multiprocessing DataLoaders.
        # The alias map is only used while constructing normalized records.
        return normalized

    def _normalize_record(
        self,
        record: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        value = dict(record)
        for source, target in self.aliases.items():
            if source not in value:
                continue
            if target in value:
                raise ValueError(
                    f"TTS record {index} contains both alias {source!r} and "
                    f"canonical field {target!r}.")
            value[target] = value.pop(source)

        for name in tuple(value):
            if name not in _PATH_FIELDS and not name.endswith(("_path", "_filepath")):
                continue
            value[name] = self._normalize_path_value(
                value[name],
                index=index,
                field_name=name,
            )
        for name in ("conversation", "messages"):
            if name in value:
                value[name] = self._normalize_conversation_paths(
                    value[name],
                    index=index,
                    field_name=name,
                )

        text = value.get("text")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError(f"TTS record {index} field 'text' must be a non-empty string.")
        return value

    def _normalize_path_value(
        self,
        field_value: Any,
        *,
        index: int,
        field_name: str,
    ) -> Any:
        if isinstance(field_value, Mapping):
            nested = dict(field_value)
            if "path" in nested:
                nested["path"] = self._normalize_path_value(
                    nested["path"],
                    index=index,
                    field_name=f"{field_name}.path",
                )
            return nested
        if isinstance(field_value, (list, tuple)):
            normalized = [
                self._normalize_path_value(
                    item,
                    index=index,
                    field_name=f"{field_name}[{item_index}]",
                ) for item_index, item in enumerate(field_value)
            ]
            return tuple(normalized) if isinstance(field_value, tuple) else normalized
        if not isinstance(field_value, (str, PathLike)):
            return field_value
        path = Path(field_value).expanduser()
        if not path.is_absolute() and self.root is not None:
            path = self.root / path
        if path.is_absolute():
            path = path.resolve()
        if self.validate_files and not path.is_file():
            raise FileNotFoundError(f"TTS record {index} field {field_name!r} does not exist: "
                                    f"{path}.")
        return str(path)

    def _normalize_conversation_paths(
        self,
        value: Any,
        *,
        index: int,
        field_name: str,
    ) -> Any:
        if isinstance(value, Mapping):
            nested = dict(value)
            if nested.get("type") == "audio":
                for name in ("audio", "path"):
                    if name in nested:
                        nested[name] = self._normalize_path_value(
                            nested[name],
                            index=index,
                            field_name=f"{field_name}.{name}",
                        )
            if "content" in nested:
                nested["content"] = self._normalize_conversation_paths(
                    nested["content"],
                    index=index,
                    field_name=f"{field_name}.content",
                )
            return nested
        if isinstance(value, (list, tuple)):
            normalized = [
                self._normalize_conversation_paths(
                    item,
                    index=index,
                    field_name=f"{field_name}[{item_index}]",
                ) for item_index, item in enumerate(value)
            ]
            return tuple(normalized) if isinstance(value, tuple) else normalized
        return value

    @staticmethod
    def _portable_record(
        record: Mapping[str, Any],
        *,
        relative_to: Path,
    ) -> dict[str, Any]:
        value = dict(record)
        for name in tuple(value):
            if name not in _PATH_FIELDS and not name.endswith(("_path", "_filepath")):
                continue
            value[name] = TTSDataset._portable_path_value(
                value[name],
                relative_to=relative_to,
            )
        for name in ("conversation", "messages"):
            if name in value:
                value[name] = TTSDataset._portable_conversation_paths(
                    value[name],
                    relative_to=relative_to,
                )
        return value

    @staticmethod
    def _portable_path_value(field_value: Any, *, relative_to: Path) -> Any:
        if isinstance(field_value, Mapping):
            nested = dict(field_value)
            if "path" in nested:
                nested["path"] = TTSDataset._portable_path_value(
                    nested["path"],
                    relative_to=relative_to,
                )
            return nested
        if isinstance(field_value, (list, tuple)):
            normalized = [
                TTSDataset._portable_path_value(
                    item,
                    relative_to=relative_to,
                ) for item in field_value
            ]
            return tuple(normalized) if isinstance(field_value, tuple) else normalized
        if not isinstance(field_value, (str, PathLike)):
            return field_value
        path = Path(field_value).expanduser()
        if not path.is_absolute():
            return str(path)
        try:
            return str(path.relative_to(relative_to))
        except ValueError:
            return str(path)

    @staticmethod
    def _portable_conversation_paths(value: Any, *, relative_to: Path) -> Any:
        if isinstance(value, Mapping):
            nested = dict(value)
            if nested.get("type") == "audio":
                for name in ("audio", "path"):
                    if name in nested:
                        nested[name] = TTSDataset._portable_path_value(
                            nested[name],
                            relative_to=relative_to,
                        )
            if "content" in nested:
                nested["content"] = TTSDataset._portable_conversation_paths(
                    nested["content"],
                    relative_to=relative_to,
                )
            return nested
        if isinstance(value, (list, tuple)):
            normalized = [
                TTSDataset._portable_conversation_paths(
                    item,
                    relative_to=relative_to,
                ) for item in value
            ]
            return tuple(normalized) if isinstance(value, tuple) else normalized
        return value


def _read_manifest(path: Path, *, delimiter: str | None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid TTS JSON manifest {path}: {exc}.") from exc
        if not isinstance(payload, list):
            raise TypeError(f"TTS JSON manifest {path} must contain a list of objects.")
        records = payload
    elif suffix in (".csv", ".tsv"):
        selected_delimiter = delimiter
        if selected_delimiter is None:
            selected_delimiter = "\t" if suffix == ".tsv" else ","
        if not isinstance(selected_delimiter, str) or len(selected_delimiter) != 1:
            raise ValueError("CSV/TSV delimiter must be exactly one character.")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=selected_delimiter)
            if not reader.fieldnames:
                raise ValueError(f"TTS tabular manifest {path} has no header.")
            records = [{
                key: _coerce_tabular_value(value)
                for key, value in row.items() if key is not None and value not in (None, "")
            } for row in reader]
    else:
        records = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid TTS JSON Lines record at {path}:{line_number}: "
                        f"{exc.msg}.") from exc
                if not isinstance(record, Mapping):
                    raise TypeError(f"TTS JSON Lines record at {path}:{line_number} must be an object.")
                records.append(dict(record))
    if not records:
        raise ValueError(f"No TTS records found in manifest {path}.")
    return records


def _coerce_tabular_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return ""
    if stripped[0] in "[{\"" or stripped in ("true", "false", "null"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
    return stripped


def _fingerprint_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, PathLike):
        return str(value)
    if isinstance(value, Mapping):
        non_string_keys = [key for key in value if not isinstance(key, str)]
        if non_string_keys:
            raise TypeError(
                "TTSDataset resume fingerprints require string mapping "
                f"keys; received {non_string_keys[0]!r}.")
        return {key: _fingerprint_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_fingerprint_value(item) for item in value]
        items.sort(
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ))
        return {
            "__set__": items,
        }
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach().cpu()
        except (AttributeError, RuntimeError, TypeError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return {
                "__class__": f"{type(value).__module__}.{type(value).__qualname__}",
                "dtype": str(getattr(value, "dtype", "")),
                "shape": list(getattr(value, "shape", ())),
                "values": _fingerprint_value(tolist()),
            }
        except (RuntimeError, TypeError, ValueError):
            pass
    raise TypeError(
        "TTSDataset resume fingerprints support JSON values, paths, "
        "sets, tensors, and array-like values; received unsupported "
        f"{type(value).__module__}.{type(value).__qualname__}.")


__all__ = ["TTSDataset"]
