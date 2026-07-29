"""Portable manifests and datasets for ASR fine-tuning."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

from voicehub.training.asr_data_contracts import ASRDataArchitecture, get_asr_dataset_spec
from voicehub.training.dataset_base import SpeechDataset

_COMMON_ALIASES = MappingProxyType({
    "audio_path": "audio",
    "audio_filepath": "audio",
    "wav_path": "audio",
    "wav": "audio",
    "waveform": "audio",
    "speech": "audio",
    "file": "audio",
    "path": "audio",
    "transcript": "text",
    "transcription": "text",
    "sentence": "text",
    "target_text": "text",
    "wrd": "text",
    "txt": "text",
    "sample_rate": "sampling_rate",
    "sr": "sampling_rate",
    "lang": "language",
    "locale": "language",
})
_MODEL_ALIASES = MappingProxyType({
    "asr_seamless_m4t_v2":
    MappingProxyType({
        "target_lang": "target_language",
    }),
    "asr_funasr":
    MappingProxyType({
        "emo_target": "emotion",
        "event_target": "event",
        "source": "audio",
        "target": "text",
        "text_language": "language",
        "with_or_wo_itn": "use_itn",
    }),
})
_PATH_FIELDS = frozenset({"audio"})
_SENSEVOICE_CONTROL_WRAPPERS = ("<|", "|>")


def _field_present(record: Mapping[str, Any], name: str) -> bool:
    if name not in record or record[name] is None:
        return False
    value = record[name]
    return not isinstance(value, (str, bytes, bytearray)) or bool(value.strip())


class EpochGroupedBatchSampler:
    """Deterministic batches that never mix model-incompatible metadata.

    Cohere ASR requires a shared language and punctuation mode within a
    batch, while SeamlessM4T-v2 requires one target language. This
    sampler groups those records before shuffling and exposes epoch-
    addressable state so exact Trainer resumes remain deterministic.
    """

    def __init__(
        self,
        dataset: ASRDataset,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("ASR batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("ASR batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _grouped_indices(self) -> list[list[int]]:
        grouped: dict[tuple[Any, ...], list[int]] = {}
        for index, record in enumerate(self.dataset._records):
            key = self.dataset.batch_group_key(record)
            grouped.setdefault(key, []).append(index)
        groups = [
            grouped[key] for key in sorted(
                grouped,
                key=lambda value: json.dumps(
                    _fingerprint_value(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        ]
        if self.shuffle:
            randomizer = random.Random(self.seed + self.epoch)
            for indices in groups:
                randomizer.shuffle(indices)
            randomizer.shuffle(groups)
        return groups

    def __iter__(self) -> Iterator[list[int]]:
        for indices in self._grouped_indices():
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        group_sizes: dict[tuple[Any, ...], int] = {}
        for record in self.dataset._records:
            key = self.dataset.batch_group_key(record)
            group_sizes[key] = group_sizes.get(key, 0) + 1
        if self.drop_last:
            return sum(size // self.batch_size for size in group_sizes.values())
        return sum(math.ceil(size / self.batch_size) for size in group_sizes.values())

    def state_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = {
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "shuffle": self.shuffle,
        }
        for name, value in expected.items():
            if state_dict.get(name) != value:
                raise ValueError(
                    f"ASR batch sampler {name} differs from the checkpoint "
                    f"({state_dict.get(name)!r} != {value!r}).")
        self.epoch = int(state_dict["epoch"])


class ASRDataset(SpeechDataset):
    """Validated ASR records from mappings, manifests, or WAV folders.

    The dataset normalizes common upstream field names, resolves
    relative paths, validates model-specific raw or preprocessed record
    contracts, and leaves waveform decoding and tokenization to the
    selected model.
    """

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        model_type: str | None = None,
        architecture: ASRDataArchitecture | str | None = None,
        root: str | PathLike[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        validate: bool = True,
        validate_files: bool = False,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ) -> None:
        if model_type is None and architecture is None:
            raise ValueError("ASRDataset requires model_type or architecture.")
        self.spec = get_asr_dataset_spec(
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

        normalized = []
        variants = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(
                    f"ASR record {index} must be a mapping, received "
                    f"{type(record).__name__}.")
            value = self._normalize_record(record, index=index)
            variants.append(self.spec.match_variant(value, index=index) if self.validate else "unchecked")
            normalized.append(value)
        if not normalized:
            raise ValueError("ASRDataset requires at least one record.")
        self.variant_names = tuple(variants)
        super().__init__(normalized, transform=transform)

    @classmethod
    def from_manifest(
        cls,
        path: str | PathLike[str],
        *,
        model_type: str | None = None,
        architecture: ASRDataArchitecture | str | None = None,
        root: str | PathLike[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        validate: bool = True,
        validate_files: bool = False,
        delimiter: str | None = None,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ) -> ASRDataset:
        """Load a JSON, JSON Lines, CSV, or TSV ASR manifest."""
        manifest = Path(path).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"ASR manifest does not exist: {manifest}.")
        return cls(
            _read_manifest(manifest, delimiter=delimiter),
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
    def from_audio_folder(
        cls,
        root: str | PathLike[str],
        *,
        model_type: str | None = None,
        architecture: ASRDataArchitecture | str | None = None,
        transcript_extension: str = ".txt",
        recursive: bool = True,
        metadata: Mapping[str, Any] | None = None,
        validate_files: bool = True,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ) -> ASRDataset:
        """Pair PCM WAV files with same-stem transcript sidecars.

        ``clips/utterance.wav`` is paired with ``clips/utterance.txt``
        by default. The transcript file must contain non-empty UTF-8
        text.
        """
        directory = Path(root).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"ASR audio directory does not exist: {directory}.")
        if (not isinstance(transcript_extension, str) or not transcript_extension.startswith(".") or
                len(transcript_extension) < 2):
            raise ValueError("transcript_extension must start with '.' and include a suffix.")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")
        audio_paths = sorted(directory.rglob("*.wav") if recursive else directory.glob("*.wav"))
        if not audio_paths:
            raise ValueError(f"No PCM WAV files found in {directory}.")
        records = []
        for audio_path in audio_paths:
            transcript_path = audio_path.with_suffix(transcript_extension)
            if not transcript_path.is_file():
                raise FileNotFoundError(f"Missing transcript sidecar for {audio_path}: {transcript_path}.")
            text = transcript_path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError(f"Transcript sidecar is empty: {transcript_path}.")
            record = dict(metadata or {})
            record.update({
                "audio": str(audio_path),
                "text": text,
            })
            records.append(record)
        return cls(
            records,
            model_type=model_type,
            architecture=architecture,
            root=directory,
            validate_files=validate_files,
            transform=transform,
            transform_fingerprint=transform_fingerprint,
        )

    @classmethod
    def from_kaldi(
        cls,
        root: str | PathLike[str],
        *,
        model_type: str | None = None,
        architecture: ASRDataArchitecture | str | None = None,
        wav_scp: str = "wav.scp",
        text_file: str = "text",
        metadata: Mapping[str, Any] | None = None,
        validate_files: bool = False,
        transform: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        transform_fingerprint: str | None = None,
    ) -> ASRDataset:
        """Load a simple Kaldi/ESPnet ``wav.scp`` plus ``text`` directory.

        Shell-pipeline entries are rejected because portable VoiceHub
        datasets use explicit audio files rather than executable
        commands.
        """
        directory = Path(root).expanduser().resolve()
        if not directory.is_dir():
            raise NotADirectoryError(f"ASR Kaldi directory does not exist: {directory}.")
        audio_rows = _read_key_value_file(directory / wav_scp, owner="wav.scp")
        text_rows = _read_key_value_file(directory / text_file, owner="text")
        missing_text = sorted(set(audio_rows) - set(text_rows))
        missing_audio = sorted(set(text_rows) - set(audio_rows))
        if missing_text or missing_audio:
            details = []
            if missing_text:
                details.append(f"missing text for {missing_text[0]!r}")
            if missing_audio:
                details.append(f"missing audio for {missing_audio[0]!r}")
            raise ValueError("Kaldi ASR keys do not match: " + "; ".join(details) + ".")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")
        records = []
        for key, audio in audio_rows.items():
            if audio.rstrip().endswith("|"):
                raise ValueError(
                    f"Kaldi wav.scp entry {key!r} is a shell pipeline; "
                    "materialize it as a PCM WAV file first.")
            record = dict(metadata or {})
            record.update({
                "audio": audio,
                "id": key,
                "text": text_rows[key],
            })
            records.append(record)
        return cls(
            records,
            model_type=model_type,
            architecture=architecture,
            root=directory,
            validate_files=validate_files,
            transform=transform,
            transform_fingerprint=transform_fingerprint,
        )

    @classmethod
    def coerce(
        cls,
        records_or_manifest: Any,
        *,
        model_type: str | None = None,
        architecture: ASRDataArchitecture | str | None = None,
        root: str | PathLike[str] | None = None,
        aliases: Mapping[str, str] | None = None,
        validate: bool = True,
        validate_files: bool = False,
        transform_fingerprint: str | None = None,
    ) -> ASRDataset:
        """Normalize an existing dataset, manifest path, or record iterable."""
        if isinstance(records_or_manifest, cls):
            requested_spec = (
                records_or_manifest.spec if model_type is None and architecture is None else
                get_asr_dataset_spec(model_type, architecture=architecture))
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
    ) -> tuple[ASRDataset, ASRDataset]:
        """Create deterministic, optionally speaker/session-disjoint splits."""
        if (isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)) or
                not 0.0 < float(validation_fraction) < 1.0):
            raise ValueError("validation_fraction must be between 0 and 1.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if len(self) < 2:
            raise ValueError("At least two ASR records are required for a split.")

        randomizer = random.Random(seed)
        if group_by is None:
            indices = list(range(len(self)))
            randomizer.shuffle(indices)
            validation_count = max(
                1,
                min(
                    len(indices) - 1,
                    round(len(indices) * float(validation_fraction)),
                ),
            )
            validation_indices = set(indices[:validation_count])
        else:
            if not isinstance(group_by, str) or not group_by.strip():
                raise ValueError("group_by must be a non-empty field name or None.")
            groups: dict[str, list[int]] = {}
            for index, record in enumerate(self._records):
                if not _field_present(record, group_by):
                    raise ValueError(f"ASR record {index} is missing split group field "
                                     f"{group_by!r}.")
                groups.setdefault(str(record[group_by]), []).append(index)
            names = sorted(groups)
            if len(names) < 2:
                raise ValueError(f"At least two {group_by!r} groups are required.")
            randomizer.shuffle(names)
            validation_group_count = max(
                1,
                min(
                    len(names) - 1,
                    round(len(names) * float(validation_fraction)),
                ),
            )
            validation_indices = {index for name in names[:validation_group_count] for index in groups[name]}

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
            type(self)(
                [record for index, record in enumerate(self._records) if index not in validation_indices],
                **options,
            ),
            type(self)(
                [record for index, record in enumerate(self._records) if index in validation_indices],
                **options,
            ),
        )

    def to_jsonl(
        self,
        path: str | PathLike[str],
        *,
        relative_to: str | PathLike[str] | None = None,
    ) -> Path:
        """Write a portable JSON Lines manifest and return its path."""
        destination = Path(path).expanduser().resolve()
        relative_root = (
            destination.parent if relative_to is None else Path(relative_to).expanduser().resolve())
        lines: list[str] = []
        for index, record in enumerate(self._records):
            serializable = self._portable_record(
                record,
                relative_to=relative_root,
            )
            try:
                lines.append(
                    json.dumps(
                        serializable,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ) + "\n")
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"ASR record {index} contains values that cannot be "
                    "written to JSON Lines. Store tensors/features "
                    "separately and put their paths in the manifest.") from exc
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(lines)
        return destination

    def resume_fingerprint(self) -> dict[str, Any]:
        """Return a stable content identity for exact Trainer resumes."""
        if self.transform is not None and self.transform_fingerprint is None:
            raise ValueError(
                "ASRDataset transforms require an explicit "
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
            "architecture": self.architecture.value,
            "content_sha256": hashlib.sha256(canonical).hexdigest(),
            "length": len(self),
            "model_type": self.model_type,
            "transform_fingerprint": self.transform_fingerprint,
        }

    @property
    def requires_homogeneous_batches(self) -> bool:
        return bool(self.spec.homogeneous_batch_fields)

    def batch_group_key(self, record: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the model-specific metadata key used for safe batching."""
        values = []
        for alternatives in self.spec.homogeneous_batch_fields:
            selected = None
            for name in alternatives:
                if _field_present(record, name):
                    selected = record[name]
                    break
            if selected is None and alternatives == ("punctuation", ):
                selected = True
            values.append(_fingerprint_value(selected))
        return tuple(values)

    def create_batch_sampler(
        self,
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> EpochGroupedBatchSampler | None:
        """Create a deterministic safe-batch sampler when the model needs
        one."""
        if not self.requires_homogeneous_batches:
            return None
        return EpochGroupedBatchSampler(
            self,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            drop_last=drop_last,
        )

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
                raise ValueError("ASR dataset alias names must be non-empty strings.")
            if not isinstance(target, str) or not target.strip():
                raise ValueError("ASR dataset alias targets must be non-empty strings.")
            source = source.strip()
            target = target.strip()
            if source != target:
                normalized[source] = target
        return normalized

    def _normalize_record(
        self,
        record: Mapping[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        value = dict(record)
        if self.model_type == "asr_seamless_m4t_v2":
            value = self._normalize_seamless_record(value, index=index)
        for source, target in self.aliases.items():
            if source not in value:
                continue
            if target in value:
                raise ValueError(
                    f"ASR record {index} contains both alias {source!r} and "
                    f"canonical field {target!r}.")
            value[target] = value.pop(source)
        if self.model_type == "asr_funasr":
            value = self._normalize_sensevoice_record(value, index=index)

        for name in _PATH_FIELDS:
            if name in value:
                value[name] = self._normalize_audio_value(
                    value[name],
                    index=index,
                    field_name=name,
                )
        text = value.get("text")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError(f"ASR record {index} field 'text' must be a non-empty string.")
        for name in ("language", "target_language"):
            item = value.get(name)
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f"ASR record {index} field {name!r} must be a non-empty string.")
        sampling_rate = value.get("sampling_rate")
        if isinstance(sampling_rate, str):
            try:
                sampling_rate = int(sampling_rate)
            except ValueError:
                pass
            else:
                value["sampling_rate"] = sampling_rate
        if sampling_rate is not None and (isinstance(sampling_rate, bool) or
                                          not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError(f"ASR record {index} field 'sampling_rate' must be a positive integer.")
        duration = value.get("duration")
        if isinstance(duration, str):
            try:
                duration = float(duration)
            except ValueError:
                pass
            else:
                value["duration"] = duration
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or
                                     not math.isfinite(float(duration)) or float(duration) <= 0.0):
            raise ValueError(f"ASR record {index} field 'duration' must be a finite positive number.")
        return value

    @staticmethod
    def _normalize_sensevoice_record(
        value: dict[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        """Translate the official SenseVoice JSONL control-token spellings."""

        def control_name(name: str) -> None:
            item = value.get(name)
            if item is None or not isinstance(item, str):
                return
            normalized = item.strip()
            prefix, suffix = _SENSEVOICE_CONTROL_WRAPPERS
            if normalized.startswith(prefix) and normalized.endswith(suffix):
                normalized = normalized[len(prefix):-len(suffix)]
            normalized = normalized.strip().lower()
            aliases = {
                "emo_unknown": "unknown",
                "event_unk": "unknown",
            }
            value[name] = aliases.get(normalized, normalized)

        for field_name in ("language", "emotion", "event"):
            control_name(field_name)

        use_itn = value.get("use_itn")
        if not isinstance(use_itn, str):
            return value
        normalized_itn = use_itn.strip()
        prefix, suffix = _SENSEVOICE_CONTROL_WRAPPERS
        if normalized_itn.startswith(prefix) and normalized_itn.endswith(suffix):
            normalized_itn = normalized_itn[len(prefix):-len(suffix)]
        normalized_itn = normalized_itn.strip().lower().replace("_", "")
        if normalized_itn in {"withitn", "true", "1", "yes"}:
            value["use_itn"] = True
        elif normalized_itn in {"woitn", "withoutitn", "false", "0", "no"}:
            value["use_itn"] = False
        else:
            raise ValueError(
                f"ASR record {index} field 'use_itn' must be a boolean or a "
                "SenseVoice <|withitn|>/<|woitn|> control token.")
        return value

    @staticmethod
    def _normalize_seamless_record(
        value: dict[str, Any],
        *,
        index: int,
    ) -> dict[str, Any]:
        """Flatten the official SeamlessM4T source/target manifest shape."""
        source = value.get("source")
        target = value.get("target")
        if not isinstance(source, Mapping) and not isinstance(target, Mapping):
            return value
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise TypeError(f"ASR record {index} Seamless `source` and `target` must both "
                            "be mappings.")
        extracted = {
            "audio":
            next(
                (
                    source[name] for name in (
                        "audio_local_path",
                        "audio",
                        "audio_path",
                        "audio_filepath",
                    ) if _field_present(source, name)),
                None,
            ),
            "sampling_rate":
            source.get("sampling_rate", source.get("sample_rate")),
            "source_language":
            source.get("lang", source.get("language")),
            "target_language":
            target.get("lang", target.get("language")),
            "text":
            target.get("text"),
        }
        flattened = {name: item for name, item in extracted.items() if item is not None}
        for name in flattened:
            if name in value:
                raise ValueError(
                    f"ASR record {index} contains both the official Seamless "
                    f"nested value and canonical field {name!r}.")
        value.update(flattened)
        value.pop("source")
        value.pop("target")
        return value

    def _normalize_audio_value(
        self,
        field_value: Any,
        *,
        index: int,
        field_name: str,
    ) -> Any:
        if isinstance(field_value, Mapping):
            nested = dict(field_value)
            if "path" in nested:
                nested["path"] = self._normalize_audio_value(
                    nested["path"],
                    index=index,
                    field_name=f"{field_name}.path",
                )
            return nested
        if isinstance(field_value, (list, tuple)):
            if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in field_value):
                return field_value
            normalized = [
                self._normalize_audio_value(
                    item,
                    index=index,
                    field_name=f"{field_name}[{item_index}]",
                ) for item_index, item in enumerate(field_value)
            ]
            return tuple(normalized) if isinstance(field_value, tuple) else normalized
        if not isinstance(field_value, (str, PathLike)):
            return field_value
        if isinstance(field_value, str) and not field_value.strip():
            raise ValueError(f"ASR record {index} field {field_name!r} must be non-empty.")
        path = Path(field_value).expanduser()
        if not path.is_absolute() and self.root is not None:
            path = self.root / path
        if path.is_absolute():
            path = path.resolve()
        if self.validate_files and not path.is_file():
            raise FileNotFoundError(f"ASR record {index} field {field_name!r} does not exist: {path}.")
        return str(path)

    @staticmethod
    def _portable_record(
        record: Mapping[str, Any],
        *,
        relative_to: Path,
    ) -> dict[str, Any]:
        value = dict(record)
        for name in _PATH_FIELDS:
            if name in value:
                value[name] = ASRDataset._portable_audio_value(
                    value[name],
                    relative_to=relative_to,
                )
        return value

    @staticmethod
    def _portable_audio_value(field_value: Any, *, relative_to: Path) -> Any:
        if isinstance(field_value, Mapping):
            nested = dict(field_value)
            if "path" in nested:
                nested["path"] = ASRDataset._portable_audio_value(
                    nested["path"],
                    relative_to=relative_to,
                )
            return nested
        if isinstance(field_value, (list, tuple)):
            normalized = [
                ASRDataset._portable_audio_value(
                    item,
                    relative_to=relative_to,
                ) for item in field_value
            ]
            return tuple(normalized) if isinstance(field_value, tuple) else normalized
        if not isinstance(field_value, (str, PathLike)):
            return field_value
        path = Path(field_value).expanduser()
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.relative_to(relative_to).as_posix()
        except ValueError:
            return str(path)


def _read_manifest(path: Path, *, delimiter: str | None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        selected_delimiter = delimiter
        if selected_delimiter is None:
            selected_delimiter = "\t" if suffix == ".tsv" else ","
        if not isinstance(selected_delimiter, str) or len(selected_delimiter) != 1:
            raise ValueError("CSV/TSV delimiter must be exactly one character.")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=selected_delimiter)
            if not reader.fieldnames:
                raise ValueError(f"ASR tabular manifest {path} has no header.")
            records = [{
                key: _coerce_tabular_value(value)
                for key, value in row.items() if key is not None and value not in (None, "")
            } for row in reader]
    elif suffix == ".json":
        source = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(source)
        except json.JSONDecodeError:
            records = _read_json_lines(path)
        else:
            if isinstance(payload, Mapping):
                records = [dict(payload)]
            elif isinstance(payload, list):
                records = payload
            else:
                raise TypeError(f"ASR JSON manifest {path} must contain an object or list of objects.")
    else:
        records = _read_json_lines(path)
    if not records:
        raise ValueError(f"No ASR records found in manifest {path}.")
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"ASR manifest record {index} in {path} must be an object.")
        normalized.append(dict(record))
    return normalized


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid ASR JSON Lines record at {path}:{line_number}: "
                    f"{exc.msg}.") from exc
            if not isinstance(record, Mapping):
                raise TypeError(f"ASR JSON Lines record at {path}:{line_number} must be an object.")
            records.append(dict(record))
    return records


def _read_key_value_file(path: Path, *, owner: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"ASR {owner} file does not exist: {path}.")
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) != 2 or not parts[1].strip():
                raise ValueError(f"Invalid ASR {owner} entry at {path}:{line_number}.")
            key, value = parts
            if key in records:
                raise ValueError(f"Duplicate ASR {owner} key {key!r} at {path}:{line_number}.")
            records[key] = value.strip()
    if not records:
        raise ValueError(f"ASR {owner} file is empty: {path}.")
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
                "ASRDataset resume fingerprints require string mapping "
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
        return {"__set__": items}
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
        "ASRDataset resume fingerprints support JSON values, paths, sets, "
        "tensors, and array-like values; received unsupported "
        f"{type(value).__module__}.{type(value).__qualname__}.")


__all__ = ["ASRDataset", "EpochGroupedBatchSampler"]
