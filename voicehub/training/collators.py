"""Schema-aware collation for token, acoustic, and waveform batches."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from voicehub.dependencies import import_optional


@dataclass(frozen=True)
class AudioFieldSchema:
    """Describe the variable-length dimension of one training field.

    ``length_field`` and ``mask_field`` are written beside the source
    field unless they contain a dot, in which case they are interpreted
    from the batch root. A mask is always shaped ``(batch,
    padded_sequence_length)``.
    """

    sequence_dim: int = 0
    padding_value: float | int | None = None
    padding_side: str = "right"
    length_field: str | None = None
    mask_field: str | None = None
    pad_to_multiple_of: int | None = None
    allow_missing: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.sequence_dim, bool) or not isinstance(self.sequence_dim, int):
            raise TypeError("sequence_dim must be an integer.")
        if self.padding_side not in ("left", "right"):
            raise ValueError("padding_side must be either 'left' or 'right'.")
        for name in ("length_field", "mask_field"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty path or None.")
        multiple = self.pad_to_multiple_of
        if multiple is not None and (isinstance(multiple, bool) or not isinstance(multiple, int) or
                                     multiple <= 0):
            raise ValueError("pad_to_multiple_of must be a positive integer or None.")
        if not isinstance(self.allow_missing, bool):
            raise TypeError("allow_missing must be a boolean.")


@dataclass
class DataCollatorForAudioTraining:
    """Collate heterogeneous audio examples without guessing task semantics.

    Equal-shaped tensors are stacked. Variable token sequences and
    common time-major/time-last acoustic values retain the historical
    inference behavior. Ambiguous fields should be declared in
    ``field_schemas`` using dotted paths such as ``"model_inputs.mel"``.
    """

    padding_value: float = 0.0
    label_pad_token_id: int = -100
    return_attention_mask: bool = True
    return_input_lengths: bool = False
    field_schemas: Mapping[str, AudioFieldSchema | Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        schemas = {}
        for path, schema in dict(self.field_schemas or {}).items():
            if not isinstance(path, str) or not path.strip():
                raise ValueError("field_schemas keys must be non-empty dotted paths.")
            if isinstance(schema, Mapping):
                schema = AudioFieldSchema(**dict(schema))
            if not isinstance(schema, AudioFieldSchema):
                raise TypeError("field_schemas values must be AudioFieldSchema instances "
                                "or mappings.")
            schemas[path.strip()] = schema
        self.field_schemas = schemas

    def resume_fingerprint(self) -> dict[str, Any]:
        """Return every collation option that can change a resumed batch."""
        return {
            "padding_value": self.padding_value,
            "label_pad_token_id": self.label_pad_token_id,
            "return_attention_mask": self.return_attention_mask,
            "return_input_lengths": self.return_input_lengths,
            "field_schemas": {
                path: {
                    field.name: getattr(schema, field.name)
                    for field in fields(AudioFieldSchema)
                }
                for path, schema in sorted(self.field_schemas.items())
            },
        }

    def __call__(self, features: list[Mapping[str, Any] | Any]) -> dict[str, Any]:
        if not features:
            raise ValueError("Audio training collators require a non-empty batch.")
        normalized = [self._as_mapping(feature) for feature in features]
        derived = []
        batch = self._collate_mapping(
            normalized,
            path=(),
            derived=derived,
        )
        generated_paths = set()
        for target_path, value in derived:
            if target_path in generated_paths:
                raise ValueError(
                    "Multiple field schemas derive the same batch field "
                    f"{'.'.join(target_path)!r}. Give each source field a "
                    "distinct length_field and mask_field.")
            generated_paths.add(target_path)
            if any(self._path_exists(feature, target_path) for feature in normalized):
                self._validate_derived_field(
                    normalized,
                    target_path,
                    value,
                )
            # Canonicalize valid per-example values to the padding policy
            # used by the source field. This preserves left padding and
            # pad-to-multiple expansion instead of independently padding a
            # caller-supplied mask.
            self._set_path(batch, target_path, value)
        return batch

    @staticmethod
    def _as_mapping(value) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if is_dataclass(value) and not isinstance(value, type):
            return {field.name: getattr(value, field.name) for field in fields(value)}
        raise TypeError("Audio training samples must be mappings or dataclass instances.")

    @staticmethod
    def _import_torch():
        return import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )

    def _collate_mapping(self, features, *, path, derived):
        batch = {}
        keys = tuple(dict.fromkeys(key for feature in features for key in feature))
        for key in keys:
            values = [feature.get(key) for feature in features]
            current_path = path + (str(key), )
            if current_path[-1] == "training_phase":
                batch[key] = self._collate_control(current_path, values)
                continue
            if all(value is None for value in values):
                continue

            present = [value for value in values if value is not None]
            if (self._has_descendant_schema(current_path) and
                    all(isinstance(value, Mapping) or is_dataclass(value) and not isinstance(value, type)
                        for value in present)):
                batch[key] = self._collate_mapping(
                    [{} if value is None else self._as_mapping(value) for value in values],
                    path=current_path,
                    derived=derived,
                )
                continue
            if any(value is None for value in values):
                schema = self._schema_for(current_path)
                if schema is None:
                    batch[key] = values
                    continue
                if not schema.allow_missing:
                    raise ValueError(
                        f"Field {'.'.join(current_path)!r} is missing from some samples. "
                        "Set allow_missing=True in its AudioFieldSchema to pad missing "
                        "values as zero-length sequences.")
            elif all(isinstance(value, Mapping) or is_dataclass(value) and not isinstance(value, type)
                     for value in values):
                batch[key] = self._collate_mapping(
                    [self._as_mapping(value) for value in values],
                    path=current_path,
                    derived=derived,
                )
                continue

            schema = self._schema_for(current_path)
            collated, lengths, padded_length = self._collate_values(
                current_path,
                values,
                schema,
            )
            batch[key] = collated
            if lengths is not None and schema is not None:
                self._append_derived_fields(
                    current_path,
                    schema,
                    lengths,
                    padded_length,
                    derived,
                    device=getattr(collated, "device", None),
                )
        return batch

    @staticmethod
    def _collate_control(path, values):
        if any(value is None for value in values):
            raise ValueError(f"Every sample in a batch must provide {'.'.join(path)!r}.")
        first = values[0]
        for value in values[1:]:
            try:
                equal = value is first or value == first
                if hasattr(equal, "numel"):
                    if equal.numel() != 1:
                        equal = False
                    else:
                        equal = bool(equal.item())
                else:
                    equal = bool(equal)
            except (RuntimeError, TypeError, ValueError):
                equal = False
            if not equal:
                raise ValueError("Every sample in a batch must select the same training_phase.")
        return first

    def _has_descendant_schema(self, path: tuple[str, ...]) -> bool:
        prefix = ".".join(path) + "."
        return any(schema_path.startswith(prefix) for schema_path in self.field_schemas)

    def _schema_for(self, path: tuple[str, ...]) -> AudioFieldSchema | None:
        dotted = ".".join(path)
        configured = self.field_schemas.get(dotted)

        key = path[-1]
        if key == "input_ids":
            defaults = AudioFieldSchema(
                sequence_dim=0,
                padding_value=0,
                length_field=("input_lengths" if self.return_input_lengths else None),
                mask_field=("attention_mask" if self.return_attention_mask else None),
            )
            if configured is None:
                return defaults
            return replace(
                configured,
                length_field=(configured.length_field or defaults.length_field),
                mask_field=(configured.mask_field or defaults.mask_field),
            )
        if configured is not None:
            return configured
        if key in ("labels", "label_ids"):
            return AudioFieldSchema(sequence_dim=0, )
        return None

    def _collate_values(self, path, values, schema):
        torch = self._import_torch()
        present = [value for value in values if value is not None]
        first = present[0]
        if isinstance(first, str):
            return values, None, None
        if isinstance(first, (int, float, bool)):
            if len(present) != len(values):
                return values, None, None
            return torch.tensor(values), None, None

        tensors = self._as_tensors(present)
        if tensors is None:
            return values, None, None
        if len(present) != len(values):
            sequence_dim = self._normalize_sequence_dim(
                schema.sequence_dim,
                tensors[0].ndim,
                path,
            )
            tensor_iterator = iter(tensors)
            reference = tensors[0]
            materialized = []
            for value in values:
                if value is None:
                    shape = list(reference.shape)
                    shape[sequence_dim] = 0
                    materialized.append(reference.new_empty(shape))
                else:
                    materialized.append(next(tensor_iterator))
            tensors = materialized
        if all(tensor.ndim == 0 for tensor in tensors):
            return torch.stack(tensors), None, None
        if not all(tensor.ndim == tensors[0].ndim for tensor in tensors):
            if schema is not None:
                raise ValueError(f"Field {'.'.join(path)!r} has tensors with different ranks.")
            return values, None, None

        if schema is not None:
            sequence_dim = self._normalize_sequence_dim(
                schema.sequence_dim,
                tensors[0].ndim,
                path,
            )
            return self._pad_tensors(
                path,
                tensors,
                sequence_dim=sequence_dim,
                padding_value=self._padding_for(path, tensors[0], schema),
                padding_side=schema.padding_side,
                pad_to_multiple_of=schema.pad_to_multiple_of,
                strict=True,
            )

        if all(tuple(tensor.shape) == tuple(tensors[0].shape) for tensor in tensors):
            return torch.stack(tensors), [int(tensors[0].shape[0])] * len(tensors), int(tensors[0].shape[0])

        tail_shape = tuple(tensors[0].shape[1:])
        if all(tuple(tensor.shape[1:]) == tail_shape for tensor in tensors):
            return self._pad_tensors(
                path,
                tensors,
                sequence_dim=0,
                padding_value=self._padding_for(path, tensors[0], None),
                padding_side="right",
                pad_to_multiple_of=None,
                strict=False,
            )

        leading_shape = tuple(tensors[0].shape[:-1])
        if all(tuple(tensor.shape[:-1]) == leading_shape for tensor in tensors):
            return self._pad_tensors(
                path,
                tensors,
                sequence_dim=tensors[0].ndim - 1,
                padding_value=self._padding_for(path, tensors[0], None),
                padding_side="right",
                pad_to_multiple_of=None,
                strict=False,
            )
        return values, None, None

    @staticmethod
    def _normalize_sequence_dim(sequence_dim, rank, path):
        normalized = sequence_dim + rank if sequence_dim < 0 else sequence_dim
        if not 0 <= normalized < rank:
            raise ValueError(
                f"Field {'.'.join(path)!r} uses sequence_dim={sequence_dim} "
                f"for rank-{rank} tensors.")
        return normalized

    def _padding_for(self, path, tensor, schema):
        if schema is not None and schema.padding_value is not None:
            return schema.padding_value
        if (path[-1] in ("labels", "label_ids") and not tensor.is_floating_point()):
            return self.label_pad_token_id
        return self.padding_value

    def _pad_tensors(
        self,
        path,
        tensors,
        *,
        sequence_dim,
        padding_value,
        padding_side,
        pad_to_multiple_of,
        strict,
    ):
        torch = self._import_torch()
        reference_shape = tuple(tensors[0].shape)
        for tensor in tensors[1:]:
            for dimension, (actual, expected) in enumerate(zip(tensor.shape, reference_shape)):
                if dimension != sequence_dim and actual != expected:
                    if strict:
                        raise ValueError(
                            f"Field {'.'.join(path)!r} differs outside its "
                            f"declared sequence dimension {sequence_dim}.")
                    return tensors, None, None

        lengths = [int(tensor.shape[sequence_dim]) for tensor in tensors]
        padded_length = max(lengths)
        if pad_to_multiple_of is not None:
            padded_length = ((padded_length + pad_to_multiple_of - 1) // pad_to_multiple_of *
                             pad_to_multiple_of)
        padded = []
        for tensor, length in zip(tensors, lengths):
            target_shape = list(tensor.shape)
            target_shape[sequence_dim] = padded_length
            output_dtype = self._padding_dtype(tensor, padding_value, torch)
            output = torch.full(
                target_shape,
                padding_value,
                dtype=output_dtype,
                device=tensor.device,
            )
            start = padded_length - length if padding_side == "left" else 0
            slices = [slice(None)] * tensor.ndim
            slices[sequence_dim] = slice(start, start + length)
            output[tuple(slices)] = tensor
            padded.append(output)
        return torch.stack(padded), lengths, padded_length

    @staticmethod
    def _padding_dtype(tensor, padding_value, torch):
        if tensor.dtype == torch.bool:
            if padding_value not in (False, True, 0, 1):
                return torch.long
            return tensor.dtype
        if not tensor.is_floating_point() and not tensor.is_complex():
            try:
                limits = torch.iinfo(tensor.dtype)
            except TypeError:
                return tensor.dtype
            if padding_value < limits.min or padding_value > limits.max:
                return torch.long
        return tensor.dtype

    def _append_derived_fields(
        self,
        source_path,
        schema,
        lengths,
        padded_length,
        derived,
        *,
        device,
    ):
        torch = self._import_torch()
        length_tensor = torch.tensor(lengths, dtype=torch.long, device=device)
        if schema.length_field is not None:
            derived.append((
                self._derived_path(source_path, schema.length_field),
                length_tensor,
            ))
        if schema.mask_field is not None:
            positions = torch.arange(
                padded_length,
                device=device,
            ).unsqueeze(0)
            if schema.padding_side == "left":
                mask = positions >= (padded_length - length_tensor.unsqueeze(1))
            else:
                mask = positions < length_tensor.unsqueeze(1)
            derived.append((
                self._derived_path(source_path, schema.mask_field),
                mask,
            ))

    @staticmethod
    def _derived_path(source_path, target):
        if "." in target:
            return tuple(target.split("."))
        return source_path[:-1] + (target, )

    @staticmethod
    def _path_exists(mapping, path):
        current = mapping
        for part in path:
            if not isinstance(current, Mapping) or part not in current:
                return False
            current = current[part]
        return True

    def _validate_derived_field(self, features, path, expected):
        torch = self._import_torch()
        for index, feature in enumerate(features):
            if not self._path_exists(feature, path):
                raise ValueError(
                    f"Every sample must provide derived field "
                    f"{'.'.join(path)!r} when any sample provides it.")
            raw_value = self._get_path(feature, path)
            try:
                if isinstance(raw_value, torch.Tensor):
                    actual = raw_value.detach().to(device=expected.device)
                else:
                    actual = torch.as_tensor(
                        raw_value,
                        device=expected.device,
                    )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(f"Derived field {'.'.join(path)!r} must be tensor-like.") from exc

            if expected.ndim == 1:
                if (actual.ndim != 0 or actual.dtype == torch.bool or actual.is_floating_point() or
                        actual.is_complex()):
                    raise TypeError(
                        f"Derived length field {'.'.join(path)!r} must contain "
                        "integer scalars.")
                matches = int(actual.item()) == int(expected[index].item())
            elif expected.ndim == 2:
                if actual.ndim != 1 or actual.is_complex():
                    raise ValueError(
                        f"Derived mask field {'.'.join(path)!r} must contain "
                        "one rank-1 mask per sample.")
                binary = (actual == 0) | (actual == 1)
                if not bool(binary.all().item()):
                    raise ValueError(
                        f"Derived mask field {'.'.join(path)!r} must contain "
                        "only boolean or 0/1 values.")
                actual = actual.bool()
                expected_row = expected[index]
                if actual.numel() == expected_row.numel():
                    matches = torch.equal(actual, expected_row)
                elif actual.numel() == int(expected_row.sum().item()):
                    matches = torch.equal(
                        actual,
                        expected_row.masked_select(expected_row),
                    )
                else:
                    matches = False
            else:
                raise RuntimeError(
                    "Derived audio fields must be batch lengths or "
                    "batch-by-sequence masks.")
            if not matches:
                raise ValueError(
                    f"Provided derived field {'.'.join(path)!r} for sample "
                    f"{index} does not match the lengths or padding mask "
                    "computed from its source field.")

    @staticmethod
    def _get_path(mapping, path):
        current = mapping
        for part in path:
            current = current[part]
        return current

    @staticmethod
    def _set_path(mapping, path, value):
        current = mapping
        for part in path[:-1]:
            child = current.get(part)
            if child is None:
                child = {}
                current[part] = child
            elif not isinstance(child, dict):
                raise ValueError(
                    f"Cannot create derived field {'.'.join(path)!r}; "
                    f"{part!r} is not a mapping.")
            current = child
        current[path[-1]] = value

    def _as_tensors(self, values):
        torch = self._import_torch()
        if all(isinstance(value, torch.Tensor) for value in values):
            return values
        if all(hasattr(value, "dtype") and hasattr(value, "shape") for value in values):
            try:
                return [torch.as_tensor(value) for value in values]
            except (TypeError, ValueError, RuntimeError):
                return None
        if all(isinstance(value, (list, tuple)) for value in values):
            try:
                return [torch.tensor(value) for value in values]
            except (TypeError, ValueError):
                return None
        return None


# Backward-compatible TTS spellings. These are exact aliases so existing
# imports, ``isinstance`` checks, and serialized references retain identical
# behavior while new speech tasks can use task-neutral public names.
TTSFieldSchema = AudioFieldSchema
DataCollatorForTTSTraining = DataCollatorForAudioTraining

__all__ = [
    "AudioFieldSchema",
    "DataCollatorForAudioTraining",
    "DataCollatorForTTSTraining",
    "TTSFieldSchema",
]
