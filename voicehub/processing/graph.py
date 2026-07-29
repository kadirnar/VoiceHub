"""Serializable, audited preprocessing graphs.

Graphs contain registered operations rather than arbitrary Python callbacks.
Artifacts can therefore describe their processor exactly without executing
repository code or serializing callables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Any, ClassVar


def _names(
    values: Iterable[str],
    *,
    field_name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    result = tuple(values)
    if not result and not allow_empty:
        raise ValueError(f"`{field_name}` must not be empty.")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"`{field_name}` must contain non-empty strings.")
    if len(result) != len(set(result)):
        raise ValueError(f"`{field_name}` cannot contain duplicates.")
    return result


class ProcessingOperation(ABC):
    """One versioned, serializable processor transformation."""

    operation_id: ClassVar[str]
    operation_version: ClassVar[str] = "1"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("operation_id", "operation_version"):
            value = getattr(cls, name, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"Processing operation {cls.__name__} must declare "
                    f"a non-empty `{name}`."
                )

    @property
    @abstractmethod
    def inputs(self) -> tuple[str, ...]:
        """Keys read from the processing context."""

    @property
    @abstractmethod
    def outputs(self) -> tuple[str, ...]:
        """Keys written to the processing context."""

    @abstractmethod
    def process(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """Apply the operation to an immutable view of available values."""

    def to_config(self) -> Mapping[str, Any]:
        """Return finite JSON data needed to rebuild this operation."""
        return {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ProcessingOperation":
        if not isinstance(config, Mapping):
            raise TypeError("Processing operation config must be a mapping.")
        return cls(**dict(config))

    def descriptor(self) -> dict[str, Any]:
        return {
            "operation": self.operation_id,
            "version": self.operation_version,
            "config": dict(self.to_config()),
        }


class ProcessingOperationRegistry:
    """Thread-safe registry used while loading serialized processor graphs."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._operations: dict[tuple[str, str], type[ProcessingOperation]] = {}
        self._view = MappingProxyType(self._operations)

    @property
    def operations(
        self,
    ) -> Mapping[tuple[str, str], type[ProcessingOperation]]:
        return self._view

    def register(
        self,
        operation: type[ProcessingOperation],
        *,
        exist_ok: bool = False,
    ) -> None:
        if not isinstance(operation, type) or not issubclass(
            operation,
            ProcessingOperation,
        ):
            raise TypeError("Only ProcessingOperation classes can be registered.")
        key = (operation.operation_id, operation.operation_version)
        with self._lock:
            if key in self._operations and not exist_ok:
                raise ValueError(
                    f"Processing operation {key[0]}@{key[1]} is already registered."
                )
            self._operations[key] = operation

    def create(
        self,
        operation_id: str,
        version: str,
        config: Mapping[str, Any],
    ) -> ProcessingOperation:
        key = (operation_id, version)
        with self._lock:
            operation = self._operations.get(key)
        if operation is None:
            raise LookupError(
                f"Unknown processing operation {operation_id}@{version}."
            )
        return operation.from_config(config)


PROCESSING_OPERATIONS = ProcessingOperationRegistry()


@dataclass(frozen=True)
class ProcessorGraph:
    """Validated directed pipeline with explicit public inputs and outputs."""

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    operations: tuple[ProcessingOperation, ...]
    graph_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "inputs",
            _names(self.inputs, field_name="inputs"),
        )
        object.__setattr__(
            self,
            "outputs",
            _names(self.outputs, field_name="outputs"),
        )
        operations = tuple(self.operations)
        if any(not isinstance(item, ProcessingOperation) for item in operations):
            raise TypeError(
                "`operations` must contain ProcessingOperation instances."
            )
        object.__setattr__(self, "operations", operations)
        if (
            isinstance(self.graph_version, bool)
            or not isinstance(self.graph_version, int)
            or self.graph_version <= 0
        ):
            raise ValueError("`graph_version` must be a positive integer.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("`metadata` must be a mapping.")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

        available = set(self.inputs)
        for operation in operations:
            operation_inputs = _names(
                operation.inputs,
                field_name=f"{operation.operation_id}.inputs",
                allow_empty=True,
            )
            operation_outputs = _names(
                operation.outputs,
                field_name=f"{operation.operation_id}.outputs",
            )
            missing = sorted(set(operation_inputs) - available)
            if missing:
                raise ValueError(
                    f"Processing operation {operation.operation_id!r} reads "
                    f"unavailable keys: {missing!r}."
                )
            collisions = sorted(set(operation_outputs) & available)
            if collisions:
                raise ValueError(
                    f"Processing operation {operation.operation_id!r} "
                    f"overwrites existing keys: {collisions!r}."
                )
            available.update(operation_outputs)
        missing_outputs = sorted(set(self.outputs) - available)
        if missing_outputs:
            raise ValueError(
                f"Processor graph outputs are never produced: "
                f"{missing_outputs!r}."
            )

    def run(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Run all operations and return only declared graph outputs."""
        if not isinstance(values, Mapping):
            raise TypeError("Processor graph input must be a mapping.")
        missing = sorted(set(self.inputs) - set(values))
        unexpected = sorted(set(values) - set(self.inputs))
        if missing or unexpected:
            raise ValueError(
                "Processor graph input mismatch: "
                f"missing={missing!r}, unexpected={unexpected!r}."
            )
        context = dict(values)
        for operation in self.operations:
            result = operation.process(MappingProxyType(context))
            if not isinstance(result, Mapping):
                raise TypeError(
                    f"Processing operation {operation.operation_id!r} must "
                    "return a mapping."
                )
            if set(result) != set(operation.outputs):
                raise ValueError(
                    f"Processing operation {operation.operation_id!r} returned "
                    f"{sorted(result)!r}; expected "
                    f"{sorted(operation.outputs)!r}."
                )
            context.update(result)
        return {name: context[name] for name in self.outputs}

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_version": self.graph_version,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "operations": [
                operation.descriptor()
                for operation in self.operations
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        registry: ProcessingOperationRegistry = PROCESSING_OPERATIONS,
    ) -> "ProcessorGraph":
        if not isinstance(value, Mapping):
            raise TypeError("Serialized processor graph must be a mapping.")
        descriptors = value.get("operations")
        if not isinstance(descriptors, list):
            raise ValueError("Serialized processor graph needs an operations list.")
        operations = []
        for descriptor in descriptors:
            if not isinstance(descriptor, Mapping):
                raise ValueError("Processing operation descriptor must be a mapping.")
            try:
                operations.append(
                    registry.create(
                        descriptor["operation"],
                        descriptor["version"],
                        descriptor.get("config", {}),
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"Processing operation descriptor is missing {error.args[0]!r}."
                ) from error
        return cls(
            graph_version=value.get("graph_version", 1),
            inputs=tuple(value.get("inputs", ())),
            outputs=tuple(value.get("outputs", ())),
            operations=tuple(operations),
            metadata=value.get("metadata", {}),
        )
