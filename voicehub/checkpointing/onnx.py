"""Small, dependency-free reader for trusted ONNX model containers.

This module reads the bounded subset of the ONNX protobuf schema needed
by explicit checkpoint converters and VoiceHub's allowlisted native
graph runtime.  It uses only the Python standard library and rejects
external tensor data, protobuf groups, malformed lengths, and
unsupported tensor encodings.

The reader deliberately exposes graph structure and attributes as
immutable data. Architecture converters remain responsible for
validating the exact operator graph, tensor namespace, shapes, dtypes,
provenance, and source digest before writing a safe internal artifact or
constructing a runtime.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from voicehub.checkpointing.errors import CheckpointFormatError

_WIRE_VARINT = 0
_WIRE_FIXED64 = 1
_WIRE_BYTES = 2
_WIRE_FIXED32 = 5
_MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_FIELD_BYTES = 1024 * 1024 * 1024
_MAX_FIELDS = 10_000_000


def _read_varint(data: memoryview, offset: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise CheckpointFormatError("Truncated protobuf varint.")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
    raise CheckpointFormatError("Protobuf varint exceeds 64 bits.")


def _signed_int64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def _fields(data: bytes | memoryview) -> Iterator[tuple[int, int, Any]]:
    view = data if isinstance(data, memoryview) else memoryview(data)
    offset = 0
    count = 0
    while offset < len(view):
        count += 1
        if count > _MAX_FIELDS:
            raise CheckpointFormatError("Protobuf message contains too many fields.")
        key, offset = _read_varint(view, offset)
        field_number = key >> 3
        wire_type = key & 7
        if field_number == 0:
            raise CheckpointFormatError("Protobuf field number zero is invalid.")
        if wire_type == _WIRE_VARINT:
            value, offset = _read_varint(view, offset)
        elif wire_type == _WIRE_FIXED64:
            end = offset + 8
            if end > len(view):
                raise CheckpointFormatError("Truncated protobuf fixed64 field.")
            value = bytes(view[offset:end])
            offset = end
        elif wire_type == _WIRE_BYTES:
            length, offset = _read_varint(view, offset)
            if length > _MAX_FIELD_BYTES:
                raise CheckpointFormatError("Protobuf byte field exceeds the safety limit.")
            end = offset + length
            if end > len(view):
                raise CheckpointFormatError("Truncated protobuf byte field.")
            value = view[offset:end]
            offset = end
        elif wire_type == _WIRE_FIXED32:
            end = offset + 4
            if end > len(view):
                raise CheckpointFormatError("Truncated protobuf fixed32 field.")
            value = bytes(view[offset:end])
            offset = end
        else:
            raise CheckpointFormatError(f"Unsupported protobuf wire type {wire_type}; groups are rejected.")
        yield field_number, wire_type, value


def _text(value: memoryview, *, field: str) -> str:
    try:
        return bytes(value).decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointFormatError(f"ONNX {field} is not valid UTF-8.") from error


def _packed_varints(value: memoryview, *, signed: bool = False) -> tuple[int, ...]:
    result = []
    offset = 0
    while offset < len(value):
        item, offset = _read_varint(value, offset)
        result.append(_signed_int64(item) if signed else item)
    return tuple(result)


def _packed_fixed32(value: memoryview) -> tuple[float, ...]:
    if len(value) % 4:
        raise CheckpointFormatError("Packed ONNX float data is misaligned.")
    return struct.unpack(f"<{len(value) // 4}f", value)


def _packed_fixed64(value: memoryview) -> tuple[float, ...]:
    if len(value) % 8:
        raise CheckpointFormatError("Packed ONNX double data is misaligned.")
    return struct.unpack(f"<{len(value) // 8}d", value)


@dataclass(frozen=True, slots=True)
class ONNXValueInfo:
    """Name, element type, and static/dynamic shape for one graph value."""

    name: str
    element_type: int | None
    shape: tuple[int | str | None, ...]


@dataclass(frozen=True, slots=True)
class ONNXTensor:
    """One inline ONNX tensor initializer."""

    name: str
    data_type: int
    dimensions: tuple[int, ...]
    raw_data: bytes
    float_data: tuple[float, ...] = ()
    int32_data: tuple[int, ...] = ()
    int64_data: tuple[int, ...] = ()
    double_data: tuple[float, ...] = ()
    uint64_data: tuple[int, ...] = ()

    @property
    def element_count(self) -> int:
        count = 1
        for dimension in self.dimensions:
            count *= dimension
        return count

    def to_torch(self):
        """Materialize a cloned CPU tensor without NumPy."""
        import torch

        dtypes = {
            2: (torch.uint8, 1),
            3: (torch.int8, 1),
            4: (torch.uint16, 2),
            5: (torch.int16, 2),
            1: (torch.float32, 4),
            6: (torch.int32, 4),
            7: (torch.int64, 8),
            9: (torch.bool, 1),
            10: (torch.float16, 2),
            11: (torch.float64, 8),
            12: (torch.uint32, 4),
            13: (torch.uint64, 8),
            16: (torch.bfloat16, 2),
        }
        try:
            dtype, width = dtypes[self.data_type]
        except KeyError:
            raise CheckpointFormatError(
                f"ONNX tensor {self.name or '<anonymous>'!r} uses "
                f"unsupported dtype {self.data_type}.") from None
        count = self.element_count
        if self.raw_data:
            expected = count * width
            if len(self.raw_data) != expected:
                raise CheckpointFormatError(
                    f"ONNX tensor {self.name or '<anonymous>'!r} contains "
                    f"{len(self.raw_data)} raw bytes; expected {expected}.")
            values = torch.frombuffer(
                bytearray(self.raw_data),
                dtype=dtype,
                count=count,
            ).clone()
        else:
            source = {
                1: self.float_data,
                2: self.int32_data,
                3: self.int32_data,
                4: self.int32_data,
                5: self.int32_data,
                6: self.int32_data,
                7: self.int64_data,
                9: self.int32_data,
                10: self.int32_data,
                11: self.double_data,
                12: self.uint64_data,
                13: self.uint64_data,
                16: self.int32_data,
            }[self.data_type]
            if len(source) != count:
                raise CheckpointFormatError(
                    f"ONNX tensor {self.name or '<anonymous>'!r} contains "
                    f"{len(source)} values; expected {count}.")
            if self.data_type in {10, 16}:
                encoded = struct.pack(
                    f"<{count}H",
                    *(int(item) & 0xFFFF for item in source),
                )
                values = torch.frombuffer(
                    bytearray(encoded),
                    dtype=dtype,
                    count=count,
                ).clone()
            else:
                values = torch.tensor(source, dtype=dtype)
        return values.reshape(self.dimensions)


@dataclass(frozen=True, slots=True)
class ONNXAttribute:
    """One immutable ONNX node attribute.

    ``attribute_type`` retains ONNX's declared ``AttributeType`` enum so
    an architecture fingerprint can distinguish semantically different
    encodings even when their Python values compare equal.
    """

    name: str
    attribute_type: int
    value: Any


@dataclass(frozen=True, slots=True)
class ONNXNode:
    """Operator identity and connections used for graph validation."""

    op_type: str
    domain: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    name: str = ""
    attributes: Mapping[str, ONNXAttribute] = field(default_factory=lambda: MappingProxyType({}), )


@dataclass(frozen=True, slots=True)
class ONNXGraph:
    """Parsed graph structure and immutable initializer mapping."""

    name: str
    inputs: tuple[ONNXValueInfo, ...]
    outputs: tuple[ONNXValueInfo, ...]
    nodes: tuple[ONNXNode, ...]
    initializers: Mapping[str, ONNXTensor]


@dataclass(frozen=True, slots=True)
class ONNXModel:
    """Safe structural representation of an ONNX ``ModelProto``."""

    ir_version: int
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    opsets: tuple[tuple[str, int], ...]
    metadata: Mapping[str, str]
    graph: ONNXGraph


def _tensor_semantic_record(
    tensor: ONNXTensor,
    *,
    include_values: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "data_type": tensor.data_type,
        "dimensions": list(tensor.dimensions),
    }
    if not include_values:
        return record
    if tensor.raw_data:
        encoded = tensor.raw_data
    else:
        values = {
            "float_data": [value.hex() for value in tensor.float_data],
            "int32_data": list(tensor.int32_data),
            "int64_data": list(tensor.int64_data),
            "double_data": [value.hex() for value in tensor.double_data],
            "uint64_data": list(tensor.uint64_data),
        }
        encoded = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    record["value_size"] = len(encoded)
    record["value_sha256"] = hashlib.sha256(encoded).hexdigest()
    return record


def _attribute_semantic_value(value: Any) -> Any:
    if isinstance(value, ONNXTensor):
        return {
            "tensor": _tensor_semantic_record(
                value,
                include_values=True,
            ),
        }
    if isinstance(value, bytes):
        return {
            "bytes_size": len(value),
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, float):
        return {"float": value.hex()}
    if isinstance(value, int):
        return {"integer": value}
    if isinstance(value, tuple):
        return [_attribute_semantic_value(item) for item in value]
    raise TypeError(f"Unsupported ONNX semantic attribute value "
                    f"{type(value).__name__}.")


def onnx_semantic_fingerprint(model: ONNXModel) -> str:
    """Hash graph semantics without hashing learned initializer values.

    Node order, connections, operator attributes, constant values, graph
    I/O, opsets, and initializer namespaces/dtypes/shapes are covered.
    Learned initializer payloads are deliberately excluded so an
    architecture can validate the same graph before and after fine-
    tuning.
    """
    if not isinstance(model, ONNXModel):
        raise TypeError("`model` must be an ONNXModel.")

    def value_info(value: ONNXValueInfo) -> dict[str, Any]:
        return {
            "name": value.name,
            "element_type": value.element_type,
            "shape": list(value.shape),
        }

    payload = {
        "ir_version": model.ir_version,
        "domain": model.domain,
        "opsets": [list(item) for item in model.opsets],
        "graph": {
            "name":
            model.graph.name,
            "inputs": [value_info(value) for value in model.graph.inputs],
            "outputs": [value_info(value) for value in model.graph.outputs],
            "initializers": [{
                "name": name,
                **_tensor_semantic_record(
                    tensor,
                    include_values=False,
                ),
            } for name, tensor in sorted(model.graph.initializers.items(), )],
            "nodes": [{
                "name":
                node.name,
                "domain":
                node.domain,
                "op_type":
                node.op_type,
                "inputs":
                list(node.inputs),
                "outputs":
                list(node.outputs),
                "attributes": [{
                    "name": name,
                    "attribute_type": attribute.attribute_type,
                    "value": _attribute_semantic_value(attribute.value, ),
                } for name, attribute in sorted(node.attributes.items(), )],
            } for node in model.graph.nodes],
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_dimension(data: memoryview) -> int | str | None:
    result: int | str | None = None
    for number, wire, value in _fields(data):
        if number == 1 and wire == _WIRE_VARINT:
            result = _signed_int64(value)
        elif number == 2 and wire == _WIRE_BYTES:
            result = _text(value, field="dimension parameter")
    return result


def _parse_value_info(data: memoryview) -> ONNXValueInfo:
    name = ""
    element_type: int | None = None
    shape: tuple[int | str | None, ...] = ()
    for number, wire, value in _fields(data):
        if number == 1 and wire == _WIRE_BYTES:
            name = _text(value, field="value name")
        elif number == 2 and wire == _WIRE_BYTES:
            for type_number, type_wire, type_value in _fields(value):
                if type_number != 1 or type_wire != _WIRE_BYTES:
                    continue
                dimensions = []
                for tensor_number, tensor_wire, tensor_value in _fields(type_value):
                    if tensor_number == 1 and tensor_wire == _WIRE_VARINT:
                        element_type = tensor_value
                    elif tensor_number == 2 and tensor_wire == _WIRE_BYTES:
                        for shape_number, shape_wire, shape_value in _fields(tensor_value):
                            if shape_number == 1 and shape_wire == _WIRE_BYTES:
                                dimensions.append(_parse_dimension(shape_value))
                shape = tuple(dimensions)
    if not name:
        raise CheckpointFormatError("ONNX graph value has an empty name.")
    return ONNXValueInfo(name=name, element_type=element_type, shape=shape)


def _parse_tensor(
    data: memoryview,
    *,
    require_name: bool = True,
) -> ONNXTensor:
    dimensions: list[int] = []
    data_type = 0
    name = ""
    raw_data = b""
    float_data: list[float] = []
    int32_data: list[int] = []
    int64_data: list[int] = []
    double_data: list[float] = []
    uint64_data: list[int] = []
    external_data = False
    data_location = 0
    for number, wire, value in _fields(data):
        if number == 1:
            if wire == _WIRE_VARINT:
                dimensions.append(_signed_int64(value))
            elif wire == _WIRE_BYTES:
                dimensions.extend(_packed_varints(value, signed=True))
        elif number == 2 and wire == _WIRE_VARINT:
            data_type = value
        elif number == 4:
            if wire == _WIRE_FIXED32:
                float_data.append(struct.unpack("<f", value)[0])
            elif wire == _WIRE_BYTES:
                float_data.extend(_packed_fixed32(value))
        elif number == 5:
            if wire == _WIRE_VARINT:
                int32_data.append(_signed_int64(value))
            elif wire == _WIRE_BYTES:
                int32_data.extend(_packed_varints(value, signed=True))
        elif number == 7:
            if wire == _WIRE_VARINT:
                int64_data.append(_signed_int64(value))
            elif wire == _WIRE_BYTES:
                int64_data.extend(_packed_varints(value, signed=True))
        elif number == 8 and wire == _WIRE_BYTES:
            name = _text(value, field="tensor name")
        elif number == 9 and wire == _WIRE_BYTES:
            raw_data = bytes(value)
        elif number == 10:
            if wire == _WIRE_FIXED64:
                double_data.append(struct.unpack("<d", value)[0])
            elif wire == _WIRE_BYTES:
                double_data.extend(_packed_fixed64(value))
        elif number == 11:
            if wire == _WIRE_VARINT:
                uint64_data.append(value)
            elif wire == _WIRE_BYTES:
                uint64_data.extend(_packed_varints(value))
        elif number == 13:
            external_data = True
        elif number == 14 and wire == _WIRE_VARINT:
            data_location = value
    if require_name and not name:
        raise CheckpointFormatError("ONNX initializer has an empty name.")
    if data_type == 0:
        raise CheckpointFormatError(f"ONNX initializer {name!r} has no dtype.")
    if any(dimension < 0 for dimension in dimensions):
        raise CheckpointFormatError(f"ONNX initializer {name!r} has a negative dimension.")
    if external_data or data_location != 0:
        raise CheckpointFormatError(f"ONNX initializer {name!r} uses external data, which is rejected.")
    return ONNXTensor(
        name=name,
        data_type=data_type,
        dimensions=tuple(dimensions),
        raw_data=raw_data,
        float_data=tuple(float_data),
        int32_data=tuple(int32_data),
        int64_data=tuple(int64_data),
        double_data=tuple(double_data),
        uint64_data=tuple(uint64_data),
    )


def _parse_attribute(data: memoryview) -> ONNXAttribute:
    name = ""
    attribute_type = 0
    scalar_float: float | None = None
    scalar_integer: int | None = None
    scalar_string: bytes | None = None
    scalar_tensor: ONNXTensor | None = None
    floats: list[float] = []
    integers: list[int] = []
    strings: list[bytes] = []
    tensors: list[ONNXTensor] = []
    unsupported_fields: set[int] = set()
    for number, wire, value in _fields(data):
        if number == 1 and wire == _WIRE_BYTES:
            name = _text(value, field="attribute name")
        elif number == 20 and wire == _WIRE_VARINT:
            attribute_type = value
        elif number == 2 and wire == _WIRE_FIXED32:
            scalar_float = struct.unpack("<f", value)[0]
        elif number == 3 and wire == _WIRE_VARINT:
            scalar_integer = _signed_int64(value)
        elif number == 4 and wire == _WIRE_BYTES:
            scalar_string = bytes(value)
        elif number == 5 and wire == _WIRE_BYTES:
            scalar_tensor = _parse_tensor(value, require_name=False)
        elif number == 7:
            if wire == _WIRE_FIXED32:
                floats.append(struct.unpack("<f", value)[0])
            elif wire == _WIRE_BYTES:
                floats.extend(_packed_fixed32(value))
        elif number == 8:
            if wire == _WIRE_VARINT:
                integers.append(_signed_int64(value))
            elif wire == _WIRE_BYTES:
                integers.extend(_packed_varints(value, signed=True))
        elif number == 9 and wire == _WIRE_BYTES:
            strings.append(bytes(value))
        elif number == 10 and wire == _WIRE_BYTES:
            tensors.append(_parse_tensor(value, require_name=False))
        elif number in {13, 21}:
            # Reference name and documentation do not affect execution.
            continue
        elif number in {6, 11, 14, 15, 22, 23}:
            unsupported_fields.add(number)
    if not name:
        raise CheckpointFormatError("ONNX node attribute has an empty name.")
    if unsupported_fields:
        fields = ", ".join(str(number) for number in sorted(unsupported_fields))
        raise CheckpointFormatError(
            f"ONNX attribute {name!r} contains unsupported nested field(s): "
            f"{fields}.")
    values = {
        1: scalar_float,
        2: scalar_integer,
        3: scalar_string,
        4: scalar_tensor,
        6: tuple(floats),
        7: tuple(integers),
        8: tuple(strings),
        9: tuple(tensors),
    }
    if attribute_type not in values:
        raise CheckpointFormatError(f"ONNX attribute {name!r} uses unsupported type "
                                    f"{attribute_type}.")
    result = values[attribute_type]
    if result is None:
        raise CheckpointFormatError(f"ONNX attribute {name!r} does not contain its declared value.")
    return ONNXAttribute(
        name=name,
        attribute_type=attribute_type,
        value=result,
    )


def _parse_node(data: memoryview) -> ONNXNode:
    inputs = []
    outputs = []
    name = ""
    op_type = ""
    domain = ""
    attributes: dict[str, ONNXAttribute] = {}
    for number, wire, value in _fields(data):
        if wire != _WIRE_BYTES:
            continue
        if number == 1:
            inputs.append(_text(value, field="node input"))
        elif number == 2:
            outputs.append(_text(value, field="node output"))
        elif number == 3:
            name = _text(value, field="node name")
        elif number == 4:
            op_type = _text(value, field="operator type")
        elif number == 7:
            domain = _text(value, field="operator domain")
        elif number == 5:
            attribute = _parse_attribute(value)
            if attribute.name in attributes:
                raise CheckpointFormatError(
                    f"ONNX node contains duplicate attribute "
                    f"{attribute.name!r}.")
            attributes[attribute.name] = attribute
    if not op_type:
        raise CheckpointFormatError("ONNX node has an empty operator type.")
    return ONNXNode(
        op_type=op_type,
        domain=domain,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        name=name,
        attributes=MappingProxyType(attributes),
    )


def _parse_graph(data: memoryview) -> ONNXGraph:
    nodes = []
    tensors: dict[str, ONNXTensor] = {}
    inputs = []
    outputs = []
    name = ""
    for number, wire, value in _fields(data):
        if wire != _WIRE_BYTES:
            continue
        if number == 1:
            nodes.append(_parse_node(value))
        elif number == 2:
            name = _text(value, field="graph name")
        elif number == 5:
            tensor = _parse_tensor(value)
            if tensor.name in tensors:
                raise CheckpointFormatError(f"Duplicate ONNX initializer {tensor.name!r}.")
            tensors[tensor.name] = tensor
        elif number == 11:
            inputs.append(_parse_value_info(value))
        elif number == 12:
            outputs.append(_parse_value_info(value))
    return ONNXGraph(
        name=name,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        nodes=tuple(nodes),
        initializers=MappingProxyType(tensors),
    )


def _parse_opset(data: memoryview) -> tuple[str, int]:
    domain = ""
    version = 0
    for number, wire, value in _fields(data):
        if number == 1 and wire == _WIRE_BYTES:
            domain = _text(value, field="opset domain")
        elif number == 2 and wire == _WIRE_VARINT:
            version = _signed_int64(value)
    return domain, version


def _parse_metadata(data: memoryview) -> tuple[str, str]:
    key = ""
    value = ""
    for number, wire, item in _fields(data):
        if number == 1 and wire == _WIRE_BYTES:
            key = _text(item, field="metadata key")
        elif number == 2 and wire == _WIRE_BYTES:
            value = _text(item, field="metadata value")
    if not key:
        raise CheckpointFormatError("ONNX metadata contains an empty key.")
    return key, value


def read_onnx_model(
    path: str | Path,
    *,
    max_file_bytes: int = _MAX_FILE_BYTES,
) -> ONNXModel:
    """Parse an inline-data ONNX model without importing ONNX or protobuf."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"ONNX checkpoint was not found: {source}.")
    if (isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int) or max_file_bytes < 1):
        raise ValueError("`max_file_bytes` must be a positive integer.")
    size = source.stat().st_size
    if size > max_file_bytes:
        raise CheckpointFormatError(
            f"ONNX checkpoint is {size} bytes, exceeding the "
            f"{max_file_bytes}-byte safety limit.")
    payload = source.read_bytes()
    ir_version = 0
    producer_name = ""
    producer_version = ""
    domain = ""
    model_version = 0
    opsets = []
    metadata: dict[str, str] = {}
    graph: ONNXGraph | None = None
    for number, wire, value in _fields(payload):
        if number == 1 and wire == _WIRE_VARINT:
            ir_version = value
        elif number == 2 and wire == _WIRE_BYTES:
            producer_name = _text(value, field="producer name")
        elif number == 3 and wire == _WIRE_BYTES:
            producer_version = _text(value, field="producer version")
        elif number == 4 and wire == _WIRE_BYTES:
            domain = _text(value, field="model domain")
        elif number == 5 and wire == _WIRE_VARINT:
            model_version = _signed_int64(value)
        elif number == 7 and wire == _WIRE_BYTES:
            if graph is not None:
                raise CheckpointFormatError("ONNX model contains multiple graphs.")
            graph = _parse_graph(value)
        elif number == 8 and wire == _WIRE_BYTES:
            opsets.append(_parse_opset(value))
        elif number == 14 and wire == _WIRE_BYTES:
            key, item = _parse_metadata(value)
            if key in metadata:
                raise CheckpointFormatError(f"Duplicate ONNX metadata key {key!r}.")
            metadata[key] = item
    if graph is None:
        raise CheckpointFormatError("ONNX model does not contain a graph.")
    return ONNXModel(
        ir_version=ir_version,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        opsets=tuple(opsets),
        metadata=MappingProxyType(metadata),
        graph=graph,
    )


__all__ = [
    "ONNXAttribute",
    "ONNXGraph",
    "ONNXModel",
    "ONNXNode",
    "ONNXTensor",
    "ONNXValueInfo",
    "onnx_semantic_fingerprint",
    "read_onnx_model",
]
