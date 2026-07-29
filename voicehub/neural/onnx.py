"""Differentiable execution of allowlisted ONNX graphs with PyTorch.

This is not a general ONNX runtime. It is a small, auditable execution
layer for reviewed speech-model artifacts whose exact graph and digest
are validated by a VoiceHub architecture adapter. Operators are
implemented with PyTorch, so floating-point initializers remain ordinary
parameters and gradients can flow through the imported graph.

The runtime intentionally rejects custom domains, control-flow graphs,
external tensor data, and operators outside its explicit allowlist.
Those constraints keep imported model behavior reviewable and prevent a
checkpoint format from becoming an arbitrary-code extension mechanism.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional

from voicehub.checkpointing.onnx import ONNXAttribute, ONNXModel, ONNXNode, ONNXTensor, read_onnx_model

_STANDARD_DOMAINS = frozenset({"", "ai.onnx"})
_FLOAT_DTYPES = frozenset({
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
})
_SUPPORTED_OPERATORS = frozenset({
    "Add",
    "BatchNormalization",
    "Cast",
    "Clip",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Conv",
    "Cos",
    "Div",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "Gather",
    "Gemm",
    "Identity",
    "LayerNormalization",
    "MatMul",
    "Mul",
    "Neg",
    "PRelu",
    "Pad",
    "Pow",
    "Reciprocal",
    "ReduceMean",
    "ReduceSum",
    "Relu",
    "Reshape",
    "Shape",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "Softplus",
    "Split",
    "Squeeze",
    "Sub",
    "Tanh",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "Where",
})
_ONNX_DTYPES = {
    1: torch.float32,
    2: torch.uint8,
    3: torch.int8,
    4: torch.uint16,
    5: torch.int16,
    6: torch.int32,
    7: torch.int64,
    9: torch.bool,
    10: torch.float16,
    11: torch.float64,
    12: torch.uint32,
    13: torch.uint64,
    16: torch.bfloat16,
}


class NativeONNXError(RuntimeError):
    """Base error for safe native ONNX graph construction and execution."""


class UnsupportedONNXGraphError(NativeONNXError, ValueError):
    """Raised when a reviewed graph uses unsupported semantics."""


class ONNXExecutionError(NativeONNXError):
    """Raised when an allowlisted node cannot execute its declared contract."""


def _attribute(
    node: ONNXNode,
    name: str,
    default: Any = None,
) -> Any:
    record = node.attributes.get(name)
    return default if record is None else record.value


def _integer_attribute(
    node: ONNXNode,
    name: str,
    default: int,
) -> int:
    value = _attribute(node, name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ONNXExecutionError(
            f"Node {node.name or node.op_type!r} attribute {name!r} "
            "must be an integer.")
    return value


def _integers_attribute(
        node: ONNXNode,
        name: str,
        default: tuple[int, ...] = (),
) -> tuple[int, ...]:
    value = _attribute(node, name, default)
    if not isinstance(value, tuple) or any(isinstance(item, bool) or not isinstance(item, int)
                                           for item in value):
        raise ONNXExecutionError(
            f"Node {node.name or node.op_type!r} attribute {name!r} "
            "must be an integer sequence.")
    return value


def _float_attribute(
    node: ONNXNode,
    name: str,
    default: float,
) -> float:
    value = _attribute(node, name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ONNXExecutionError(
            f"Node {node.name or node.op_type!r} attribute {name!r} "
            "must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ONNXExecutionError(f"Node {node.name or node.op_type!r} attribute {name!r} "
                                 "must be finite.")
    return result


def _string_attribute(
    node: ONNXNode,
    name: str,
    default: str,
) -> str:
    value = _attribute(node, name, default.encode("utf-8"))
    if not isinstance(value, bytes):
        raise ONNXExecutionError(
            f"Node {node.name or node.op_type!r} attribute {name!r} "
            "must be a byte string.")
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ONNXExecutionError(
            f"Node {node.name or node.op_type!r} attribute {name!r} "
            "is not UTF-8.") from error


def _tensor_values(value: Tensor, *, name: str) -> tuple[int, ...]:
    if value.ndim > 1:
        raise ONNXExecutionError(f"`{name}` must be a scalar or vector.")
    if value.dtype == torch.bool or value.is_floating_point():
        raise ONNXExecutionError(f"`{name}` must contain integers.")
    return tuple(int(item) for item in value.detach().cpu().reshape(-1))


def _optional_tensor(
    values: tuple[Tensor | None, ...],
    index: int,
) -> Tensor | None:
    return values[index] if index < len(values) else None


def _normalize_axis(axis: int, rank: int, *, name: str) -> int:
    normalized = axis + rank if axis < 0 else axis
    if not 0 <= normalized < rank:
        raise ONNXExecutionError(f"{name} axis {axis} is invalid for rank {rank}.")
    return normalized


def _constant_tensor(attribute: ONNXAttribute, *, node: ONNXNode) -> Tensor:
    if not isinstance(attribute.value, ONNXTensor):
        raise UnsupportedONNXGraphError(f"Constant node {node.name!r} must contain a tensor value.")
    return attribute.value.to_torch()


def _node_label(node: ONNXNode) -> str:
    return node.name or node.op_type


class NativeONNXGraph(nn.Module):
    """Execute one reviewed standard-domain ONNX graph with PyTorch.

    Args:
        model: Parsed ONNX model.
        trainable: ``True`` marks every floating initializer trainable except
            inference BatchNorm statistics. ``False`` freezes every
            initializer. An iterable selects exact initializer names.

    Architecture adapters should validate the source digest and graph
    fingerprint before constructing this module.
    """

    def __init__(
        self,
        model: ONNXModel,
        *,
        trainable: bool | Iterable[str] = True,
    ) -> None:
        super().__init__()
        if not isinstance(model, ONNXModel):
            raise TypeError("`model` must be a parsed ONNXModel.")
        self._validate_model(model)
        graph = model.graph
        self.ir_version = model.ir_version
        self.opsets = tuple(model.opsets)
        self.graph_name = graph.name
        initializer_names = tuple(sorted(graph.initializers))
        selected = self._resolve_trainable_names(
            model,
            initializer_names,
            trainable,
        )
        slots: dict[str, str] = {}
        trainable_slots: set[str] = set()
        for index, name in enumerate(initializer_names):
            tensor = graph.initializers[name].to_torch()
            slot = f"initializer_{index:06d}"
            slots[name] = slot
            if tensor.dtype in _FLOAT_DTYPES:
                parameter = nn.Parameter(
                    tensor,
                    requires_grad=name in selected,
                )
                self.register_parameter(slot, parameter)
                if parameter.requires_grad:
                    trainable_slots.add(slot)
            else:
                self.register_buffer(slot, tensor, persistent=True)
        constant_slots: dict[str, str] = {}
        for index, node in enumerate(graph.nodes):
            if node.op_type != "Constant":
                continue
            try:
                attribute = node.attributes["value"]
            except KeyError:
                raise UnsupportedONNXGraphError(
                    f"Constant node {_node_label(node)!r} has no tensor value.") from None
            tensor = _constant_tensor(attribute, node=node)
            if len(node.outputs) != 1 or not node.outputs[0]:
                raise UnsupportedONNXGraphError(f"Constant node {_node_label(node)!r} must have one output.")
            slot = f"constant_{index:06d}"
            self.register_buffer(slot, tensor, persistent=False)
            constant_slots[node.outputs[0]] = slot
        self._initializer_slots = MappingProxyType(slots)
        self._constant_slots = MappingProxyType(constant_slots)
        self._trainable_slots = frozenset(trainable_slots)
        self._nodes = tuple(graph.nodes)
        self._input_names = tuple(
            value.name for value in graph.inputs if value.name not in graph.initializers)
        self._output_names = tuple(value.name for value in graph.outputs)
        self._persistent_names = frozenset((*self._input_names, *initializer_names, *self._output_names))
        self._use_counts = self._build_use_counts(self._nodes)
        self._initializer_names = initializer_names

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        trainable: bool | Iterable[str] = True,
        max_file_bytes: int = 2 * 1024 * 1024 * 1024,
    ) -> NativeONNXGraph:
        """Read and materialize an inline-data ONNX graph."""
        model = read_onnx_model(path, max_file_bytes=max_file_bytes)
        return cls(model, trainable=trainable)

    @staticmethod
    def _validate_model(model: ONNXModel) -> None:
        if not model.graph.outputs:
            raise UnsupportedONNXGraphError("ONNX graph has no outputs.")
        opsets = dict(model.opsets)
        unsupported_domains = {domain for domain in opsets if domain not in _STANDARD_DOMAINS}
        unsupported_domains.update(
            node.domain for node in model.graph.nodes if node.domain not in _STANDARD_DOMAINS)
        if unsupported_domains:
            names = ", ".join(sorted(unsupported_domains))
            raise UnsupportedONNXGraphError(f"Custom ONNX domains are unsupported: {names}.")
        default_opset = opsets.get("", opsets.get("ai.onnx", 0))
        if not 13 <= default_opset <= 19:
            raise UnsupportedONNXGraphError(
                "Native ONNX execution supports standard opsets 13 through "
                f"19; found {default_opset}.")
        unsupported_ops = sorted(
            {node.op_type
             for node in model.graph.nodes if node.op_type not in _SUPPORTED_OPERATORS})
        if unsupported_ops:
            raise UnsupportedONNXGraphError(
                "Unsupported ONNX operator(s): "
                f"{', '.join(unsupported_ops)}.")
        available = {value.name for value in model.graph.inputs} | set(model.graph.initializers)
        produced = set(available)
        for node in model.graph.nodes:
            missing = [name for name in node.inputs if name and name not in produced]
            if missing:
                raise UnsupportedONNXGraphError(
                    f"Node {_node_label(node)!r} reads unavailable value(s): "
                    f"{', '.join(missing)}.")
            for output in node.outputs:
                if not output:
                    continue
                if output in produced:
                    raise UnsupportedONNXGraphError(f"ONNX value {output!r} is produced more than once.")
                produced.add(output)
        missing_outputs = [value.name for value in model.graph.outputs if value.name not in produced]
        if missing_outputs:
            raise UnsupportedONNXGraphError(
                "ONNX graph output(s) are unavailable: "
                f"{', '.join(missing_outputs)}.")

    @staticmethod
    def _resolve_trainable_names(
        model: ONNXModel,
        initializer_names: tuple[str, ...],
        trainable: bool | Iterable[str],
    ) -> frozenset[str]:
        floating = {
            name
            for name, value in model.graph.initializers.items() if value.data_type in {1, 10, 11, 16}
        }
        batch_norm_statistics = {
            name
            for node in model.graph.nodes if node.op_type == "BatchNormalization" for name in node.inputs[3:5]
            if name
        }
        if isinstance(trainable, bool):
            return (frozenset(floating - batch_norm_statistics) if trainable else frozenset())
        names = tuple(trainable)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("`trainable` must contain non-empty initializer names.")
        if len(names) != len(set(names)):
            raise ValueError("`trainable` cannot contain duplicate names.")
        unknown = sorted(set(names) - set(initializer_names))
        if unknown:
            raise ValueError(f"Unknown trainable initializer(s): {', '.join(unknown)}.")
        non_floating = sorted(set(names) - floating)
        if non_floating:
            raise ValueError(
                "Integer or boolean initializers cannot be trainable: "
                f"{', '.join(non_floating)}.")
        return frozenset(names)

    @staticmethod
    def _build_use_counts(nodes: tuple[ONNXNode, ...], ) -> Mapping[str, int]:
        counts: dict[str, int] = {}
        for node in nodes:
            for name in node.inputs:
                if name:
                    counts[name] = counts.get(name, 0) + 1
        return MappingProxyType(counts)

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def initializer_names(self) -> tuple[str, ...]:
        return self._initializer_names

    @property
    def trainable_initializer_names(self) -> tuple[str, ...]:
        return tuple(name for name, slot in self._initializer_slots.items() if slot in self._trainable_slots)

    def initializer_tensor(self, name: str) -> Tensor:
        """Return one initializer by its original ONNX name."""
        try:
            slot = self._initializer_slots[name]
        except KeyError:
            raise KeyError(f"Unknown ONNX initializer {name!r}.") from None
        return getattr(self, slot)

    def named_onnx_initializers(self, ) -> tuple[tuple[str, Tensor], ...]:
        """Return stable original names paired with live tensors."""
        return tuple((name, getattr(self, slot)) for name, slot in self._initializer_slots.items())

    def onnx_state_dict(self) -> dict[str, Tensor]:
        """Return a detached original-name state dictionary."""
        return {name: tensor.detach().clone() for name, tensor in self.named_onnx_initializers()}

    def load_onnx_state_dict(
        self,
        state: Mapping[str, Tensor],
        *,
        strict: bool = True,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Copy an original-name state dictionary into the live graph."""
        if not isinstance(state, Mapping):
            raise TypeError("`state` must be a tensor mapping.")
        expected = set(self._initializer_names)
        provided = set(state)
        missing = tuple(sorted(expected - provided))
        unexpected = tuple(sorted(provided - expected))
        if strict and (missing or unexpected):
            raise ValueError(
                f"ONNX state mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected.")
        with torch.no_grad():
            for name in sorted(expected & provided):
                source = state[name]
                target = self.initializer_tensor(name)
                if not isinstance(source, Tensor):
                    raise TypeError(f"State value {name!r} is not a tensor.")
                if source.shape != target.shape:
                    raise ValueError(
                        f"State tensor {name!r} has shape "
                        f"{tuple(source.shape)}; expected {tuple(target.shape)}.")
                target.copy_(source.to(device=target.device, dtype=target.dtype))
        return missing, unexpected

    def _environment(
        self,
        inputs: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        if not isinstance(inputs, Mapping):
            raise TypeError("ONNX inputs must be a tensor mapping.")
        expected = set(self._input_names)
        provided = set(inputs)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            raise ONNXExecutionError(f"Invalid ONNX inputs ({'; '.join(details)}).")
        environment: dict[str, Tensor] = {}
        for name in self._input_names:
            value = inputs[name]
            if not isinstance(value, Tensor):
                raise TypeError(f"ONNX input {name!r} must be a tensor.")
            environment[name] = value
        environment.update({name: getattr(self, slot) for name, slot in self._initializer_slots.items()})
        environment.update({name: getattr(self, slot) for name, slot in self._constant_slots.items()})
        return environment

    def run(self, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Execute the graph and return an output-name mapping."""
        environment = self._environment(inputs)
        remaining_uses = dict(self._use_counts)
        for node in self._nodes:
            if node.op_type == "Constant":
                continue
            values = tuple(None if not name else environment[name] for name in node.inputs)
            try:
                outputs = self._execute(node, values)
            except NativeONNXError:
                raise
            except Exception as error:
                raise ONNXExecutionError(
                    f"Failed ONNX node {_node_label(node)!r} "
                    f"({node.op_type}): {error}.") from error
            if len(outputs) != len(node.outputs):
                raise ONNXExecutionError(
                    f"Node {_node_label(node)!r} produced {len(outputs)} "
                    f"value(s); graph declares {len(node.outputs)}.")
            for name, value in zip(node.outputs, outputs):
                if not name:
                    continue
                if not isinstance(value, Tensor):
                    raise ONNXExecutionError(f"Node {_node_label(node)!r} returned a non-tensor.")
                environment[name] = value
            for name in node.inputs:
                if not name or name not in remaining_uses:
                    continue
                remaining_uses[name] -= 1
                if (remaining_uses[name] == 0 and name not in self._persistent_names and
                        name not in self._constant_slots):
                    environment.pop(name, None)
        return {name: environment[name] for name in self._output_names}

    def forward(
        self,
        inputs: Mapping[str, Tensor] | None = None,
        **kwargs: Tensor,
    ) -> Tensor | tuple[Tensor, ...]:
        values = {} if inputs is None else dict(inputs)
        overlap = set(values) & set(kwargs)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Duplicate ONNX input(s): {names}.")
        values.update(kwargs)
        outputs = self.run(values)
        ordered = tuple(outputs[name] for name in self._output_names)
        return ordered[0] if len(ordered) == 1 else ordered

    def _execute(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> tuple[Tensor, ...]:
        operation = getattr(self, f"_op_{node.op_type}", None)
        if operation is None:
            raise UnsupportedONNXGraphError(f"Operator {node.op_type!r} is not implemented.")
        result = operation(node, values)
        return result if isinstance(result, tuple) else (result, )

    def _op_Add(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values
        return left + right

    def _op_Sub(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values
        return left - right

    def _op_Mul(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values
        return left * right

    def _op_Div(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values
        if not left.is_floating_point() and not right.is_floating_point():
            return torch.div(left, right, rounding_mode="trunc")
        return left / right

    def _op_Pow(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        base, exponent = values
        return torch.pow(base, exponent)

    def _op_Neg(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return -values[0]

    def _op_Reciprocal(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.reciprocal(values[0])

    def _op_Exp(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.exp(values[0])

    def _op_Erf(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.erf(values[0])

    def _op_Sin(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.sin(values[0])

    def _op_Cos(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.cos(values[0])

    def _op_Relu(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return functional.relu(values[0])

    def _op_Sigmoid(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.sigmoid(values[0])

    def _op_Tanh(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return torch.tanh(values[0])

    def _op_Softplus(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return functional.softplus(values[0])

    def _op_Identity(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return values[0]

    def _op_Equal(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values
        return torch.eq(left, right)

    def _op_Where(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        condition, left, right = values
        return torch.where(condition, left, right)

    def _op_MatMul(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values
        return torch.matmul(left, right)

    def _op_Gemm(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        left, right = values[:2]
        bias = _optional_tensor(values, 2)
        if _integer_attribute(node, "transA", 0):
            left = left.transpose(-1, -2)
        if _integer_attribute(node, "transB", 0):
            right = right.transpose(-1, -2)
        result = _float_attribute(node, "alpha", 1.0) * torch.matmul(
            left,
            right,
        )
        if bias is not None:
            result = result + _float_attribute(node, "beta", 1.0) * bias
        return result

    def _op_Conv(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, weight = values[:2]
        bias = _optional_tensor(values, 2)
        spatial_rank = data.ndim - 2
        if spatial_rank not in {1, 2, 3}:
            raise ONNXExecutionError(
                f"Conv supports one to three spatial dimensions, found "
                f"{spatial_rank}.")
        strides = _integers_attribute(
            node,
            "strides",
            (1, ) * spatial_rank,
        )
        dilations = _integers_attribute(
            node,
            "dilations",
            (1, ) * spatial_rank,
        )
        pads = _integers_attribute(
            node,
            "pads",
            (0, ) * (2 * spatial_rank),
        )
        if len(pads) != 2 * spatial_rank:
            raise ONNXExecutionError("Conv `pads` has the wrong rank.")
        if any(pads):
            pytorch_padding = []
            for begin, end in reversed(tuple(zip(pads[:spatial_rank], pads[spatial_rank:]))):
                pytorch_padding.extend((begin, end))
            data = functional.pad(data, tuple(pytorch_padding))
        group = _integer_attribute(node, "group", 1)
        convolution = {
            1: functional.conv1d,
            2: functional.conv2d,
            3: functional.conv3d,
        }[spatial_rank]
        return convolution(
            data,
            weight,
            bias,
            stride=strides,
            padding=0,
            dilation=dilations,
            groups=group,
        )

    def _op_BatchNormalization(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        if _integer_attribute(node, "training_mode", 0):
            raise UnsupportedONNXGraphError(
                "ONNX BatchNormalization training mode is unsupported; "
                "VoiceHub trains affine parameters against fixed published "
                "running statistics.")
        data, scale, bias, running_mean, running_variance = values
        return functional.batch_norm(
            data,
            running_mean,
            running_variance,
            scale,
            bias,
            training=False,
            momentum=_float_attribute(node, "momentum", 0.9),
            eps=_float_attribute(node, "epsilon", 1.0e-5),
        )

    def _op_LayerNormalization(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor | tuple[Tensor, ...]:
        data, scale = values[:2]
        bias = _optional_tensor(values, 2)
        axis = _normalize_axis(
            _integer_attribute(node, "axis", -1),
            data.ndim,
            name="LayerNormalization",
        )
        normalized_shape = tuple(data.shape[axis:])
        normalized = functional.layer_norm(
            data,
            normalized_shape,
            scale,
            bias,
            _float_attribute(node, "epsilon", 1.0e-5),
        )
        if len(node.outputs) == 1:
            return normalized
        dimensions = tuple(range(axis, data.ndim))
        mean = data.mean(dim=dimensions, keepdim=True)
        variance = (data - mean).square().mean(
            dim=dimensions,
            keepdim=True,
        )
        inverse_std = (variance + _float_attribute(node, "epsilon", 1.0e-5)).rsqrt()
        outputs = (normalized, mean, inverse_std)
        return outputs[:len(node.outputs)]

    def _op_PRelu(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, slope = values
        return functional.prelu(data, slope.reshape(-1))

    def _op_Softmax(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data = values[0]
        axis = _normalize_axis(
            _integer_attribute(node, "axis", -1),
            data.ndim,
            name="Softmax",
        )
        return functional.softmax(data, dim=axis)

    def _op_Clip(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data = values[0]
        minimum = _optional_tensor(values, 1)
        maximum = _optional_tensor(values, 2)
        minimum_value = (None if minimum is None else minimum.to(
            device=data.device,
            dtype=data.dtype,
        ))
        maximum_value = (None if maximum is None else maximum.to(
            device=data.device,
            dtype=data.dtype,
        ))
        if minimum_value is not None:
            data = torch.maximum(data, minimum_value)
        if maximum_value is not None:
            data = torch.minimum(data, maximum_value)
        return data

    def _op_Cast(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data_type = _integer_attribute(node, "to", 0)
        try:
            dtype = _ONNX_DTYPES[data_type]
        except KeyError:
            raise UnsupportedONNXGraphError(f"Cast target dtype {data_type} is unsupported.") from None
        return values[0].to(dtype=dtype)

    def _op_Shape(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data = values[0]
        start = _integer_attribute(node, "start", 0)
        end = _integer_attribute(node, "end", data.ndim)
        dimensions = tuple(data.shape)[slice(start, end)]
        return torch.tensor(
            dimensions,
            dtype=torch.int64,
            device=data.device,
        )

    def _op_Reshape(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, shape_tensor = values
        shape = list(_tensor_values(shape_tensor, name="Reshape shape"))
        if not _integer_attribute(node, "allowzero", 0):
            for index, dimension in enumerate(shape):
                if dimension == 0:
                    if index >= data.ndim:
                        raise ONNXExecutionError(
                            "Reshape zero dimension cannot copy an absent "
                            "input dimension.")
                    shape[index] = data.shape[index]
        return data.reshape(tuple(shape))

    def _op_Transpose(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data = values[0]
        permutation = _integers_attribute(
            node,
            "perm",
            tuple(reversed(range(data.ndim))),
        )
        if len(permutation) != data.ndim:
            raise ONNXExecutionError("Transpose permutation does not match the input rank.")
        return data.permute(permutation)

    def _op_Concat(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        tensors = tuple(value for value in values if value is not None)
        if not tensors:
            raise ONNXExecutionError("Concat requires at least one tensor.")
        axis = _normalize_axis(
            _integer_attribute(node, "axis", 0),
            tensors[0].ndim,
            name="Concat",
        )
        return torch.cat(tensors, dim=axis)

    def _op_Unsqueeze(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, axes_tensor = values
        axes = _tensor_values(axes_tensor, name="Unsqueeze axes")
        output_rank = data.ndim + len(axes)
        normalized = sorted(_normalize_axis(axis, output_rank, name="Unsqueeze") for axis in axes)
        if len(normalized) != len(set(normalized)):
            raise ONNXExecutionError("Unsqueeze axes cannot repeat.")
        result = data
        for axis in normalized:
            result = result.unsqueeze(axis)
        return result

    def _op_Squeeze(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data = values[0]
        axes_tensor = _optional_tensor(values, 1)
        if axes_tensor is None:
            return data.squeeze()
        axes = _tensor_values(axes_tensor, name="Squeeze axes")
        normalized = sorted(
            (_normalize_axis(axis, data.ndim, name="Squeeze") for axis in axes),
            reverse=True,
        )
        result = data
        for axis in normalized:
            if result.shape[axis] != 1:
                raise ONNXExecutionError(
                    f"Cannot squeeze dimension {axis} with size "
                    f"{result.shape[axis]}.")
            result = result.squeeze(axis)
        return result

    def _op_Expand(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, shape_tensor = values
        requested = _tensor_values(shape_tensor, name="Expand shape")
        try:
            shape = torch.broadcast_shapes(tuple(data.shape), requested)
        except RuntimeError as error:
            raise ONNXExecutionError(
                f"Expand cannot broadcast shape {tuple(data.shape)} with "
                f"{requested}.") from error
        return torch.broadcast_to(data, shape)

    def _op_Tile(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, repeats = values
        return torch.tile(
            data,
            _tensor_values(repeats, name="Tile repeats"),
        )

    def _op_Gather(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, indices = values
        axis = _normalize_axis(
            _integer_attribute(node, "axis", 0),
            data.ndim,
            name="Gather",
        )
        flat = indices.to(device=data.device, dtype=torch.long).reshape(-1)
        flat = torch.where(flat < 0, flat + data.shape[axis], flat)
        selected = torch.index_select(data, axis, flat)
        shape = (tuple(data.shape[:axis]) + tuple(indices.shape) + tuple(data.shape[axis + 1:]))
        return selected.reshape(shape)

    def _op_Slice(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, starts_tensor, ends_tensor = values[:3]
        axes_tensor = _optional_tensor(values, 3)
        steps_tensor = _optional_tensor(values, 4)
        starts = _tensor_values(starts_tensor, name="Slice starts")
        ends = _tensor_values(ends_tensor, name="Slice ends")
        axes = (
            tuple(range(len(starts))) if axes_tensor is None else _tensor_values(
                axes_tensor, name="Slice axes"))
        steps = ((1, ) *
                 len(starts) if steps_tensor is None else _tensor_values(steps_tensor, name="Slice steps"))
        if not (len(starts) == len(ends) == len(axes) == len(steps)):
            raise ONNXExecutionError("Slice controls must have identical lengths.")
        result = data
        normalized_axes = [_normalize_axis(axis, data.ndim, name="Slice") for axis in axes]
        if len(normalized_axes) != len(set(normalized_axes)):
            raise ONNXExecutionError("Slice axes cannot repeat.")
        for start, end, axis, step in zip(
                starts,
                ends,
                normalized_axes,
                steps,
        ):
            if step == 0:
                raise ONNXExecutionError("Slice step cannot be zero.")
            indices = tuple(range(result.shape[axis]))[slice(start, end, step)]
            index = torch.tensor(
                indices,
                dtype=torch.long,
                device=result.device,
            )
            result = torch.index_select(result, axis, index)
        return result

    def _op_Split(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> tuple[Tensor, ...]:
        data = values[0]
        axis = _normalize_axis(
            _integer_attribute(node, "axis", 0),
            data.ndim,
            name="Split",
        )
        split_tensor = _optional_tensor(values, 1)
        if split_tensor is None:
            if data.shape[axis] % len(node.outputs):
                raise ONNXExecutionError("Equal Split cannot divide the selected dimension.")
            size = data.shape[axis] // len(node.outputs)
            sections: int | tuple[int, ...] = size
        else:
            sections = _tensor_values(
                split_tensor,
                name="Split lengths",
            )
        return tuple(torch.split(data, sections, dim=axis))

    def _op_Pad(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        data, pads_tensor = values[:2]
        constant_value = _optional_tensor(values, 2)
        axes_tensor = _optional_tensor(values, 3)
        pads = _tensor_values(pads_tensor, name="Pad pads")
        axes = (
            tuple(range(data.ndim)) if axes_tensor is None else tuple(
                _normalize_axis(axis, data.ndim, name="Pad")
                for axis in _tensor_values(axes_tensor, name="Pad axes")))
        if len(pads) != 2 * len(axes):
            raise ONNXExecutionError("Pad controls have incompatible lengths.")
        by_axis = {axis: [0, 0] for axis in range(data.ndim)}
        for index, axis in enumerate(axes):
            by_axis[axis] = [pads[index], pads[index + len(axes)]]
        pytorch_padding: list[int] = []
        for axis in reversed(range(data.ndim)):
            pytorch_padding.extend(by_axis[axis])
        mode = _string_attribute(node, "mode", "constant")
        if mode == "edge":
            mode = "replicate"
        if mode not in {"constant", "reflect", "replicate"}:
            raise UnsupportedONNXGraphError(f"Pad mode {mode!r} is unsupported.")
        if mode != "constant":
            padded_axes = {axis for axis, pair in by_axis.items() if pair != [0, 0]}
            trailing = set(range(data.ndim - len(padded_axes), data.ndim))
            if padded_axes and padded_axes != trailing:
                raise UnsupportedONNXGraphError("Reflect/edge Pad supports contiguous trailing axes only.")
            compact = pytorch_padding[:2 * len(padded_axes)]
            return functional.pad(data, compact, mode=mode)
        fill = (0.0 if constant_value is None else float(constant_value.detach().cpu().reshape(()).item()))
        return functional.pad(
            data,
            tuple(pytorch_padding),
            mode="constant",
            value=fill,
        )

    def _op_ConstantOfShape(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        shape = _tensor_values(values[0], name="ConstantOfShape shape")
        attribute = node.attributes.get("value")
        if attribute is None:
            scalar = torch.tensor(0.0, dtype=torch.float32)
        else:
            scalar = _constant_tensor(attribute, node=node)
        if scalar.numel() != 1:
            raise ONNXExecutionError("ConstantOfShape value must contain one element.")
        return torch.full(
            shape,
            scalar.item(),
            dtype=scalar.dtype,
            device=values[0].device,
        )

    def _op_ReduceSum(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return self._reduce(node, values, reduction="sum")

    def _op_ReduceMean(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
    ) -> Tensor:
        return self._reduce(node, values, reduction="mean")

    def _reduce(
        self,
        node: ONNXNode,
        values: tuple[Tensor | None, ...],
        *,
        reduction: str,
    ) -> Tensor:
        data = values[0]
        axes_tensor = _optional_tensor(values, 1)
        if axes_tensor is None:
            axes = tuple(range(data.ndim))
        else:
            axes = tuple(
                _normalize_axis(axis, data.ndim, name=node.op_type) for axis in _tensor_values(
                    axes_tensor,
                    name=f"{node.op_type} axes",
                ))
            if (not axes and _integer_attribute(node, "noop_with_empty_axes", 0)):
                return data
            if not axes:
                axes = tuple(range(data.ndim))
        keep_dimensions = bool(_integer_attribute(node, "keepdims", 1))
        function = torch.sum if reduction == "sum" else torch.mean
        return function(data, dim=axes, keepdim=keep_dimensions)


SUPPORTED_ONNX_OPERATORS = _SUPPORTED_OPERATORS

__all__ = [
    "SUPPORTED_ONNX_OPERATORS",
    "NativeONNXError",
    "NativeONNXGraph",
    "ONNXExecutionError",
    "UnsupportedONNXGraphError",
]
