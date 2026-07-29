from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import torch

from voicehub.checkpointing import (
    ONNXAttribute,
    ONNXGraph,
    ONNXModel,
    ONNXNode,
    ONNXTensor,
    ONNXValueInfo,
    onnx_semantic_fingerprint,
    read_onnx_model,
)
from voicehub.neural.onnx import NativeONNXGraph, ONNXExecutionError, UnsupportedONNXGraphError


def _varint(value: int) -> bytes:
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _varint_field(number: int, value: int) -> bytes:
    return _varint(number << 3) + _varint(value)


def _bytes_field(number: int, value: bytes) -> bytes:
    return (_varint((number << 3) | 2) + _varint(len(value)) + value)


def _value_info(name: str, shape: tuple[int, ...]) -> ONNXValueInfo:
    return ONNXValueInfo(
        name=name,
        element_type=1,
        shape=shape,
    )


def _model(
    *,
    inputs: tuple[ONNXValueInfo, ...],
    outputs: tuple[ONNXValueInfo, ...],
    nodes: tuple[ONNXNode, ...],
    initializers: dict[str, ONNXTensor],
) -> ONNXModel:
    return ONNXModel(
        ir_version=9,
        producer_name="voicehub-test",
        producer_version="1",
        domain="",
        model_version=1,
        opsets=(("", 19), ),
        metadata=MappingProxyType({}),
        graph=ONNXGraph(
            name="test",
            inputs=inputs,
            outputs=outputs,
            nodes=nodes,
            initializers=MappingProxyType(initializers),
        ),
    )


class NativeONNXRuntimeTests(unittest.TestCase):

    def test_affine_graph_is_differentiable_and_uses_original_state_names(self):
        graph = _model(
            inputs=(_value_info("x", (1, 2)), ),
            outputs=(_value_info("y", (1, 2)), ),
            nodes=(
                ONNXNode(
                    op_type="MatMul",
                    domain="",
                    inputs=("x", "weight"),
                    outputs=("projected", ),
                ),
                ONNXNode(
                    op_type="Add",
                    domain="",
                    inputs=("projected", "bias"),
                    outputs=("y", ),
                ),
            ),
            initializers={
                "bias":
                ONNXTensor(
                    name="bias",
                    data_type=1,
                    dimensions=(2, ),
                    raw_data=b"",
                    float_data=(0.5, -1.0),
                ),
                "weight":
                ONNXTensor(
                    name="weight",
                    data_type=1,
                    dimensions=(2, 2),
                    raw_data=b"",
                    float_data=(1.0, 2.0, 3.0, 4.0),
                ),
            },
        )
        runtime = NativeONNXGraph(graph)
        inputs = torch.tensor([[2.0, -1.0]], requires_grad=True)

        output = runtime(x=inputs)
        torch.testing.assert_close(
            output,
            torch.tensor([[-0.5, -1.0]]),
        )
        output.square().sum().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertEqual(
            runtime.trainable_initializer_names,
            ("bias", "weight"),
        )
        self.assertEqual(
            tuple(runtime.onnx_state_dict()),
            ("bias", "weight"),
        )
        self.assertTrue(all(parameter.grad is not None for parameter in runtime.parameters()))

    def test_multidirectional_expand_preserves_existing_non_singleton_axes(self):
        graph = _model(
            inputs=(_value_info("x", (1, 3, 1)), ),
            outputs=(_value_info("y", (1, 3, 2)), ),
            nodes=(ONNXNode(
                op_type="Expand",
                domain="",
                inputs=("x", "shape"),
                outputs=("y", ),
            ), ),
            initializers={
                "shape":
                ONNXTensor(
                    name="shape",
                    data_type=7,
                    dimensions=(3, ),
                    raw_data=b"",
                    int64_data=(1, 1, 2),
                ),
            },
        )
        runtime = NativeONNXGraph(graph, trainable=False)
        values = torch.tensor([[[1.0], [2.0], [3.0]]])

        actual = runtime(x=values)

        self.assertEqual(actual.shape, (1, 3, 2))
        torch.testing.assert_close(
            actual,
            values.expand(1, 3, 2),
        )

    def test_state_loading_is_shape_checked_and_supports_partial_updates(self):
        graph = _model(
            inputs=(_value_info("x", (1, )), ),
            outputs=(_value_info("y", (1, )), ),
            nodes=(ONNXNode(
                op_type="Add",
                domain="",
                inputs=("x", "offset"),
                outputs=("y", ),
            ), ),
            initializers={
                "offset":
                ONNXTensor(
                    name="offset",
                    data_type=1,
                    dimensions=(1, ),
                    raw_data=b"",
                    float_data=(1.0, ),
                ),
            },
        )
        runtime = NativeONNXGraph(graph)

        missing, unexpected = runtime.load_onnx_state_dict({"offset": torch.tensor([2.0])}, )

        self.assertEqual((missing, unexpected), ((), ()))
        torch.testing.assert_close(
            runtime(x=torch.tensor([3.0])),
            torch.tensor([5.0]),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            runtime.load_onnx_state_dict({
                "offset": torch.ones(2),
            })

    def test_custom_domains_and_invalid_input_sets_are_rejected(self):
        graph = _model(
            inputs=(_value_info("x", (1, )), ),
            outputs=(_value_info("y", (1, )), ),
            nodes=(ONNXNode(
                op_type="Identity",
                domain="vendor.custom",
                inputs=("x", ),
                outputs=("y", ),
            ), ),
            initializers={},
        )
        with self.assertRaisesRegex(
                UnsupportedONNXGraphError,
                "Custom ONNX domains",
        ):
            NativeONNXGraph(graph)

        valid = _model(
            inputs=(_value_info("x", (1, )), ),
            outputs=(_value_info("y", (1, )), ),
            nodes=(ONNXNode(
                op_type="Identity",
                domain="",
                inputs=("x", ),
                outputs=("y", ),
            ), ),
            initializers={},
        )
        runtime = NativeONNXGraph(valid)
        with self.assertRaisesRegex(ONNXExecutionError, "missing x"):
            runtime()

    def test_semantic_fingerprint_ignores_weights_but_covers_attributes(self):

        def graph(
            *,
            weight: float,
            axis: int,
        ) -> ONNXModel:
            return _model(
                inputs=(_value_info("x", (1, )), ),
                outputs=(_value_info("y", (1, )), ),
                nodes=(
                    ONNXNode(
                        op_type="Softmax",
                        domain="",
                        inputs=("x", ),
                        outputs=("y", ),
                        attributes=MappingProxyType({
                            "axis":
                            ONNXAttribute(
                                name="axis",
                                attribute_type=2,
                                value=axis,
                            ),
                        }),
                    ), ),
                initializers={
                    "unused":
                    ONNXTensor(
                        name="unused",
                        data_type=1,
                        dimensions=(1, ),
                        raw_data=b"",
                        float_data=(weight, ),
                    ),
                },
            )

        first = onnx_semantic_fingerprint(graph(weight=1.0, axis=-1))
        changed_weight = onnx_semantic_fingerprint(graph(weight=2.0, axis=-1))
        changed_attribute = onnx_semantic_fingerprint(graph(weight=1.0, axis=0))

        self.assertEqual(first, changed_weight)
        self.assertNotEqual(first, changed_attribute)

    def test_dependency_free_reader_preserves_constant_tensor_attributes(self):
        tensor = (_varint_field(1, 2) + _varint_field(2, 1) + _bytes_field(9, struct.pack("<2f", 1.25, -2.5)))
        attribute = (_bytes_field(1, b"value") + _varint_field(20, 4) + _bytes_field(5, tensor))
        node = (_bytes_field(2, b"y") + _bytes_field(4, b"Constant") + _bytes_field(5, attribute))
        output = _bytes_field(1, b"y")
        graph = (_bytes_field(1, node) + _bytes_field(2, b"constant-test") + _bytes_field(12, output))
        opset = _varint_field(2, 19)
        model_bytes = (_varint_field(1, 9) + _bytes_field(7, graph) + _bytes_field(8, opset))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "constant.onnx"
            path.write_bytes(model_bytes)
            parsed = read_onnx_model(path)
            runtime = NativeONNXGraph(parsed, trainable=False)

            result = runtime()

        record = parsed.graph.nodes[0].attributes["value"]
        self.assertIsInstance(record, ONNXAttribute)
        self.assertEqual(record.attribute_type, 4)
        torch.testing.assert_close(
            record.value.to_torch(),
            torch.tensor([1.25, -2.5]),
        )
        torch.testing.assert_close(
            result,
            torch.tensor([1.25, -2.5]),
        )


if __name__ == "__main__":
    unittest.main()
