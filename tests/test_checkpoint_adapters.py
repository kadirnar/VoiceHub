from __future__ import annotations

import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from voicehub.checkpointing import (
    CastTensor,
    CheckpointAdapter,
    CheckpointCompatibilityError,
    ConcatenateTensors,
    CopyTensor,
    ReshapeTensor,
    SplitTensor,
    SqueezeTensor,
    TensorPlan,
    TransposeTensor,
)


@unittest.skipUnless(torch is not None, "Native checkpoint adapters use PyTorch")
class TensorPlanTests(unittest.TestCase):

    def test_explicit_tensor_operations_and_lazy_source_coverage(self):
        source = {
            "packed": torch.arange(12, dtype=torch.float32).reshape(6, 2),
            "left": torch.ones(1, 2),
            "right": torch.zeros(1, 2),
            "matrix": torch.arange(6).reshape(2, 3),
            "singleton": torch.ones(1, 2, 1),
            "flat": torch.arange(6),
            "integer": torch.tensor([1, 2], dtype=torch.int32),
        }
        plan = TensorPlan(
            rules=(
                SplitTensor(
                    "packed",
                    ("q", "k", "v"),
                    sizes=(2, 2, 2),
                ),
                ConcatenateTensors(("left", "right"), "joined"),
                TransposeTensor("matrix", "transposed", (1, 0)),
                SqueezeTensor("singleton", "squeezed", (0, 2)),
                ReshapeTensor("flat", "reshaped", (2, 3)),
                CastTensor("integer", "floating", torch.float32),
            ), )

        converted, consumed = plan.materialize(source)

        self.assertEqual(consumed, frozenset(source))
        self.assertEqual(tuple(converted["q"].shape), (2, 2))
        self.assertEqual(tuple(converted["joined"].shape), (2, 2))
        self.assertEqual(tuple(converted["transposed"].shape), (3, 2))
        self.assertEqual(tuple(converted["squeezed"].shape), (2, ))
        self.assertEqual(tuple(converted["reshaped"].shape), (2, 3))
        self.assertEqual(converted["floating"].dtype, torch.float32)

    def test_duplicate_canonical_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate targets"):
            TensorPlan(
                rules=(
                    CopyTensor("a", "same"),
                    CopyTensor("b", "same"),
                ), )


@unittest.skipUnless(torch is not None, "Native checkpoint adapters use PyTorch")
class CheckpointAdapterTests(unittest.TestCase):

    class _Model(torch.nn.Module if torch is not None else object):

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.register_buffer("scale", torch.zeros(1))

    class _Adapter(CheckpointAdapter):
        architecture_id = "test"
        adapter_id = "test-upstream"
        adapter_version = "1"

        def probe(self, files: tuple[Path, ...], config):
            return any(path.name == "model.safetensors" for path in files)

        def tensor_plan(self, config):
            return TensorPlan(
                rules=(
                    CopyTensor("upstream.weight", "weight"),
                    CopyTensor("upstream.scale", "scale"),
                ),
                ignored_source_patterns=("optimizer.*", ),
            )

    def test_strict_load_requires_total_explained_coverage(self):
        model = self._Model()
        source = {
            "upstream.weight": torch.ones(2, 2),
            "upstream.scale": torch.ones(1),
            "optimizer.step": torch.tensor(10),
        }

        report = self._Adapter().load(model, source, {}, strict=True)

        self.assertTrue(report.is_compatible)
        self.assertEqual(report.ignored_sources, ("optimizer.step", ))
        torch.testing.assert_close(model.weight, torch.ones(2, 2))
        torch.testing.assert_close(model.scale, torch.ones(1))

    def test_strict_load_validates_before_mutating_model(self):
        model = self._Model()
        source = {
            "upstream.weight": torch.ones(3, 2),
            "upstream.scale": torch.ones(1),
            "unexplained": torch.tensor(1),
        }

        with self.assertRaises(CheckpointCompatibilityError):
            self._Adapter().load(model, source, {}, strict=True)

        torch.testing.assert_close(model.weight, torch.zeros(2, 2))
        torch.testing.assert_close(model.scale, torch.zeros(1))

    def test_non_strict_load_reports_and_loads_only_compatible_tensors(self):
        model = self._Model()
        source = {
            "upstream.weight": torch.ones(3, 2),
            "upstream.scale": torch.ones(1),
        }

        report = self._Adapter().load(model, source, {}, strict=False)

        self.assertFalse(report.is_compatible)
        self.assertEqual(
            tuple(item.name for item in report.shape_mismatches),
            ("weight", ),
        )
        torch.testing.assert_close(model.weight, torch.zeros(2, 2))
        torch.testing.assert_close(model.scale, torch.ones(1))

    def test_streaming_load_validates_headers_then_reads_each_tensor(self):
        model = self._Model()

        class Source:

            def __init__(self):
                self.tensors = {
                    "upstream.weight": torch.ones(2, 2),
                    "upstream.scale": torch.ones(1),
                    "optimizer.step": torch.tensor(10),
                }
                self.events = []

            def keys(self):
                return tuple(self.tensors)

            def tensor_shape(self, name):
                self.events.append(("shape", name))
                return tuple(self.tensors[name].shape)

            def get_tensor(self, name):
                self.events.append(("read", name))
                return self.tensors[name]

        source = Source()
        report = self._Adapter().load_streaming(
            model,
            source,
            {},
            strict=True,
        )

        self.assertTrue(report.is_compatible)
        self.assertEqual(
            source.events,
            [
                ("shape", "upstream.scale"),
                ("shape", "upstream.weight"),
                ("read", "upstream.scale"),
                ("read", "upstream.weight"),
            ],
        )
        torch.testing.assert_close(model.weight, torch.ones(2, 2))
        torch.testing.assert_close(model.scale, torch.ones(1))

    def test_streaming_load_does_not_read_payload_after_failed_validation(self):
        model = self._Model()

        class Source:

            def keys(self):
                return ("upstream.weight", "upstream.scale")

            def tensor_shape(self, name):
                return (3, 2) if name == "upstream.weight" else (1, )

            def get_tensor(self, name):
                raise AssertionError(f"read incompatible payload {name}")

        with self.assertRaises(CheckpointCompatibilityError):
            self._Adapter().load_streaming(
                model,
                Source(),
                {},
                strict=True,
            )

        torch.testing.assert_close(model.weight, torch.zeros(2, 2))
        torch.testing.assert_close(model.scale, torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
