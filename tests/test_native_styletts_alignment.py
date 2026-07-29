from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from voicehub.models.styletts2.monotonic_align import mask_from_lens, maximum_path, maximum_path_c


class NativeStyleTTSAlignmentTests(unittest.TestCase):

    def test_alignment_module_has_no_array_or_extension_dependency(self):
        source = (Path(__file__).parents[1] / "voicehub" / "models" / "styletts2" / "monotonic_align.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertTrue(
            all(name == "__future__" or name.split(".", 1)[0] in {"torch", "voicehub"} for name in imports))

    def test_cython_shaped_api_mutates_tensor_storage_and_preserves_ties(self):
        values = torch.zeros(1, 2, 3)
        paths = torch.full((1, 2, 3), -1, dtype=torch.int32)

        maximum_path_c(
            paths,
            values,
            torch.tensor([2]),
            torch.tensor([3]),
        )

        expected = torch.tensor([[[1, 0, 0], [0, 1, 1]]], dtype=torch.int32)
        self.assertTrue(torch.equal(paths, expected))

    def test_public_alignment_respects_ragged_rectangular_masks(self):
        values = torch.tensor([
            [[4.0, 3.0, 0.0, 0.0], [0.0, 2.0, 5.0, 6.0]],
            [[1.0, 1.0, 1.0, 0.0], [0.0, 2.0, 3.0, 0.0]],
        ])
        mask = mask_from_lens(
            values,
            torch.tensor([2, 2]),
            torch.tensor([4, 3]),
        )

        paths = maximum_path(values, mask)

        self.assertEqual(paths.shape, values.shape)
        self.assertTrue(torch.equal(paths * (1 - mask), torch.zeros_like(paths)))
        self.assertTrue(
            torch.equal(
                paths.sum(dim=1),
                torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]),
            ))


if __name__ == "__main__":
    unittest.main()
