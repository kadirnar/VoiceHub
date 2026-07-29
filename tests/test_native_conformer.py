from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch

from voicehub.components.neural.conformer.conformer import (
    Attention,
    Conformer,
    ConformerConvModule,
)


class NativeConformerTests(unittest.TestCase):

    def test_shared_conformer_uses_only_torch_and_the_standard_library(self):
        source = (
            Path(__file__).parents[1]
            / "voicehub"
            / "components"
            / "neural"
            / "conformer"
            / "conformer.py"
        )
        tree = ast.parse(source.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertTrue(
            all(
                name == "__future__"
                or name.split(".", 1)[0] in {"torch"}
                for name in imports
            )
        )

    def test_attention_supports_masked_self_and_cross_attention(self):
        torch.manual_seed(7)
        attention = Attention(
            dim=12,
            heads=3,
            dim_head=4,
            dropout=0.0,
            max_pos_emb=8,
        )
        values = torch.randn(2, 5, 12, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
            ]
        )
        self_output = attention(values, mask=mask)
        self.assertEqual(self_output.shape, values.shape)

        context = torch.randn(2, 3, 12)
        context_mask = torch.tensor(
            [[True, True, False], [True, True, True]]
        )
        cross_output = attention(
            values,
            context=context,
            mask=mask,
            context_mask=context_mask,
        )
        self.assertEqual(cross_output.shape, values.shape)
        (self_output.square().mean() + cross_output.square().mean()).backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_convolution_preserves_time_for_causal_and_symmetric_padding(self):
        values = torch.randn(2, 11, 8, requires_grad=True)
        for causal in (False, True):
            with self.subTest(causal=causal):
                module = ConformerConvModule(
                    dim=8,
                    causal=causal,
                    expansion_factor=2,
                    kernel_size=5,
                )
                output = module(values)
                self.assertEqual(output.shape, values.shape)
                self.assertTrue(torch.isfinite(output).all())

    def test_stack_forwards_masks_and_allows_end_to_end_gradients(self):
        model = Conformer(
            8,
            depth=2,
            dim_head=4,
            heads=2,
            conv_kernel_size=5,
        )
        values = torch.randn(2, 7, 8, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, False, False, False],
                [True, True, True, True, True, True, True],
            ]
        )
        output = model(values, mask=mask)
        self.assertEqual(output.shape, values.shape)
        output.square().mean().backward()
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())


if __name__ == "__main__":
    unittest.main()
