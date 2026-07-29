from __future__ import annotations

import unittest

import torch
from torch import nn

from voicehub.optimization import LoRAConfig, LoRALinear, inject_lora


class _AttentionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.block = nn.Module()
        self.block.v_proj = nn.Linear(4, 4, bias=True)

    def forward(self, inputs):
        return self.q_proj(inputs) + self.k_proj(inputs) + self.block.v_proj(inputs)


class NativeLoRATests(unittest.TestCase):

    def test_injection_is_initially_output_equivalent_and_freezes_only_targets(self):
        model = _AttentionModel()
        inputs = torch.randn(2, 4)
        expected = model(inputs)

        injection = inject_lora(
            model,
            LoRAConfig(
                rank=2,
                target_modules=("q_proj", "block.v_proj"),
            ),
        )

        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        self.assertEqual(
            injection.module_names,
            ("block.v_proj", "q_proj"),
        )
        self.assertIsInstance(model.q_proj, LoRALinear)
        self.assertFalse(model.q_proj.base.weight.requires_grad)
        self.assertTrue(model.k_proj.weight.requires_grad)

    def test_adapter_gradients_merge_and_unmerge_preserve_outputs(self):
        model = _AttentionModel()
        injection = inject_lora(
            model,
            LoRAConfig(rank=2, target_modules=("q_proj",), dropout=0.0),
        )
        with torch.no_grad():
            model.q_proj.lora_b.fill_(0.25)
        inputs = torch.randn(3, 4)
        expected = model(inputs)

        injection.merge()
        torch.testing.assert_close(model(inputs), expected, rtol=1e-6, atol=1e-6)
        injection.unmerge()
        torch.testing.assert_close(model(inputs), expected, rtol=1e-6, atol=1e-6)

        model(inputs).square().mean().backward()
        self.assertIsNotNone(model.q_proj.lora_a.grad)
        self.assertIsNotNone(model.q_proj.lora_b.grad)
        self.assertIsNone(model.q_proj.base.weight.grad)

    def test_adapter_state_load_is_shape_checked_before_copy(self):
        model = _AttentionModel()
        injection = inject_lora(
            model,
            LoRAConfig(rank=2, target_modules=("q_proj",)),
        )
        original = injection.adapter_state_dict()
        invalid = dict(original)
        invalid["q_proj.lora_a"] = torch.zeros(3, 4)

        with self.assertRaisesRegex(ValueError, "shape"):
            injection.load_adapter_state_dict(invalid)

        current = injection.adapter_state_dict()
        for key in original:
            torch.testing.assert_close(current[key], original[key])

    def test_restore_recovers_original_graph_and_trainability(self):
        model = _AttentionModel()
        original_q = model.q_proj
        injection = inject_lora(
            model,
            LoRAConfig(rank=2, target_modules=("q_proj",)),
        )

        restored = injection.restore()

        self.assertIs(restored, model)
        self.assertIs(model.q_proj, original_q)
        self.assertTrue(model.q_proj.weight.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "already been restored"):
            injection.merge()

    def test_incompatible_rank_fails_before_any_replacement(self):
        model = _AttentionModel()
        original_q = model.q_proj
        with self.assertRaisesRegex(ValueError, "rank"):
            inject_lora(
                model,
                LoRAConfig(rank=5, target_modules=("q_proj",)),
            )
        self.assertIs(model.q_proj, original_q)


if __name__ == "__main__":
    unittest.main()
