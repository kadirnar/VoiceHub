from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class Zonos2PortableMoETests(unittest.TestCase):

    @staticmethod
    def _fixture():
        import torch

        generator = torch.Generator().manual_seed(17)
        hidden_states = torch.randn(4, 3, generator=generator)
        first_weights = torch.randn(3, 6, 3, generator=generator)
        second_weights = torch.randn(3, 3, 3, generator=generator)
        route_weights = torch.tensor([
            [0.75, 0.25],
            [0.6, 0.4],
            [0.2, 0.8],
            [1.0, 0.0],
        ])
        route_ids = torch.tensor([
            [0, 1],
            [2, 0],
            [1, 2],
            [0, -1],
        ], dtype=torch.int32)
        return (
            hidden_states,
            first_weights,
            second_weights,
            route_weights,
            route_ids,
        )

    @staticmethod
    def _reference(
        hidden_states,
        first_weights,
        second_weights,
        route_weights,
        route_ids,
        *,
        activation,
        weight_first,
    ):
        import torch
        import torch.nn.functional as functional

        routed = hidden_states.new_zeros(
            hidden_states.shape[0],
            route_ids.shape[1],
            second_weights.shape[1],
        )
        for token_index in range(hidden_states.shape[0]):
            for route_index in range(route_ids.shape[1]):
                expert_index = int(route_ids[token_index, route_index])
                if expert_index < 0:
                    continue
                projected = torch.mv(
                    first_weights[expert_index],
                    hidden_states[token_index],
                )
                weight = route_weights[token_index, route_index]
                if weight_first:
                    projected = projected * weight
                gate, values = projected.chunk(2)
                if activation == "silu":
                    activated = functional.silu(gate) * values
                else:
                    activated = (functional.gelu(gate, approximate="none") * values)
                output = torch.mv(
                    second_weights[expert_index],
                    activated,
                )
                if not weight_first:
                    output = output * weight
                routed[token_index, route_index] = output
        return routed

    def test_torch_fallback_matches_reference_for_both_weight_modes(self):
        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import fused_moe_impl

        values = self._fixture()
        for activation in ("silu", "gelu"):
            for weight_first in (False, True):
                with self.subTest(
                        activation=activation,
                        weight_first=weight_first,
                ):
                    routed = self._reference(
                        *values,
                        activation=activation,
                        weight_first=weight_first,
                    )
                    with patch.object(
                            fused_moe_impl,
                            "_optimized_kernels_for",
                            return_value=None,
                    ):
                        output = fused_moe_impl.fused_experts(
                            *values,
                            activation=activation,
                            apply_router_weight_on_input=weight_first,
                            routed_scaling_factor=1.5,
                        )
                        uncombined = fused_moe_impl.fused_experts(
                            *values,
                            activation=activation,
                            apply_router_weight_on_input=weight_first,
                            no_combine=True,
                            routed_scaling_factor=1.5,
                        )
                    self.assertTrue(output.allclose(
                        routed.sum(dim=1) * 1.5,
                        atol=1e-6,
                        rtol=1e-5,
                    ))
                    self.assertTrue(uncombined.allclose(
                        routed,
                        atol=1e-6,
                        rtol=1e-5,
                    ))

    def test_inplace_fallback_updates_and_returns_the_input(self):
        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import fused_moe_impl

        values = list(self._fixture())
        expected = self._reference(
            *values,
            activation="silu",
            weight_first=False,
        ).sum(dim=1)
        with patch.object(
                fused_moe_impl,
                "_optimized_kernels_for",
                return_value=None,
        ):
            output = fused_moe_impl.fused_experts(
                *values,
                inplace=True,
            )
        self.assertIs(output, values[0])
        self.assertTrue(output.allclose(expected, atol=1e-6, rtol=1e-5))

    def test_optimized_dispatch_remains_available(self):
        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import fused_moe_impl

        values = list(self._fixture())
        values[4] = values[4].clone()
        values[4][values[4] < 0] = 1
        expected = values[0].new_full(values[0].shape, 3.0)
        kernels = object()
        optimized = Mock(return_value=expected)
        with (
                patch.object(
                    fused_moe_impl,
                    "_optimized_kernels_for",
                    return_value=kernels,
                ),
                patch.object(
                    fused_moe_impl,
                    "_fused_experts_optimized",
                    optimized,
                ),
        ):
            output = fused_moe_impl.fused_experts(*values)
        self.assertIs(output, expected)
        self.assertIs(optimized.call_args.kwargs["kernels"], kernels)

    def test_router_path_threads_first_projection_weighting(self):
        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import fused_moe_impl

        values = self._fixture()
        router_logits = values[0].new_zeros(
            values[0].shape[0],
            values[1].shape[0],
        )
        expected = values[0].new_zeros(values[0].shape)
        with (
                patch.object(
                    fused_moe_impl,
                    "select_experts",
                    return_value=(values[3], values[4]),
                ),
                patch.object(
                    fused_moe_impl,
                    "fused_experts",
                    return_value=expected,
                ) as experts,
        ):
            output = fused_moe_impl.fused_moe(
                values[0],
                values[1],
                values[2],
                router_logits,
                topk=2,
                renormalize=True,
                apply_router_weight_on_input=True,
            )
        self.assertIs(output, expected)
        self.assertTrue(experts.call_args.kwargs["apply_router_weight_on_input"])

    def test_single_route_scaling_matches_reference(self):
        import torch

        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import fused_moe_impl

        values = list(self._fixture())
        values[3] = values[3][:, :1].contiguous()
        values[4] = values[4][:, :1].contiguous()
        routed = self._reference(
            *values,
            activation="silu",
            weight_first=False,
        )
        with patch.object(
                fused_moe_impl,
                "_optimized_kernels_for",
                return_value=None,
        ):
            output = fused_moe_impl.fused_experts(
                *values,
                routed_scaling_factor=0.5,
            )
        self.assertTrue(torch.allclose(
            output,
            routed[:, 0] * 0.5,
            atol=1e-6,
            rtol=1e-5,
        ))
        empty = fused_moe_impl.fused_experts(
            values[0][:0],
            values[1],
            values[2],
            values[3][:0],
            values[4][:0],
        )
        self.assertEqual(tuple(empty.shape), (0, values[2].shape[1]))

    def test_router_topk_has_a_torch_fallback(self):
        import torch

        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import topk

        hidden_states = torch.zeros(2, 3)
        router_logits = torch.tensor([
            [1.0, 3.0, 2.0],
            [4.0, 1.0, 2.0],
        ])
        with patch.object(
                topk,
                "_load_sgl_topk",
                return_value=None,
        ):
            weights, expert_ids = topk.select_experts(
                hidden_states,
                router_logits,
                top_k=2,
                renormalize=True,
            )
        probabilities = torch.softmax(router_logits, dim=-1)
        expected_weights, expected_ids = probabilities.topk(2, dim=-1)
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
        self.assertTrue(weights.allclose(expected_weights))
        self.assertTrue(expert_ids.equal(expected_ids.to(torch.int32)))
        with patch.object(
                topk,
                "_load_sgl_topk",
                return_value=None,
        ):
            padded_weights, padded_ids = topk.select_experts(
                hidden_states,
                router_logits,
                top_k=2,
                renormalize=True,
                num_token_non_padded=torch.tensor(1),
            )
            empty_weights, empty_ids = topk.select_experts(
                hidden_states,
                router_logits,
                top_k=2,
                renormalize=True,
                num_token_non_padded=torch.tensor(0),
            )
        self.assertTrue(padded_ids[1].eq(-1).all())
        self.assertTrue(padded_weights[1].eq(0).all())
        self.assertTrue(empty_ids.eq(-1).all())
        self.assertTrue(empty_weights.eq(0).all())
        from voicehub.models.zonos2.source.zonos2.layers.moe.fused_moe import fused_moe_impl

        values = self._fixture()
        empty_output = fused_moe_impl.fused_experts(
            values[0][:2],
            values[1],
            values[2],
            empty_weights,
            empty_ids,
        )
        self.assertTrue(empty_output.eq(0).all())
        with self.assertRaisesRegex(ValueError, "same device"):
            topk.select_experts(
                hidden_states,
                torch.empty(2, 3, device="meta"),
                top_k=2,
            )

    def test_source_import_does_not_require_optimized_kernel_packages(self):
        script = """
import builtins
import torch

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in {"triton", "sgl_kernel"}:
        raise ModuleNotFoundError(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import voicehub.models.zonos2.source.zonos2.tts
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
