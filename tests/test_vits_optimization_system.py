from __future__ import annotations

import unittest
from unittest import mock

import torch
from torch import nn

from tests.test_native_vits_model import _tiny_config
from voicehub import (
    VITSArchitectureKind,
    VITSCUDAGraphPolicy,
    VITSOptimizationConfig,
    get_tts_optimization_support,
    get_vits_model_optimization_support,
    list_vits_model_optimization_support,
    vits_acceleration_plan,
)
from voicehub.architectures.gptsovits.modeling import PosteriorEncoder
from voicehub.architectures.inflecttts.modules import WN as InflectWaveNet
from voicehub.architectures.vits.modeling import VitsModel, VitsWaveNet
from voicehub.kernels import CapabilityStatus, KernelBackend, fused_add_tanh_sigmoid, fused_add_tanh_sigmoid_reference
from voicehub.models.melotts.source.melo.modules import WN as MeloWaveNet
from voicehub.models.openvoice.source.openvoice.modules import WN as OpenVoiceWaveNet
from voicehub.optimization import (
    CustomKernelPass,
    OptimizationCompatibilityError,
    OptimizationContext,
    OptimizationPassManager,
    accelerators,
)


class VITSFamilyInventoryTests(unittest.TestCase):

    def test_inventory_is_trait_driven_and_exact(self):
        support = list_vits_model_optimization_support()

        self.assertEqual(
            [item.model_type for item in support],
            ["gptsovits", "inflecttts", "melotts", "openvoice", "vits"],
        )
        self.assertEqual(
            {item.kind
             for item in support},
            {
                VITSArchitectureKind.CLASSIC,
                VITSArchitectureKind.VITS2,
                VITSArchitectureKind.HYBRID_ACOUSTIC,
                VITSArchitectureKind.CONVERTER,
            },
        )
        for item in support:
            with self.subTest(model_type=item.model_type):
                self.assertIn("compile", item.optimization_passes)
                self.assertIn("custom-kernels", item.optimization_passes)
                self.assertEqual(
                    item.kernel_operations,
                    ("tts.vits.fused_add_tanh_sigmoid", ),
                )
                universal = get_tts_optimization_support(item.model_type)
                self.assertIn("triton", universal.kernel_backends)
                self.assertIn("cuda_extension", universal.kernel_backends)

    def test_aliases_resolve_without_misclassifying_compatibility_cohorts(self):
        self.assertEqual(
            get_vits_model_optimization_support("mms-tts").model_type,
            "vits",
        )
        for model_type in ("styletts2", "xtts"):
            with (
                    self.subTest(model_type=model_type),
                    self.assertRaisesRegex(ValueError, "not a registered VITS"),
            ):
                get_vits_model_optimization_support(model_type)


class VITSFusedGateTests(unittest.TestCase):

    def test_broadcast_noncontiguous_forward_and_backward_match_torch(self):
        torch.manual_seed(17)
        input_a = (torch.randn(2, 9, 8, dtype=torch.float64).transpose(1, 2).requires_grad_())
        input_b = torch.randn(
            2,
            8,
            1,
            dtype=torch.float64,
            requires_grad=True,
        )
        actual = fused_add_tanh_sigmoid(
            input_a,
            input_b,
            4,
            backend="torch",
        )
        expected_a = input_a.detach().clone().requires_grad_()
        expected_b = input_b.detach().clone().requires_grad_()
        expected = fused_add_tanh_sigmoid_reference(
            expected_a,
            expected_b,
            4,
        )
        gradient = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            actual,
            (input_a, input_b),
            gradient,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (expected_a, expected_b),
            gradient,
        )

        torch.testing.assert_close(actual, expected)
        for value, reference in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(value, reference)
        self.assertEqual(actual_gradients[1].shape, (2, 8, 1))

    def test_empty_frames_use_the_portable_fallback(self):
        input_a = torch.empty(2, 8, 0)
        input_b = torch.empty(2, 8, 1)
        output = fused_add_tanh_sigmoid(input_a, input_b, 4)
        self.assertEqual(output.shape, (2, 4, 0))

    def test_real_wavenet_boundary_compiles_fullgraph(self):
        model = VitsWaveNet(_tiny_config(), num_layers=1).eval()
        compiled = torch.compile(
            model,
            backend="aot_eager",
            fullgraph=True,
        )
        inputs = torch.randn(1, model.hidden_size, 7)
        mask = torch.ones(1, 1, 7)
        torch.testing.assert_close(
            compiled(inputs, mask),
            model(inputs, mask),
        )


class VITSStructuralKernelTests(unittest.TestCase):

    @staticmethod
    def _blocks() -> tuple[nn.Module, ...]:
        native = VitsWaveNet(_tiny_config(), num_layers=1)
        inflect = InflectWaveNet(4, 3, 1, 1)
        melo = MeloWaveNet(4, 3, 1, 1)
        openvoice = OpenVoiceWaveNet(4, 3, 1, 1)
        gptsovits = PosteriorEncoder(4, 2, 4, 3, 1, 1).enc
        return native, inflect, melo, openvoice, gptsovits

    def test_all_active_wavenet_implementations_share_one_protocol(self):
        for block in self._blocks():
            with self.subTest(block=type(block).__qualname__):
                keys = tuple(block.state_dict())
                self.assertEqual(
                    block.supported_kernel_operations,
                    ("tts.vits.fused_add_tanh_sigmoid", ),
                )
                block.set_kernel_backend("torch")
                self.assertIs(block.kernel_backend, KernelBackend.TORCH)
                self.assertEqual(tuple(block.state_dict()), keys)

    def test_auto_is_resolved_before_graph_capture_and_restored(self):
        model = nn.ModuleList(self._blocks())
        keys = tuple(model.state_dict())
        result = OptimizationPassManager().apply(
            model,
            (CustomKernelPass(backend="auto"), ),
            OptimizationContext(
                mode="training",
                device="cpu",
                dtype="float32",
                persist_result=True,
            ),
        )

        self.assertTrue(all(block.kernel_backend is KernelBackend.TORCH for block in model))
        metadata = result.manifest_metadata()[0]["metadata"]
        self.assertEqual(metadata["requested_selection"], "auto")
        self.assertEqual(metadata["selection"], "torch")
        self.assertEqual(
            metadata["kernel_operations"],
            ["tts.vits.fused_add_tanh_sigmoid"],
        )
        self.assertEqual(tuple(model.state_dict()), keys)
        result.restore()
        self.assertEqual(tuple(model.state_dict()), keys)

    def test_explicit_missing_triton_fails_before_model_mutation(self):
        model = self._blocks()[0]
        unavailable = CapabilityStatus(
            False,
            "test Triton installation is absent",
        )
        with (
                mock.patch.object(
                    accelerators,
                    "triton_capability",
                    return_value=unavailable,
                ),
                self.assertRaisesRegex(
                    OptimizationCompatibilityError,
                    "triton backend is unavailable",
                ),
        ):
            OptimizationPassManager().apply(
                model,
                (CustomKernelPass(backend="triton"), ),
                OptimizationContext(
                    mode="inference",
                    device="cuda",
                    dtype="float16",
                ),
            )
        self.assertIs(model.kernel_backend, KernelBackend.TORCH)

    def test_native_vits_compile_target_matches_public_inference(self):
        model = VitsModel.__new__(VitsModel)
        nn.Module.__init__(model)
        inference = model.optimization_compile_targets("inference")
        training = model.optimization_compile_targets("training")
        self.assertEqual(inference[0].attribute, "synthesize")
        self.assertEqual(training[0].attribute, "forward")


class VITSAccelerationPresetTests(unittest.TestCase):

    def test_cuda_graph_policy_requires_static_shapes(self):
        graph_plan = vits_acceleration_plan(cuda_graphs=True)
        compile_config = graph_plan[-1].config
        self.assertEqual(compile_config.mode, "reduce-overhead")
        self.assertFalse(compile_config.dynamic)
        with self.assertRaisesRegex(ValueError, "static/bucketed"):
            vits_acceleration_plan(
                cuda_graphs=VITSCUDAGraphPolicy.REQUIRED,
                compile_dynamic=True,
            )

    def test_default_training_profile_keeps_dynamic_graphs_safe(self):
        plan = VITSOptimizationConfig().acceleration_plan()
        compile_config = plan[-1].config
        self.assertEqual(
            compile_config.mode,
            "max-autotune-no-cudagraphs",
        )
        self.assertTrue(compile_config.dynamic)
        arguments = VITSOptimizationConfig().training_arguments("output")
        self.assertTrue(arguments.adamw_fused)
        self.assertFalse(arguments.adamw_torch_compile)


if __name__ == "__main__":
    unittest.main()
