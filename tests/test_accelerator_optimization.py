from __future__ import annotations

import json
import subprocess
import sys
import unittest
from unittest import mock

import torch
from torch import nn

from voicehub.kernels import KernelBackend
from voicehub.neural.backends import FlashAttention4Policy
from voicehub.optimization import (
    OPTIMIZATION_PASSES,
    AcceleratorStateDictError,
    CodecKernelPass,
    CustomKernelPass,
    FlashAttention4Pass,
    OptimizationApplicationError,
    OptimizationCompatibilityError,
    OptimizationContext,
    OptimizationPassManager,
    accelerators,
)
from voicehub.training.adapters import BaseTrainingAdapter


class _FlashAttentionLeaf(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.flash_attention4_policy = FlashAttention4Policy.DISABLED
        self.calls = []

    def set_flash_attention4_policy(self, policy):
        self.flash_attention4_policy = FlashAttention4Policy.coerce(policy)
        self.calls.append(self.flash_attention4_policy)


class _KernelLeaf(nn.Module):

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.kernel_backend = KernelBackend.TORCH
        self.calls = []

    def set_kernel_backend(self, backend):
        self.kernel_backend = KernelBackend.coerce(backend)
        self.calls.append(self.kernel_backend)


class _MockTrainingAdapter(BaseTrainingAdapter):

    def __init__(self):
        self.primary_model = nn.Sequential(_KernelLeaf())
        self.auxiliary = nn.Sequential(_KernelLeaf())
        self._components = [
            ("primary-duplicate", self.primary_model),
            ("auxiliary", self.auxiliary),
            ("auxiliary-duplicate", self.auxiliary),
        ]

    def state_dict(self):
        return {
            "components": {
                "primary": self.primary_model.state_dict(),
                "auxiliary": self.auxiliary.state_dict(),
            },
        }


def _context(
    mode: str = "inference",
    *,
    device: str = "cpu",
    dtype: str = "float32",
) -> OptimizationContext:
    return OptimizationContext(
        mode=mode,
        device=device,
        dtype=dtype,
        persist_result=mode == "training",
    )


class AcceleratorOptimizationTests(unittest.TestCase):

    def test_named_defaults_are_registered_without_importing_accelerators(self):
        code = (
            "import json,sys;"
            "import voicehub.optimization as optimization;"
            "print(json.dumps({"
            "'loaded':'voicehub.optimization.accelerators' in sys.modules,"
            "'passes':optimization.OPTIMIZATION_PASSES.list()}))")
        result = subprocess.run(
            (sys.executable, "-c", code),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertFalse(payload["loaded"])
        self.assertIn("flash-attention-4", payload["passes"])
        self.assertIn("custom-kernels", payload["passes"])

        flash = OPTIMIZATION_PASSES.create("flash-attention-4")
        kernels = OPTIMIZATION_PASSES.create("custom-kernels")
        self.assertIsInstance(flash, FlashAttention4Pass)
        self.assertIsInstance(kernels, CustomKernelPass)
        self.assertEqual(flash.policy, FlashAttention4Policy.AUTO)
        self.assertEqual(kernels.backend, KernelBackend.AUTO)

    def test_flash_pass_traverses_module_and_restores_unique_targets(self):
        model = nn.ModuleDict({
            "first": _FlashAttentionLeaf(),
            "nested": nn.Sequential(_FlashAttentionLeaf()),
        })
        original_keys = tuple(model.state_dict())
        optimization_pass = FlashAttention4Pass(policy="auto")
        result = OptimizationPassManager().apply(
            model,
            (optimization_pass, ),
            _context(),
        )

        leaves = (model["first"], model["nested"][0])
        self.assertTrue(all(leaf.flash_attention4_policy is FlashAttention4Policy.AUTO for leaf in leaves))
        self.assertEqual(tuple(model.state_dict()), original_keys)
        manifest = result.manifest()
        json.dumps(manifest, allow_nan=False)
        entry = manifest["passes"][0]
        self.assertEqual(entry["kind"], "attention-backend")
        self.assertEqual(entry["configuration"], {"policy": "auto"})
        self.assertEqual(entry["metadata"]["target_count"], 2)
        self.assertEqual(
            entry["metadata"]["targets"],
            ["model.first", "model.nested.0"],
        )
        self.assertTrue(entry["metadata"]["state_dict_safe"])

        restored = result.restore()
        self.assertIs(restored, model)
        self.assertTrue(
            all(leaf.flash_attention4_policy is FlashAttention4Policy.DISABLED for leaf in leaves))
        self.assertEqual(tuple(restored.state_dict()), original_keys)

    def test_custom_kernel_pass_traverses_adapter_components_once(self):
        adapter = _MockTrainingAdapter()
        original_keys = tuple(adapter.state_dict()["components"])
        result = OptimizationPassManager().apply(
            adapter,
            (CustomKernelPass(backend="auto"), ),
            _context("training"),
        )

        primary = adapter.primary_model[0]
        auxiliary = adapter.auxiliary[0]
        self.assertIs(primary.kernel_backend, KernelBackend.AUTO)
        self.assertIs(auxiliary.kernel_backend, KernelBackend.AUTO)
        self.assertEqual(primary.calls, [KernelBackend.AUTO])
        self.assertEqual(auxiliary.calls, [KernelBackend.AUTO])
        metadata = result.manifest_metadata()[0]["metadata"]
        self.assertEqual(metadata["target_count"], 2)
        self.assertEqual(
            metadata["targets"],
            [
                "primary_model.0",
                "component:auxiliary.0",
            ],
        )
        self.assertEqual(metadata["previous_selections"], ["torch"])
        self.assertEqual(tuple(adapter.state_dict()["components"]), original_keys)

        result.restore()
        self.assertIs(primary.kernel_backend, KernelBackend.TORCH)
        self.assertIs(auxiliary.kernel_backend, KernelBackend.TORCH)
        self.assertEqual(
            primary.calls,
            [KernelBackend.AUTO, KernelBackend.TORCH],
        )
        self.assertEqual(
            auxiliary.calls,
            [KernelBackend.AUTO, KernelBackend.TORCH],
        )

    def test_passes_report_absent_targets_and_fail_closed_on_malformed_targets(self):
        for optimization_pass in (
                CodecKernelPass(),
                FlashAttention4Pass(),
                CustomKernelPass(),
        ):
            with self.subTest(optimization_pass=optimization_pass.pass_id):
                model = nn.Linear(2, 2)
                original_keys = tuple(model.state_dict())
                result = OptimizationPassManager().apply(
                    model,
                    (optimization_pass, ),
                    _context(),
                )
                metadata = result.manifest_metadata()[0]["metadata"]

                self.assertIs(result.model, model)
                self.assertEqual(metadata["outcome"], "not-applicable")
                self.assertIn("no submodule exposing", metadata["reason"])
                self.assertEqual(tuple(model.state_dict()), original_keys)
                self.assertIs(result.restore(), model)
                self.assertEqual(tuple(model.state_dict()), original_keys)

        class MissingState(nn.Module):

            def set_kernel_backend(self, backend):
                del backend

        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "reversible selector state",
        ):
            OptimizationPassManager().apply(
                MissingState(),
                (CustomKernelPass(), ),
                _context(),
            )

    def test_explicit_requirements_are_validated_against_context(self):
        flash_model = _FlashAttentionLeaf()
        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "needs a CUDA context.*float16 or bfloat16",
        ):
            OptimizationPassManager().apply(
                flash_model,
                (FlashAttention4Pass(policy="required"), ),
                _context(),
            )

        kernel_model = _KernelLeaf()
        with self.assertRaisesRegex(
                OptimizationCompatibilityError,
                "triton custom kernels need a CUDA context",
        ):
            OptimizationPassManager().apply(
                kernel_model,
                (CustomKernelPass(backend="triton"), ),
                _context(),
            )

    def test_explicit_triton_is_preloaded_for_a_plain_selector(self):
        model = _KernelLeaf()
        available = mock.Mock(available=True, reason="")
        with (
                mock.patch.object(
                    accelerators,
                    "triton_capability",
                    return_value=available,
                ),
                mock.patch.object(
                    accelerators,
                    "load_tts_activation_triton_kernels",
                ) as preload,
        ):
            result = OptimizationPassManager().apply(
                model,
                (CustomKernelPass(backend="triton"), ),
                _context(device="cuda", dtype="float16"),
            )

        preload.assert_called_once_with("cuda")
        self.assertIs(model.kernel_backend, KernelBackend.TRITON)
        result.restore()

    def test_cuda_extension_backend_must_be_preloaded_and_never_compiles(self):
        model = _KernelLeaf()
        optimization_pass = CustomKernelPass(backend="cuda-extension")
        cuda_context = _context(device="cuda", dtype="float16")

        with (
                mock.patch.object(
                    accelerators.CUDA_EXTENSIONS,
                    "is_loaded",
                    return_value=False,
                ),
                mock.patch("voicehub.kernels.cuda_extensions._compile_extension", ) as compile_extension,
                self.assertRaisesRegex(
                    OptimizationCompatibilityError,
                    "not already loaded",
                ),
        ):
            OptimizationPassManager().apply(
                model,
                (optimization_pass, ),
                cuda_context,
            )
        compile_extension.assert_not_called()
        self.assertIs(model.kernel_backend, KernelBackend.TORCH)

        with (
                mock.patch.object(
                    accelerators.CUDA_EXTENSIONS,
                    "is_loaded",
                    return_value=True,
                ),
                mock.patch("voicehub.kernels.cuda_extensions._compile_extension", ) as compile_extension,
        ):
            result = OptimizationPassManager().apply(
                model,
                (optimization_pass, ),
                cuda_context,
            )
            self.assertIs(
                model.kernel_backend,
                KernelBackend.CUDA_EXTENSION,
            )
            self.assertTrue(result.manifest_metadata()[0]["metadata"]["cuda_extension_loaded"])
            result.restore()
        compile_extension.assert_not_called()
        self.assertIs(model.kernel_backend, KernelBackend.TORCH)

        for dtype in ("float16", "bfloat16", "float32"):
            with (
                    self.subTest(dtype=dtype),
                    mock.patch.object(
                        accelerators.CUDA_EXTENSIONS,
                        "is_loaded",
                        return_value=True,
                    ),
                    mock.patch("voicehub.kernels.cuda_extensions._compile_extension", ) as compile_extension,
            ):
                result = OptimizationPassManager().apply(
                    model,
                    (optimization_pass, ),
                    _context("training", device="cuda", dtype=dtype),
                )
                self.assertIs(
                    model.kernel_backend,
                    KernelBackend.CUDA_EXTENSION,
                )
                result.restore()
                compile_extension.assert_not_called()
                self.assertIs(model.kernel_backend, KernelBackend.TORCH)

    def test_partial_selector_failure_is_rolled_back(self):

        class FailingKernelLeaf(_KernelLeaf):

            def set_kernel_backend(self, backend):
                selected = KernelBackend.coerce(backend)
                if selected is KernelBackend.AUTO:
                    raise RuntimeError("deliberate selector failure")
                super().set_kernel_backend(selected)

        model = nn.Sequential(_KernelLeaf(), FailingKernelLeaf())
        with self.assertRaises(OptimizationApplicationError) as context:
            OptimizationPassManager().apply(
                model,
                (CustomKernelPass(), ),
                _context(),
            )

        self.assertIn("deliberate selector failure", str(context.exception))
        self.assertIs(model[0].kernel_backend, KernelBackend.TORCH)
        self.assertIs(model[1].kernel_backend, KernelBackend.TORCH)

    def test_state_dict_key_changes_are_rejected_and_restored(self):

        class TopologyChangingLeaf(_KernelLeaf):

            def set_kernel_backend(self, backend):
                selected = KernelBackend.coerce(backend)
                self.kernel_backend = selected
                if selected is KernelBackend.AUTO:
                    self.register_buffer("unexpected", torch.ones(()))
                elif "unexpected" in self._buffers:
                    delattr(self, "unexpected")

        model = TopologyChangingLeaf()
        original_keys = tuple(model.state_dict())
        with self.assertRaises(OptimizationApplicationError) as context:
            OptimizationPassManager().apply(
                model,
                (CustomKernelPass(), ),
                _context(),
            )

        self.assertIsInstance(
            context.exception.cause,
            AcceleratorStateDictError,
        )
        self.assertIs(model.kernel_backend, KernelBackend.TORCH)
        self.assertEqual(tuple(model.state_dict()), original_keys)

    def test_configuration_is_normalized_and_strict_json(self):
        flash = FlashAttention4Pass(policy="DISABLED")
        kernels = CustomKernelPass(backend="cuda-extension")
        self.assertEqual(
            flash.manifest_configuration(),
            {"policy": "disabled"},
        )
        self.assertEqual(
            kernels.manifest_configuration(),
            {"backend": "cuda_extension"},
        )
        json.dumps(flash.manifest_configuration(), allow_nan=False)
        json.dumps(kernels.manifest_configuration(), allow_nan=False)
        with self.assertRaises(ValueError):
            FlashAttention4Pass(policy="sometimes")
        with self.assertRaises(ValueError):
            CustomKernelPass(backend="compiler")
        with self.assertRaises(TypeError):
            CustomKernelPass(backend=object())


if __name__ == "__main__":
    unittest.main()
