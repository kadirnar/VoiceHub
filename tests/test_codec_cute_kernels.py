from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from voicehub.components.audio.codecs.dac.nn.layers import Snake1d
from voicehub.components.audio.codecs.dac.nn.quantize import VectorQuantize
from voicehub.kernels import (
    AUDIO_CODEC_EUCLIDEAN_VQ,
    CodecKernelBackend,
    CodecKernelBackendUnavailableError,
    codec_euclidean_vq_search_reference,
    cute_codecs,
    cute_operator_capability,
)
from voicehub.kernels.capabilities import CapabilityStatus
from voicehub.optimization.capabilities import OptimizationContext
from voicehub.optimization.codecs import CodecOptimizationConfig, resolve_codec_optimization


class _FakeGemmArguments:

    def __init__(self, *, A, B, out, accumulator_type):
        self.A = A
        self.B = B
        self.out = out
        self.accumulator_type = accumulator_type


class _FakeGemmOperator:

    def __init__(self) -> None:
        self.compile_calls = 0
        self.run_calls = 0

    def compile(self, arguments, *, target_sm):
        self.compile_calls += 1
        return ("artifact", target_sm, arguments.accumulator_type)

    def get_workspace_size(self, arguments):
        del arguments
        return SimpleNamespace(size_bytes=0)

    def run(
        self,
        arguments,
        *,
        compiled_artifact,
        stream,
        workspace,
        assume_supported_args,
    ):
        del compiled_artifact, stream, workspace
        if not assume_supported_args:
            raise AssertionError("compiled arguments must be marked supported")
        self.run_calls += 1
        arguments.out.copy_(arguments.A @ arguments.B)


class _FakeOperators:
    GemmArguments = _FakeGemmArguments

    def __init__(self) -> None:
        self.operator = _FakeGemmOperator()
        self.queries = []

    def get_operators(self, arguments, *, target_sm):
        self.queries.append((arguments, target_sm))
        return [self.operator]


class CodecCuTeKernelTests(unittest.TestCase):

    def tearDown(self) -> None:
        cute_codecs.clear_codec_cute_gemm_cache()

    def test_reference_search_matches_the_original_distance_formula(self):
        torch.manual_seed(13)
        encodings = torch.randn(7, 4)
        codebook = torch.randn(11, 4)

        expected_distances = (
            encodings.square().sum(1, keepdim=True) - 2 * encodings @ codebook.transpose(0, 1) +
            codebook.square().sum(1, keepdim=True).transpose(0, 1))
        actual = codec_euclidean_vq_search_reference(encodings, codebook)

        torch.testing.assert_close(actual, expected_distances.argmin(dim=1))

    def test_cute_adapter_uses_operator_api_and_reuses_compiled_plan(self):
        operators = _FakeOperators()
        left = torch.randn(5, 3)
        right = torch.randn(3, 7)
        first = torch.empty(5, 7)
        second = torch.empty(5, 7)

        for output in (first, second):
            cute_codecs._execute_cute_gemm(
                operators,
                left,
                right,
                output,
                target_sm="80",
                stream=None,
            )

        torch.testing.assert_close(first, left @ right)
        torch.testing.assert_close(second, left @ right)
        self.assertEqual(operators.operator.compile_calls, 1)
        self.assertEqual(operators.operator.run_calls, 2)
        self.assertEqual(len(operators.queries), 1)

    def test_cute_adapter_will_not_compile_during_graph_capture(self):
        operators = _FakeOperators()
        with self.assertRaisesRegex(
                CodecKernelBackendUnavailableError,
                "warmup call before CUDA Graph capture",
        ):
            cute_codecs._execute_cute_gemm(
                operators,
                torch.randn(5, 3),
                torch.randn(3, 7),
                torch.empty(5, 7),
                target_sm="80",
                stream=None,
                allow_compile=False,
            )

        self.assertEqual(operators.operator.compile_calls, 0)
        self.assertEqual(operators.operator.run_calls, 0)

    def test_cute_public_operation_rejects_cpu_before_optional_import(self):
        with mock.patch.object(cute_codecs, "import_module") as import_module:
            with self.assertRaisesRegex(
                    CodecKernelBackendUnavailableError,
                    "requires CUDA",
            ):
                cute_codecs.codec_euclidean_vq_search_cute(
                    torch.randn(3, 2),
                    torch.randn(5, 2),
                )

        import_module.assert_not_called()

    def test_cute_custom_op_exposes_fake_tensor_shape_contract(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            encodings = torch.empty(
                5,
                3,
                device="cuda",
                dtype=torch.float16,
            )
            codebook = torch.empty(
                7,
                3,
                device="cuda",
                dtype=torch.float16,
            )
            indices = cute_codecs.codec_euclidean_vq_search_cute(
                encodings,
                codebook,
            )

        self.assertEqual(indices.shape, (5, ))
        self.assertEqual(indices.dtype, torch.int64)
        self.assertEqual(indices.device.type, "cuda")

    def test_dac_vq_declares_only_the_operation_it_accelerates(self):
        quantizer = VectorQuantize(
            input_dim=8,
            codebook_size=16,
            codebook_dim=4,
        )
        keys = tuple(quantizer.state_dict())

        self.assertEqual(
            quantizer.supported_kernel_operations,
            (AUDIO_CODEC_EUCLIDEAN_VQ, ),
        )
        self.assertEqual(
            quantizer.supported_codec_kernel_backends,
            (CodecKernelBackend.TORCH, CodecKernelBackend.CUTE),
        )
        quantizer.set_codec_kernel_backend("cute")
        self.assertIs(
            quantizer.codec_kernel_backend,
            CodecKernelBackend.CUTE,
        )
        self.assertEqual(tuple(quantizer.state_dict()), keys)

    def test_codec_pass_routes_cute_to_vq_and_torch_to_snake(self):
        model = nn.Sequential(
            VectorQuantize(
                input_dim=8,
                codebook_size=16,
                codebook_dim=4,
            ),
            Snake1d(8),
        )
        context = OptimizationContext(
            mode="inference",
            device="cuda",
            dtype="float16",
        )
        plan = resolve_codec_optimization(
            model,
            CodecOptimizationConfig(
                policy="relaxed",
                kernel_backend="cute",
                compile=False,
            ),
            context=context,
        )

        available = CapabilityStatus(
            True,
            "mock CuTe operator API",
        )
        with mock.patch(
                "voicehub.optimization.codec_accelerators."
                "cute_operator_capability",
                return_value=available,
        ):
            result = plan.apply(model)

        self.assertIs(
            model[0].codec_kernel_backend,
            CodecKernelBackend.CUTE,
        )
        self.assertIs(
            model[1].codec_kernel_backend,
            CodecKernelBackend.TORCH,
        )
        metadata = result.manifest_metadata()[0]["metadata"]
        self.assertEqual(metadata["selection"], "mixed:cute/torch")
        self.assertEqual(
            metadata["kernel_operations"],
            [AUDIO_CODEC_EUCLIDEAN_VQ, "audio.codec.snake"],
        )
        result.restore()
        self.assertIs(
            model[0].codec_kernel_backend,
            CodecKernelBackend.TORCH,
        )

    def test_vq_only_codec_rejects_an_explicit_unimplemented_backend(self):
        model = nn.Sequential(VectorQuantize(
            input_dim=8,
            codebook_size=16,
            codebook_dim=4,
        ), )
        with self.assertRaisesRegex(
                ValueError,
                "registered 'triton' implementation",
        ):
            resolve_codec_optimization(
                model,
                CodecOptimizationConfig(
                    policy="relaxed",
                    kernel_backend="triton",
                    compile=False,
                ),
                context=OptimizationContext(
                    mode="inference",
                    device="cuda",
                    dtype="float16",
                ),
            )

    def test_vq_only_relaxed_auto_reports_torch_when_triton_is_irrelevant(self):
        model = nn.Sequential(VectorQuantize(
            input_dim=8,
            codebook_size=16,
            codebook_dim=4,
        ), )
        context = OptimizationContext(
            mode="inference",
            device="cuda",
            dtype="float16",
        )

        with (
                mock.patch(
                    "voicehub.kernels.cuda_extensions."
                    "CUDA_EXTENSIONS.is_loaded",
                    return_value=False,
                ),
                mock.patch(
                    "voicehub.kernels.capabilities.triton_capability",
                    return_value=CapabilityStatus(
                        True,
                        "mock Triton",
                    ),
                ),
        ):
            plan = resolve_codec_optimization(
                model,
                CodecOptimizationConfig(
                    policy="relaxed",
                    kernel_backend="auto",
                    compile=False,
                ),
                context=context,
            )

        self.assertIs(
            plan.passes[0].backend,
            CodecKernelBackend.TORCH,
        )
        self.assertEqual(plan.decisions[1].selected, "torch")

    def test_operator_capability_fails_closed_when_gemm_api_is_missing(self):
        cuda = CapabilityStatus(
            True,
            "mock CUDA",
            {"compute_capability": "8.0"},
        )

        def fake_import(name):
            if name == "cutlass":
                return SimpleNamespace(__version__="4.6")
            if name == "cutlass.cute":
                return SimpleNamespace()
            if name == "cutlass.operators":
                return SimpleNamespace(GemmArguments=object)
            raise AssertionError(name)

        with (
                mock.patch(
                    "voicehub.kernels.capabilities.sys.platform",
                    "linux",
                ),
                mock.patch(
                    "voicehub.kernels.capabilities.cuda_runtime_capability",
                    return_value=cuda,
                ),
                mock.patch(
                    "voicehub.kernels.capabilities.import_module",
                    side_effect=fake_import,
                ),
        ):
            capability = cute_operator_capability("cuda")

        self.assertFalse(capability.available)
        self.assertIn("get_operators", capability.reason)

    def test_operator_capability_rejects_pre_ampere_before_api_import(self):
        cute = CapabilityStatus(
            True,
            "mock CuTe",
            {"compute_capability": "7.5"},
        )
        with (
                mock.patch(
                    "voicehub.kernels.capabilities.cute_dsl_capability",
                    return_value=cute,
                ),
                mock.patch("voicehub.kernels.capabilities.import_module", ) as import_module,
        ):
            capability = cute_operator_capability("cuda")

        self.assertFalse(capability.available)
        self.assertIn("Ampere-or-newer", capability.reason)
        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
