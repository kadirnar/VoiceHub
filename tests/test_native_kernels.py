from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.kernels import (
    ACTIVATION_CUDA_EXTENSION_NAME,
    CUDA_EXTENSIONS,
    DIFFUSION_FUSED_BIAS_GELU,
    LLM_GATED_SILU,
    VITS_FUSED_ADD_TANH_SIGMOID,
    VITS_TANH_SIGMOID_GATE,
    CapabilityStatus,
    CudaExtensionRegistry,
    CudaExtensionSpec,
    CudaExtensionUnavailableError,
    KernelBackend,
    KernelDispatchError,
    KernelRegistrationError,
    KernelRegistry,
    KernelSupport,
    cuda_extension_capability,
    fused_add_tanh_sigmoid,
    fused_add_tanh_sigmoid_reference,
    fused_bias_gelu,
    fused_bias_gelu_reference,
    gated_silu,
    gated_silu_reference,
    load_tts_activation_cuda_extension,
    resolve_kernel,
    tanh_sigmoid_gate,
    tanh_sigmoid_gate_reference,
    triton_capability,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_TRITON_KERNELS = (os.environ.get("VOICEHUB_TEST_TRITON_KERNELS") == "1" and triton_capability().available)
RUN_CUDA_EXTENSION = (
    os.environ.get("VOICEHUB_TEST_CUDA_EXTENSIONS") == "1" and cuda_extension_capability().available)


def _clone_kernel_arguments(arguments):
    return tuple(
        argument.detach().clone(memory_format=torch.preserve_format, ).
        requires_grad_(True) if isinstance(argument, torch.Tensor) else argument for argument in arguments)


class KernelRegistryTests(unittest.TestCase):

    def test_auto_dispatch_skips_an_unavailable_high_priority_backend(self):
        registry = KernelRegistry()
        registry.register(
            "test.operation",
            KernelBackend.TORCH,
            lambda value: value + 1,
        )
        registry.register(
            "test.operation",
            KernelBackend.TRITON,
            lambda value: value + 100,
            priority=100,
            support_check=lambda value: KernelSupport(
                False,
                "test device is CPU-only",
            ),
        )

        self.assertEqual(registry.dispatch("test.operation", 2), 3)
        self.assertEqual(
            registry.resolve("test.operation", 2).backend,
            KernelBackend.TORCH,
        )
        with self.assertRaisesRegex(KernelDispatchError, "CPU-only"):
            registry.dispatch(
                "test.operation",
                2,
                backend=KernelBackend.TRITON,
            )

    def test_duplicate_registration_requires_explicit_replacement(self):
        registry = KernelRegistry()
        registry.register("test.operation", "torch", lambda: 1)

        with self.assertRaises(KernelRegistrationError):
            registry.register("test.operation", "torch", lambda: 2)

        registry.register(
            "test.operation",
            "torch",
            lambda: 3,
            replace=True,
        )
        self.assertEqual(registry.dispatch("test.operation"), 3)

    def test_capability_exceptions_are_reported_without_running_kernel(self):
        registry = KernelRegistry()

        def broken_support():
            raise RuntimeError("probe failed")

        registry.register(
            "test.operation",
            "triton",
            lambda: self.fail("unavailable kernel was executed"),
            support_check=broken_support,
        )
        with self.assertRaisesRegex(KernelDispatchError, "probe failed"):
            registry.dispatch("test.operation")


class NativeActivationKernelTests(unittest.TestCase):

    def test_import_is_lazy_for_optional_compilers(self):
        code = """
import json
import sys
import voicehub.kernels
print(json.dumps({
    "triton": "triton" in sys.modules,
    "cpp_extension": "torch.utils.cpp_extension" in sys.modules,
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.stdout.strip(),
            '{"triton": false, "cpp_extension": false}',
        )

    def test_cpu_auto_dispatch_uses_torch_for_every_tts_family(self):
        gate = torch.randn(2, 3)
        up = torch.randn(2, 3)
        activation = torch.randn(2, 3)
        vits_input = torch.randn(2, 6, 5)
        vits_condition = torch.randn(2, 6, 1)
        inputs = torch.randn(2, 4)
        bias = torch.randn(4)

        torch.testing.assert_close(
            gated_silu(gate, up),
            gated_silu_reference(gate, up),
        )
        torch.testing.assert_close(
            tanh_sigmoid_gate(activation, gate),
            tanh_sigmoid_gate_reference(activation, gate),
        )
        torch.testing.assert_close(
            fused_add_tanh_sigmoid(vits_input, vits_condition, 3),
            fused_add_tanh_sigmoid_reference(
                vits_input,
                vits_condition,
                3,
            ),
        )
        torch.testing.assert_close(
            fused_bias_gelu(inputs, bias),
            fused_bias_gelu_reference(inputs, bias),
        )
        for operation, args in (
            (LLM_GATED_SILU, (gate, up)),
            (VITS_TANH_SIGMOID_GATE, (activation, gate)),
            (
                VITS_FUSED_ADD_TANH_SIGMOID,
                (vits_input, vits_condition, 3),
            ),
            (DIFFUSION_FUSED_BIAS_GELU, (inputs, bias)),
        ):
            with self.subTest(operation=operation):
                self.assertEqual(
                    resolve_kernel(operation, *args).backend,
                    KernelBackend.TORCH,
                )

    def test_cpu_fallback_does_not_call_lazy_triton_import(self):
        gate = torch.randn(2, 3)
        up = torch.randn(2, 3)

        with patch("voicehub.kernels.activations.import_module") as optional_import:
            output = gated_silu(gate, up)

        torch.testing.assert_close(output, gated_silu_reference(gate, up))
        optional_import.assert_not_called()

    def test_triton_bridge_uses_compile_composable_library_contracts(self):
        source = (PROJECT_ROOT / "voicehub" / "kernels" / "triton_activations.py").read_text(encoding="utf-8")
        self.assertIn("torch.library.triton_op", source)
        self.assertIn("torch.library.wrap_triton", source)
        self.assertIn("torch.library.register_autograd", source)
        self.assertNotIn("torch.autograd.Function", source)

    def test_structured_triton_ops_register_with_a_mocked_lazy_module(self):
        code = """
import json
import sys
from types import ModuleType

triton = ModuleType("triton")
triton.jit = lambda function: function
triton.__version__ = "mock"
language = ModuleType("triton.language")
language.constexpr = object()
triton.language = language
sys.modules["triton"] = triton
sys.modules["triton.language"] = language

from voicehub.kernels import triton_activations

print(json.dumps([
    str(triton_activations.gated_silu_triton),
    str(triton_activations.tanh_sigmoid_gate_triton),
    str(triton_activations.fused_add_tanh_sigmoid_triton),
    str(triton_activations.fused_bias_gelu_triton),
]))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("voicehub_triton::gated_silu", result.stdout)
        self.assertIn("voicehub_triton::tanh_sigmoid_gate", result.stdout)
        self.assertIn(
            "voicehub_triton::fused_add_tanh_sigmoid",
            result.stdout,
        )
        self.assertIn("voicehub_triton::fused_bias_gelu", result.stdout)

    def test_torch_fallback_gradients_cover_all_inputs_and_bias(self):
        gate = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        up = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
        activation = torch.randn(
            2,
            3,
            dtype=torch.float64,
            requires_grad=True,
        )
        inputs = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float64, requires_grad=True)

        for function, arguments in (
            (gated_silu, (gate, up)),
            (tanh_sigmoid_gate, (activation, gate)),
            (fused_bias_gelu, (inputs, bias)),
        ):
            with self.subTest(function=function.__name__):
                output = function(
                    *arguments,
                    backend=KernelBackend.TORCH,
                )
                gradients = torch.autograd.grad(output.sum(), arguments)
                self.assertEqual(len(gradients), len(arguments))
                for argument, gradient in zip(arguments, gradients):
                    self.assertEqual(gradient.shape, argument.shape)
                    self.assertTrue(torch.isfinite(gradient).all())

    def test_explicit_triton_rejects_cpu_without_importing_triton(self):
        gate = torch.randn(2, 3)
        up = torch.randn(2, 3)

        with self.assertRaisesRegex(KernelDispatchError, "CUDA"):
            gated_silu(gate, up, backend=KernelBackend.TRITON)

    def test_input_contracts_reject_implicit_broadcasting(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            gated_silu(torch.randn(2, 3), torch.randn(3))
        with self.assertRaisesRegex(ValueError, "last dimension"):
            fused_bias_gelu(torch.randn(2, 4), torch.randn(3))
        with self.assertRaisesRegex(TypeError, "floating-point"):
            tanh_sigmoid_gate(
                torch.ones(2, dtype=torch.long),
                torch.ones(2, dtype=torch.long),
            )


class CudaExtensionInfrastructureTests(unittest.TestCase):

    def test_registered_extension_points_to_real_cpp_and_cuda_sources(self):
        spec = CUDA_EXTENSIONS.get(ACTIVATION_CUDA_EXTENSION_NAME)
        suffixes = {source.suffix for source in spec.sources}
        self.assertEqual(suffixes, {".cpp", ".cu"})
        self.assertTrue(all(source.is_file() for source in spec.sources))
        cuda_source = next(
            source.read_text(encoding="utf-8") for source in spec.sources if source.suffix == ".cu")
        for kernel_name in (
                "gated_silu_kernel",
                "tanh_sigmoid_gate_kernel",
                "fused_add_tanh_sigmoid_kernel",
                "fused_bias_gelu_kernel",
        ):
            self.assertIn(kernel_name, cuda_source)

    def test_cuda_python_registrations_pass_opcheck_and_compile_without_cuda(self):
        code = """
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.nn import functional as F
from voicehub.kernels import activations

definitions = torch.library.Library("voicehub_kernels", "DEF")
definitions.define("gated_silu(Tensor gate, Tensor up) -> Tensor")
definitions.define(
    "tanh_sigmoid_gate(Tensor activation, Tensor gate) -> Tensor"
)
definitions.define(
    "fused_add_tanh_sigmoid(Tensor input_a, Tensor input_b, int channels) "
    "-> Tensor"
)
definitions.define("fused_bias_gelu(Tensor input, Tensor bias) -> Tensor")
implementations = torch.library.Library("voicehub_kernels", "IMPL", "CPU")
implementations.impl(
    "gated_silu",
    lambda gate, up: (F.silu(gate) * up).contiguous(),
)
implementations.impl(
    "tanh_sigmoid_gate",
    lambda activation, gate: (
        torch.tanh(activation) * torch.sigmoid(gate)
    ).contiguous(),
)
implementations.impl(
    "fused_add_tanh_sigmoid",
    lambda input_a, input_b, channels: (
        torch.tanh((input_a + input_b)[:, :channels])
        * torch.sigmoid((input_a + input_b)[:, channels:])
    ).contiguous(),
)
implementations.impl(
    "fused_bias_gelu",
    lambda inputs, bias: F.gelu(
        inputs + bias,
        approximate="tanh",
    ).contiguous(),
)

with ThreadPoolExecutor(max_workers=4) as executor:
    tuple(executor.map(
        lambda _: activations._register_cuda_autograd(),
        range(8),
    ))
registration_library = activations._CUDA_REGISTRATION_LIBRARY
activations._register_cuda_autograd()
assert activations._CUDA_REGISTRATION_LIBRARY is registration_library

cases = (
    (
        "gated_silu",
        (
            torch.randn(4, 3).t().requires_grad_(),
            torch.randn(4, 3).t().requires_grad_(),
        ),
    ),
    (
        "tanh_sigmoid_gate",
        (
            torch.randn(4, 3).t().requires_grad_(),
            torch.randn(4, 3).t().requires_grad_(),
        ),
    ),
    (
        "fused_add_tanh_sigmoid",
        (
            torch.randn(2, 8, 5).requires_grad_(),
            torch.randn(2, 8, 1).requires_grad_(),
            4,
        ),
    ),
    (
        "fused_bias_gelu",
        (
            torch.randn(2, 4, 3).transpose(1, 2).requires_grad_(),
            torch.randn(4, requires_grad=True),
        ),
    ),
)
for name, arguments in cases:
    operation = getattr(torch.ops.voicehub_kernels, name).default
    results = torch.library.opcheck(operation, arguments)
    assert set(results.values()) == {"SUCCESS"}, results
    compiled = torch.compile(
        operation,
        backend="aot_eager",
        fullgraph=True,
    )
    compiled_arguments = tuple(
        (
            argument.detach().clone(
                memory_format=torch.preserve_format,
            ).requires_grad_(True)
            if isinstance(argument, torch.Tensor)
            else argument
        )
        for argument in arguments
    )
    output = compiled(*compiled_arguments)
    assert output.is_contiguous()
    tensor_arguments = tuple(
        argument
        for argument in compiled_arguments
        if isinstance(argument, torch.Tensor)
    )
    gradients = torch.autograd.grad(
        output.sum(),
        tensor_arguments,
    )
    assert all(
        gradient.shape == argument.shape
        for gradient, argument in zip(gradients, tensor_arguments)
    )

print("cuda registrations passed")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "cuda registrations passed")

    def test_loader_is_explicit_idempotent_and_mockable_without_compilation(self):
        registry = CudaExtensionRegistry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpp_source = root / "test.cpp"
            cuda_source = root / "test.cu"
            cpp_source.write_text("// registration", encoding="utf-8")
            cuda_source.write_text("// CUDA kernel", encoding="utf-8")
            spec = CudaExtensionSpec(
                name="voicehub_test_extension",
                sources=(cpp_source, cuda_source),
            )
            registry.register(spec)
            capability = CapabilityStatus(
                True,
                "mock CUDA toolchain",
            )
            sentinel = object()
            with (
                    patch(
                        "voicehub.kernels.cuda_extensions.cuda_extension_capability",
                        return_value=capability,
                    ),
                    patch(
                        "voicehub.kernels.cuda_extensions._compile_extension",
                        return_value=sentinel,
                    ) as compile_extension,
            ):
                first = registry.load("voicehub_test_extension")
                second = registry.load("voicehub_test_extension")

        self.assertIs(first, second)
        self.assertIs(first.module, sentinel)
        compile_extension.assert_called_once()

    def test_unavailable_toolchain_never_invokes_compiler(self):
        registry = CudaExtensionRegistry()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpp_source = root / "test.cpp"
            cuda_source = root / "test.cu"
            cpp_source.write_text("// registration", encoding="utf-8")
            cuda_source.write_text("// CUDA kernel", encoding="utf-8")
            registry.register(
                CudaExtensionSpec(
                    name="voicehub_unavailable_extension",
                    sources=(cpp_source, cuda_source),
                ))
            capability = CapabilityStatus(
                False,
                "CUDA toolkit is absent",
            )
            with (
                    patch(
                        "voicehub.kernels.cuda_extensions.cuda_extension_capability",
                        return_value=capability,
                    ),
                    patch("voicehub.kernels.cuda_extensions._compile_extension", ) as compile_extension,
                    self.assertRaisesRegex(
                        CudaExtensionUnavailableError,
                        "toolkit is absent",
                    ),
            ):
                registry.load("voicehub_unavailable_extension")
        compile_extension.assert_not_called()


@unittest.skipUnless(
    RUN_TRITON_KERNELS,
    "Set VOICEHUB_TEST_TRITON_KERNELS=1 on a Triton CUDA host.",
)
class TritonActivationKernelTests(unittest.TestCase):

    def _compare_forward_and_backward(
        self,
        function,
        reference,
        arguments,
    ):
        triton_arguments = _clone_kernel_arguments(arguments)
        reference_arguments = _clone_kernel_arguments(arguments)
        triton_output = function(
            *triton_arguments,
            backend=KernelBackend.TRITON,
        )
        reference_output = reference(*reference_arguments)
        gradient = torch.randn_like(triton_output)
        triton_tensors = tuple(
            argument for argument in triton_arguments if isinstance(argument, torch.Tensor))
        reference_tensors = tuple(
            argument for argument in reference_arguments if isinstance(argument, torch.Tensor))
        triton_gradients = torch.autograd.grad(
            triton_output,
            triton_tensors,
            gradient,
        )
        reference_gradients = torch.autograd.grad(
            reference_output,
            reference_tensors,
            gradient,
        )
        torch.testing.assert_close(
            triton_output,
            reference_output,
            rtol=2e-4,
            atol=2e-5,
        )
        for actual, expected in zip(triton_gradients, reference_gradients):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=5e-4,
                atol=5e-5,
            )

    def test_triton_training_matches_torch_for_all_architectures(self):
        device = torch.device("cuda")
        self._compare_forward_and_backward(
            gated_silu,
            gated_silu_reference,
            (
                torch.randn(8, 256, device=device),
                torch.randn(8, 256, device=device),
            ),
        )
        self._compare_forward_and_backward(
            tanh_sigmoid_gate,
            tanh_sigmoid_gate_reference,
            (
                torch.randn(8, 256, device=device),
                torch.randn(8, 256, device=device),
            ),
        )
        self._compare_forward_and_backward(
            fused_add_tanh_sigmoid,
            fused_add_tanh_sigmoid_reference,
            (
                torch.randn(2, 257, 128, device=device).transpose(1, 2),
                torch.randn(2, 128, 1, device=device),
                64,
            ),
        )
        self._compare_forward_and_backward(
            fused_bias_gelu,
            fused_bias_gelu_reference,
            (
                torch.randn(4, 8, 256, device=device),
                torch.randn(256, device=device),
            ),
        )


@unittest.skipUnless(
    RUN_CUDA_EXTENSION,
    "Set VOICEHUB_TEST_CUDA_EXTENSIONS=1 on a CUDA toolkit host.",
)
class CompiledCudaActivationKernelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_tts_activation_cuda_extension()

    def test_compiled_cuda_kernels_match_torch_forward_and_backward(self):
        device = torch.device("cuda")
        gate = torch.randn(8, 256, device=device)
        up = torch.randn(8, 256, device=device)
        activation = torch.randn(8, 256, device=device)
        vits_input = torch.randn(2, 257, 128, device=device).transpose(1, 2)
        vits_condition = torch.randn(2, 128, 1, device=device)
        inputs = torch.randn(4, 8, 256, device=device)
        bias = torch.randn(256, device=device)
        for function, reference, arguments in (
            (gated_silu, gated_silu_reference, (gate, up)),
            (
                tanh_sigmoid_gate,
                tanh_sigmoid_gate_reference,
                (activation, gate),
            ),
            (
                fused_add_tanh_sigmoid,
                fused_add_tanh_sigmoid_reference,
                (vits_input, vits_condition, 64),
            ),
            (
                fused_bias_gelu,
                fused_bias_gelu_reference,
                (inputs, bias),
            ),
        ):
            with self.subTest(function=function.__name__):
                cuda_arguments = _clone_kernel_arguments(arguments)
                reference_arguments = _clone_kernel_arguments(arguments)
                actual = function(
                    *cuda_arguments,
                    backend=KernelBackend.CUDA_EXTENSION,
                )
                expected = reference(*reference_arguments)
                gradient = torch.randn_like(actual)
                cuda_tensors = tuple(
                    argument for argument in cuda_arguments if isinstance(argument, torch.Tensor))
                reference_tensors = tuple(
                    argument for argument in reference_arguments if isinstance(argument, torch.Tensor))
                actual_gradients = torch.autograd.grad(
                    actual,
                    cuda_tensors,
                    gradient,
                )
                expected_gradients = torch.autograd.grad(
                    expected,
                    reference_tensors,
                    gradient,
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-4,
                    atol=2e-5,
                )
                for actual_gradient, expected_gradient in zip(
                        actual_gradients,
                        expected_gradients,
                ):
                    torch.testing.assert_close(
                        actual_gradient,
                        expected_gradient,
                        rtol=5e-4,
                        atol=5e-5,
                    )

    def test_compiled_cuda_operator_has_fake_tensor_and_autograd_support(self):
        compiled = torch.compile(
            torch.ops.voicehub_kernels.gated_silu.default,
            backend="inductor",
            fullgraph=True,
        )
        gate = torch.randn(256, 8, device="cuda").t().requires_grad_()
        up = torch.randn(256, 8, device="cuda").t().requires_grad_()
        reference_gate = gate.detach().clone(memory_format=torch.preserve_format, ).requires_grad_()
        reference_up = up.detach().clone(memory_format=torch.preserve_format, ).requires_grad_()
        actual = compiled(gate, up)
        expected = gated_silu_reference(reference_gate, reference_up)
        gradient = torch.randn_like(actual)
        actual_gradients = torch.autograd.grad(
            actual,
            (gate, up),
            gradient,
        )
        expected_gradients = torch.autograd.grad(
            expected,
            (reference_gate, reference_up),
            gradient,
        )
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-4,
            atol=2e-5,
        )
        for actual_gradient, expected_gradient in zip(
                actual_gradients,
                expected_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient,
                expected_gradient,
                rtol=5e-4,
                atol=5e-5,
            )


if __name__ == "__main__":
    unittest.main()
