from __future__ import annotations

import json
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch.nn import functional

from voicehub.neural.backends import flash_attention4 as backend
from voicehub.neural.backends.flash_attention4 import (
    FLASH_ATTENTION4_TESTED_VERSION,
    FlashAttention4Capability,
    FlashAttention4CapabilityError,
    FlashAttention4ExecutionError,
    FlashAttention4UnavailableError,
    flash_attention4_or_sdpa,
)


def _supported() -> FlashAttention4Capability:
    return FlashAttention4Capability(
        supported=True,
        compute_capability=(9, 0),
    )


class FlashAttention4BackendTests(unittest.TestCase):

    def tearDown(self):
        backend._load_flash_attention4.cache_clear()

    def test_import_does_not_load_optional_flash_attention_package(self):
        code = (
            "import json,sys;"
            "import voicehub.neural.backends.flash_attention4;"
            "print(json.dumps(sorted(name for name in sys.modules "
            "if name == 'flash_attn' or name.startswith('flash_attn.'))))")
        result = subprocess.run(
            (sys.executable, "-c", code),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(json.loads(result.stdout), [])

    def test_auto_rejects_cpu_before_lazy_import_and_matches_sdpa(self):
        torch.manual_seed(13)
        query = torch.randn(2, 3, 4, 8)
        key = torch.randn(2, 3, 5, 8)
        value = torch.randn(2, 3, 5, 8)
        expected = functional.scaled_dot_product_attention(query, key, value, scale=0.25)

        with mock.patch.object(backend, "_load_flash_attention4") as load:
            actual = flash_attention4_or_sdpa(
                query,
                key,
                value,
                scale=0.25,
            )

        load.assert_not_called()
        torch.testing.assert_close(actual, expected)

    def test_required_policy_reports_static_capability_reasons(self):
        query = torch.randn(1, 2, 3, 8)

        with self.assertRaises(FlashAttention4CapabilityError) as context:
            flash_attention4_or_sdpa(
                query,
                query,
                query,
                policy="required",
            )

        message = str(context.exception)
        self.assertIn("must be CUDA tensors", message)
        self.assertIn("dtype must be torch.float16 or torch.bfloat16", message)
        self.assertIn("query=(1, 2, 3, 8)", message)

    def test_masks_and_attention_dropout_never_reach_dense_fa4(self):
        query = torch.randn(1, 2, 3, 8)
        mask = torch.ones(3, 3, dtype=torch.bool)
        sentinel = torch.empty_like(query)

        with (
                mock.patch.object(backend, "_load_flash_attention4") as load,
                mock.patch.object(backend, "_pytorch_sdpa", return_value=sentinel) as sdpa,
        ):
            actual = flash_attention4_or_sdpa(
                query,
                query,
                query,
                attention_mask=mask,
                dropout_p=0.1,
            )

        self.assertIs(actual, sentinel)
        load.assert_not_called()
        sdpa.assert_called_once()
        with self.assertRaisesRegex(
                FlashAttention4CapabilityError,
                "does not expose attention dropout.*varlen adapter",
        ):
            flash_attention4_or_sdpa(
                query,
                query,
                query,
                attention_mask=mask,
                dropout_p=0.1,
                policy="required",
            )

    def test_sdpa_fallback_matches_fa4_bottom_right_causal_alignment(self):
        query = torch.randn(1, 2, 2, 8)
        key = torch.randn(1, 2, 5, 8)
        value = torch.randn_like(key)
        sentinel = torch.empty_like(query)
        calls = []

        def capture_sdpa(query_states, key_states, value_states, **kwargs):
            calls.append(kwargs)
            return sentinel

        with mock.patch.object(
                backend.functional,
                "scaled_dot_product_attention",
                side_effect=capture_sdpa,
        ):
            actual = flash_attention4_or_sdpa(
                query,
                key,
                value,
                is_causal=True,
                policy="disabled",
            )

        self.assertIs(actual, sentinel)
        self.assertFalse(calls[0]["is_causal"])
        torch.testing.assert_close(
            calls[0]["attn_mask"],
            torch.tensor([
                [True, True, True, True, False],
                [True, True, True, True, True],
            ]),
        )

    def test_required_missing_package_has_pinned_install_guidance(self):
        query = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        backend._load_flash_attention4.cache_clear()

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(
                    backend,
                    "import_module",
                    side_effect=ModuleNotFoundError("No module named 'flash_attn'"),
                ),
                self.assertRaises(FlashAttention4UnavailableError) as context,
        ):
            flash_attention4_or_sdpa(
                query,
                query,
                query,
                policy="required",
            )

        message = str(context.exception)
        self.assertIn(f"flash-attn-4=={FLASH_ATTENTION4_TESTED_VERSION}", message)
        self.assertIn("pip install", message)
        self.assertIn(backend.FLASH_ATTENTION4_UPSTREAM_REVISION, message)

    def test_version_diagnostic_probes_official_and_internal_distribution_names(self):

        def distribution_version(name):
            if name == "flash-attn-4":
                raise backend.PackageNotFoundError
            if name == "fa4":
                return "4.0.0b24"
            raise AssertionError(name)

        with mock.patch.object(backend, "version", side_effect=distribution_version) as lookup:
            installed = backend._installed_version()

        self.assertEqual(installed, "4.0.0b24 (fa4)")
        self.assertEqual(
            [call.args[0] for call in lookup.call_args_list],
            ["flash-attn-4", "fa4"],
        )

    def test_auto_missing_package_falls_back_to_sdpa(self):
        query = torch.randn(1, 2, 3, 8)
        expected = torch.full_like(query, 7.0)

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(
                    backend,
                    "_load_flash_attention4",
                    side_effect=FlashAttention4UnavailableError("not installed"),
                ),
                mock.patch.object(backend, "_pytorch_sdpa", return_value=expected) as sdpa,
        ):
            actual = flash_attention4_or_sdpa(query, query, query)

        self.assertIs(actual, expected)
        sdpa.assert_called_once()

    def test_sdpa_fallback_expands_gqa_when_native_keyword_is_unavailable(self):
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn_like(key)
        calls = []

        def capture_sdpa(query_states, key_states, value_states, **kwargs):
            calls.append((query_states.shape, key_states.shape, value_states.shape, kwargs))
            return torch.zeros_like(query_states)

        with (
                mock.patch.object(backend, "_SDPA_SUPPORTS_GQA", False),
                mock.patch.object(
                    backend.functional,
                    "scaled_dot_product_attention",
                    side_effect=capture_sdpa,
                ),
        ):
            actual = flash_attention4_or_sdpa(
                query,
                key,
                value,
                policy="disabled",
            )

        self.assertEqual(tuple(actual.shape), tuple(query.shape))
        self.assertEqual(calls[0][1][1], query.shape[1])
        self.assertEqual(calls[0][2][1], query.shape[1])
        self.assertNotIn("enable_gqa", calls[0][3])

    def test_sdpa_fallback_only_recovers_known_native_gqa_failures(self):
        query = torch.randn(1, 4, 3, 8)
        key = torch.randn(1, 2, 3, 8)
        value = torch.randn_like(key)
        calls = []

        def unavailable_gqa(query_states, key_states, value_states, **kwargs):
            calls.append((key_states.shape, dict(kwargs)))
            if kwargs.get("enable_gqa"):
                raise RuntimeError("No available kernel for grouped query attention")
            return torch.zeros_like(query_states)

        with (
                mock.patch.object(backend, "_SDPA_SUPPORTS_GQA", True),
                mock.patch.object(
                    backend.functional,
                    "scaled_dot_product_attention",
                    side_effect=unavailable_gqa,
                ),
        ):
            flash_attention4_or_sdpa(
                query,
                key,
                value,
                policy="disabled",
            )

        self.assertTrue(calls[0][1]["enable_gqa"])
        self.assertEqual(calls[1][0][1], query.shape[1])
        self.assertNotIn("enable_gqa", calls[1][1])

        with (
                mock.patch.object(backend, "_SDPA_SUPPORTS_GQA", True),
                mock.patch.object(
                    backend.functional,
                    "scaled_dot_product_attention",
                    side_effect=RuntimeError("CUDA out of memory"),
                ),
                self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            flash_attention4_or_sdpa(
                query,
                key,
                value,
                policy="disabled",
            )

    def test_fa4_path_transposes_layout_and_forwards_gqa_controls(self):
        query = torch.randn(2, 4, 3, 8, dtype=torch.float16)
        key = torch.randn(2, 2, 5, 8, dtype=torch.float16)
        value = torch.randn(2, 2, 5, 8, dtype=torch.float16)
        calls = []

        def fake_flash_attention(query_bshd, key_bshd, value_bshd, **kwargs):
            calls.append((query_bshd, key_bshd, value_bshd, kwargs))
            output = query_bshd + 2
            return output, None

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "_load_flash_attention4", return_value=fake_flash_attention),
        ):
            actual = flash_attention4_or_sdpa(
                query,
                key,
                value,
                is_causal=False,
                scale=0.125,
                policy="required",
                deterministic=True,
            )

        query_bshd, key_bshd, value_bshd, keywords = calls[0]
        self.assertEqual(tuple(query_bshd.shape), (2, 3, 4, 8))
        self.assertEqual(tuple(key_bshd.shape), (2, 5, 2, 8))
        self.assertEqual(tuple(value_bshd.shape), (2, 5, 2, 8))
        self.assertTrue(query_bshd.is_contiguous())
        self.assertTrue(key_bshd.is_contiguous())
        self.assertEqual(
            keywords, {
                "softmax_scale": 0.125,
                "causal": False,
                "pack_gqa": True,
                "deterministic": True,
                "return_lse": False,
            })
        self.assertEqual(tuple(actual.shape), tuple(query.shape))
        torch.testing.assert_close(actual, query + 2)

    def test_fa4_path_accepts_cached_non_square_causal_attention(self):
        query = torch.randn(1, 2, 1, 8, dtype=torch.float16)
        key = torch.randn(1, 2, 5, 8, dtype=torch.float16)
        value = torch.randn_like(key)
        calls = []

        def fake_flash_attention(query_bshd, key_bshd, value_bshd, **kwargs):
            calls.append(kwargs)
            return torch.zeros_like(query_bshd), None

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "_load_flash_attention4", return_value=fake_flash_attention),
        ):
            output = flash_attention4_or_sdpa(
                query,
                key,
                value,
                is_causal=True,
                policy="required",
            )

        self.assertEqual(tuple(output.shape), tuple(query.shape))
        self.assertTrue(calls[0]["causal"])

    def test_incompatible_installed_api_fails_required_and_falls_back_auto(self):

        def stale_flash_attention(q, k, v):
            return q

        module = SimpleNamespace(flash_attn_func=stale_flash_attention)
        query = torch.randn(1, 2, 3, 8, dtype=torch.float16)

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "import_module", return_value=module),
                self.assertRaisesRegex(FlashAttention4UnavailableError, "missing parameters"),
        ):
            flash_attention4_or_sdpa(
                query,
                query,
                query,
                policy="required",
            )

        backend._load_flash_attention4.cache_clear()
        sentinel = torch.empty_like(query)
        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "import_module", return_value=module),
                mock.patch.object(backend, "_pytorch_sdpa", return_value=sentinel),
        ):
            actual = flash_attention4_or_sdpa(query, query, query)
        self.assertIs(actual, sentinel)

    def test_execution_compatibility_failure_falls_back_or_fails_when_required(self):
        query = torch.randn(1, 2, 3, 8, dtype=torch.float16)
        sentinel = torch.empty_like(query)

        def unsupported_kernel(*args, **kwargs):
            raise NotImplementedError("shape is not compiled")

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "_load_flash_attention4", return_value=unsupported_kernel),
                mock.patch.object(backend, "_pytorch_sdpa", return_value=sentinel) as sdpa,
        ):
            actual = flash_attention4_or_sdpa(query, query, query)

        self.assertIs(actual, sentinel)
        sdpa.assert_called_once()

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "_load_flash_attention4", return_value=unsupported_kernel),
                self.assertRaisesRegex(FlashAttention4ExecutionError, "shape is not compiled"),
        ):
            flash_attention4_or_sdpa(
                query,
                query,
                query,
                policy="required",
            )

    def test_auto_does_not_hide_fatal_cuda_failures(self):
        query = torch.randn(1, 2, 3, 8, dtype=torch.float16)

        def out_of_memory(*args, **kwargs):
            raise RuntimeError("CUDA out of memory while launching FA4")

        with (
                mock.patch.object(backend, "flash_attention4_capability", return_value=_supported()),
                mock.patch.object(backend, "_load_flash_attention4", return_value=out_of_memory),
                mock.patch.object(backend, "_pytorch_sdpa") as sdpa,
                self.assertRaisesRegex(RuntimeError, "CUDA out of memory"),
        ):
            flash_attention4_or_sdpa(query, query, query)

        sdpa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
