from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_smoke_module():
    path = Path(__file__).parents[1] / "scripts" / "smoke_finetune_vits.py"
    spec = importlib.util.spec_from_file_location(
        "voicehub_test_vits_finetune_smoke",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the VITS fine-tune smoke module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_smoke_module()


class VITSFineTuneSmokeScriptTests(unittest.TestCase):

    def test_checkpoint_identity_uses_config_revision_and_skips_index_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors.index.json"
            checkpoint.write_bytes(b'{"weight_map": {}}')
            model = SimpleNamespace(
                config=SimpleNamespace(revision="requested-revision"),
                artifacts=SimpleNamespace(
                    source="resolved/repository",
                    revision="b" * 40,
                    checkpoint=checkpoint,
                ),
            )

            result = smoke.checkpoint_identity(
                model,
                requested_source="requested/repository",
            )

        self.assertEqual(result["requested_revision"], "requested-revision")
        self.assertTrue(result["requested_revision_was_explicit"])
        self.assertEqual(result["resolved_revision"], "b" * 40)
        self.assertEqual(result["local_checkpoint_path"], str(checkpoint.resolve()))
        self.assertIsNone(result["local_weight_sha256"])
        self.assertEqual(
            result["weight_digest_status"],
            "checkpoint-index-only",
        )

    def test_checkpoint_identity_hashes_explicit_weight_file(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            checkpoint.write_bytes(b"trained-weights")
            model = SimpleNamespace(
                config=SimpleNamespace(revision=None),
                artifacts=SimpleNamespace(
                    source=str(Path(directory)),
                    revision=None,
                    checkpoint=checkpoint,
                ),
            )

            result = smoke.checkpoint_identity(
                model,
                requested_source=directory,
            )

        self.assertEqual(
            result["local_weight_sha256"],
            hashlib.sha256(b"trained-weights").hexdigest(),
        )
        self.assertFalse(result["requested_revision_was_explicit"])
        self.assertEqual(result["weight_digest_status"], "sha256")

    def test_output_directory_must_not_overwrite_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "new-export"
            self.assertEqual(
                smoke.prepare_output_directory(destination),
                destination.resolve(),
            )
            (destination / "model.safetensors").write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                smoke.prepare_output_directory(destination)

    def test_linear_spectrogram_has_vits_checkpoint_frequency_bins(self):
        waveform = torch.linspace(-1.0, 1.0, 4_096)

        spectrogram = smoke.build_linear_spectrogram(waveform, torch)

        self.assertEqual(spectrogram.ndim, 2)
        self.assertEqual(spectrogram.shape[0], 513)
        self.assertTrue(bool(torch.isfinite(spectrogram).all().item()))
        with self.assertRaisesRegex(ValueError, "at least 1024"):
            smoke.build_linear_spectrogram(torch.zeros(100), torch)

    def test_training_reference_preserves_complete_aligned_utterance(self):

        class Output:
            sample_rate = 4
            audio = torch.arange(12, dtype=torch.float32)

        reference, sample_rate = smoke.training_reference(
            Output(),
            torch,
            minimum_seconds=2.5,
        )

        self.assertEqual(sample_rate, 4)
        self.assertEqual(reference.tolist(), list(range(12)))
        with self.assertRaisesRegex(RuntimeError, "required at least"):
            smoke.training_reference(
                Output(),
                torch,
                minimum_seconds=3.5,
            )

    def test_state_fingerprint_is_exact_and_order_independent(self):
        first = {
            "b": torch.tensor([3.0]),
            "a": torch.tensor([1.0, 2.0]),
        }
        reordered = {
            "a": first["a"].clone(),
            "b": first["b"].clone(),
        }
        changed = {
            "a": torch.tensor([1.0, 2.5]),
            "b": first["b"].clone(),
        }
        bfloat = {
            "a": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        }

        self.assertEqual(
            smoke.state_fingerprint(first),
            smoke.state_fingerprint(reordered),
        )
        self.assertNotEqual(
            smoke.state_fingerprint(first),
            smoke.state_fingerprint(changed),
        )
        self.assertEqual(
            smoke.state_fingerprint(bfloat),
            smoke.state_fingerprint({"a": bfloat["a"].clone()}),
        )

    def test_gradient_clip_norm_must_be_positive(self):
        with self.assertRaisesRegex(
                smoke.argparse.ArgumentTypeError,
                "greater than zero",
        ):
            smoke._positive_float("0")


if __name__ == "__main__":
    unittest.main()
