from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from voicehub.checkpointing import (
    ArtifactFile,
    CheckpointFormatError,
    CheckpointIntegrityError,
    SafeTensorIndex,
    SafeTensorReader,
    ShardedSafeTensorReader,
    VoiceHubManifest,
    build_manifest_files,
    load_numpy_tensor,
    load_sharded_safetensors,
    save_safetensors,
)


def _write_npy_v1(path: Path, header: dict, payload: bytes) -> None:
    encoded = repr(header).encode("latin1")
    padding = (-(10 + len(encoded) + 1)) % 16
    encoded += b" " * padding + b"\n"
    path.write_bytes(b"\x93NUMPY" + bytes((1, 0)) + struct.pack("<H", len(encoded)) + encoded + payload)


@unittest.skipUnless(torch is not None, "Native checkpoints use PyTorch tensors")
class SafeTensorTests(unittest.TestCase):

    def test_writer_streams_large_tensor_without_materializing_storage_bytes(self):

        class StorageBytesForbiddenTensor(torch.Tensor):

            @staticmethod
            def __new__(cls, value):
                return torch.Tensor._make_subclass(cls, value, False)

            def untyped_storage(self):
                raise AssertionError("Safetensors writer must not iterate storage into bytes.")

        # Four MiB is large enough to exercise the mapped copy path without
        # adding meaningful memory or runtime cost to CI.
        value = StorageBytesForbiddenTensor(torch.arange(1024 * 1024, dtype=torch.float32), )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "streamed.safetensors"
            save_safetensors({"weight": value}, path)

            with SafeTensorReader(path) as reader:
                restored = reader.get_tensor("weight")
            torch.testing.assert_close(
                restored,
                value.as_subclass(torch.Tensor),
                rtol=0,
                atol=0,
            )

    def test_round_trip_is_deterministic_and_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.safetensors"
            second = Path(directory) / "second.safetensors"
            tensors = {
                "decoder.bias": torch.tensor([1, -2, 3], dtype=torch.int16),
                "encoder.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                "empty": torch.empty((2, 0), dtype=torch.float16),
                "scalar": torch.tensor(4.5, dtype=torch.float64),
            }
            save_safetensors(tensors, first, metadata={"format": "native"})
            save_safetensors(tensors, second, metadata={"format": "native"})

            self.assertEqual(first.read_bytes(), second.read_bytes())
            with SafeTensorReader(first) as reader:
                self.assertEqual(reader.metadata, {"format": "native"})
                self.assertEqual(
                    reader.keys(),
                    ("decoder.bias", "empty", "encoder.weight", "scalar"),
                )
                for name, expected in tensors.items():
                    torch.testing.assert_close(
                        reader.get_tensor(name),
                        expected,
                        rtol=0,
                        atol=0,
                    )

    def test_reader_rejects_overlapping_ranges_before_materializing_data(self):
        header = {
            "first": {
                "dtype": "U8",
                "shape": [2],
                "data_offsets": [0, 2],
            },
            "second": {
                "dtype": "U8",
                "shape": [2],
                "data_offsets": [1, 3],
            },
        }
        encoded = json.dumps(header).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0\0\0")
            with self.assertRaisesRegex(CheckpointFormatError, "overlaps"):
                SafeTensorReader(path)

    def test_reader_rejects_declared_shape_size_mismatch(self):
        header = {
            "weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, 4],
            },
        }
        encoded = json.dumps(header).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-size.safetensors"
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
            with self.assertRaisesRegex(CheckpointFormatError, "require 8"):
                SafeTensorReader(path)

    def test_sharded_index_loads_selected_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_safetensors({"a": torch.tensor([1.0])}, root / "part-1.safetensors")
            save_safetensors({"b": torch.tensor([2.0])}, root / "part-2.safetensors")
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({
                    "metadata": {
                        "total_size": 8
                    },
                    "weight_map": {
                        "a": "part-1.safetensors",
                        "b": "part-2.safetensors",
                    },
                }),
                encoding="utf-8",
            )

            index = SafeTensorIndex.from_file(index_path)
            self.assertEqual(index.keys(), ("a", "b"))
            loaded = load_sharded_safetensors(index_path, names=("b", ))
            self.assertEqual(tuple(loaded), ("b", ))
            torch.testing.assert_close(loaded["b"], torch.tensor([2.0]))

            with ShardedSafeTensorReader(index_path) as reader:
                self.assertEqual(reader.keys(), ("a", "b"))
                self.assertEqual(len(reader._readers), 0)
                torch.testing.assert_close(
                    reader.get_tensor("b"),
                    torch.tensor([2.0]),
                )
                self.assertEqual(
                    tuple(path.name for path in reader._readers),
                    ("part-2.safetensors", ),
                )
            with self.assertRaisesRegex(RuntimeError, "closed"):
                reader.get_tensor("a")

    def test_sharded_index_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps({"weight_map": {
                    "a": "../outside.safetensors"
                }}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CheckpointFormatError, "Unsafe"):
                SafeTensorIndex.from_file(index_path)


@unittest.skipUnless(torch is not None, "Native checkpoints use PyTorch tensors")
class NumpyTensorTests(unittest.TestCase):

    def test_reader_loads_c_and_fortran_order_without_numpy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c_path = root / "c.npy"
            fortran_path = root / "fortran.npy"
            header = {
                "descr": "<f4",
                "fortran_order": False,
                "shape": (2, 3),
            }
            _write_npy_v1(
                c_path,
                header,
                struct.pack("<6f", 1, 2, 3, 4, 5, 6),
            )
            _write_npy_v1(
                fortran_path,
                {
                    **header, "fortran_order": True
                },
                struct.pack("<6f", 1, 4, 2, 5, 3, 6),
            )
            expected = torch.tensor(
                [[1, 2, 3], [4, 5, 6]],
                dtype=torch.float32,
            )

            torch.testing.assert_close(load_numpy_tensor(c_path), expected)
            torch.testing.assert_close(
                load_numpy_tensor(fortran_path),
                expected,
            )

    def test_reader_rejects_object_arrays_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            object_path = root / "object.npy"
            trailing_path = root / "trailing.npy"
            _write_npy_v1(
                object_path,
                {
                    "descr": "|O8",
                    "fortran_order": False,
                    "shape": (1, ),
                },
                b"\0" * 8,
            )
            _write_npy_v1(
                trailing_path,
                {
                    "descr": "|u1",
                    "fortran_order": False,
                    "shape": (1, ),
                },
                b"\1\2",
            )

            with self.assertRaisesRegex(
                    CheckpointFormatError,
                    "Unsupported NPY dtype",
            ):
                load_numpy_tensor(object_path)
            with self.assertRaisesRegex(
                    CheckpointFormatError,
                    "trailing bytes",
            ):
                load_numpy_tensor(trailing_path)


class ManifestTests(unittest.TestCase):

    def _manifest(self, files=(), processor_assets=()):
        return VoiceHubManifest(
            architecture="whisper",
            architecture_version="1.0.0",
            checkpoint_format="safetensors",
            adapter_version="openai-v1",
            source="openai/whisper-small",
            source_revision="0123456789abcdef",
            source_license="MIT",
            weight_license="MIT",
            processor_assets=processor_assets,
            training_recipe="whisper-seq2seq-v1",
            files=files,
            metadata={"task": "asr"},
        )

    def test_manifest_round_trip_and_integrity_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors").write_bytes(b"weights")
            files = build_manifest_files(
                root,
                ("tokenizer.json", "model.safetensors"),
            )
            manifest = self._manifest(
                files=files,
                processor_assets=("tokenizer.json", ),
            )
            path = manifest.save(root)
            restored = VoiceHubManifest.load(path)

            self.assertEqual(restored, manifest)
            restored.verify(root)

    def test_manifest_detects_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model.safetensors"
            artifact.write_bytes(b"first")
            record = ArtifactFile.from_path(root, artifact.name)
            manifest = self._manifest(files=(record, ))
            artifact.write_bytes(b"other")

            with self.assertRaises(CheckpointIntegrityError):
                manifest.verify(root)

    def test_manifest_rejects_unsafe_paths(self):
        with self.assertRaisesRegex(ValueError, "relative and safe"):
            ArtifactFile(path="../secret", size=0, sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
