from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from voicehub.components.audio.codecs.encodec import EncodecModel
from voicehub.models.xtts.source.TTS.tts.layers.bark import inference_funcs
from voicehub.models.xtts.source.TTS.tts.models import bark
from voicehub.processing.waveform import save_pcm_wave


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XTTS_SOURCE = PROJECT_ROOT / "voicehub" / "models" / "xtts" / "source" / "TTS"
BARK_RUNTIME = (
    XTTS_SOURCE / "tts" / "models" / "bark.py",
    XTTS_SOURCE / "tts" / "layers" / "bark",
)


class _LoaderCodec:
    sample_rate = 24_000
    channels = 1
    quantizer = SimpleNamespace(bins=1_024)

    def __init__(self):
        self.set_target_bandwidth = Mock()


class _RuntimeCodec(nn.Module):
    sample_rate = 24_000
    channels = 1

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.decoded_frames = None

    def decode(self, frames):
        self.decoded_frames = frames
        codes, _ = frames[0]
        return codes[:, :1].to(dtype=self.weight.dtype)


def _config(**overrides):
    values = {
        "sample_rate": 24_000,
        "ENCODEC_CHECKPOINT": None,
        "ENCODEC_CACHE_DIR": None,
        "ENCODEC_LOCAL_FILES_ONLY": False,
        "TRUST_OFFICIAL_ENCODEC_PICKLE": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _absolute_imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
        ):
            yield node.module


class NativeXTTSBarkCodecTests(unittest.TestCase):

    def test_entire_xtts_source_has_no_external_encodec_import(self):
        violations = []
        for path in XTTS_SOURCE.rglob("*.py"):
            for imported in _absolute_imports(path):
                if imported.partition(".")[0] == "encodec":
                    violations.append((path.relative_to(PROJECT_ROOT), imported))
        self.assertEqual(violations, [])
        self.assertIs(bark.EncodecModel, EncodecModel)

    def test_bark_audio_execution_has_no_torchaudio_import(self):
        paths = [BARK_RUNTIME[0], *BARK_RUNTIME[1].rglob("*.py")]
        violations = []
        for path in paths:
            for imported in _absolute_imports(path):
                if imported.partition(".")[0] == "torchaudio":
                    violations.append((path.relative_to(PROJECT_ROOT), imported))
        self.assertEqual(violations, [])

    def test_missing_safe_checkpoint_fails_before_network_download(self):
        with patch.object(
            bark,
            "load_encodec_model",
            side_effect=FileNotFoundError("not cached"),
        ) as loader:
            with self.assertRaisesRegex(
                PermissionError,
                "Provide a converted.*safetensors",
            ):
                bark.load_bark_encodec(_config())

        loader.assert_called_once_with(
            "encodec_24khz",
            checkpoint=None,
            cache_dir=None,
            local_files_only=True,
            trust_official_pickle=False,
        )

    def test_explicit_trust_is_forwarded_to_strict_native_loader(self):
        codec = _LoaderCodec()
        with patch.object(
            bark,
            "load_encodec_model",
            return_value=codec,
        ) as loader:
            result = bark.load_bark_encodec(
                _config(TRUST_OFFICIAL_ENCODEC_PICKLE=True),
            )

        self.assertIs(result, codec)
        loader.assert_called_once_with(
            "encodec_24khz",
            checkpoint=None,
            cache_dir=None,
            local_files_only=False,
            trust_official_pickle=True,
        )
        codec.set_target_bandwidth.assert_called_once_with(6.0)

    def test_native_safetensors_checkpoint_needs_no_pickle_trust(self):
        codec = _LoaderCodec()
        checkpoint = Path("/model/encodec_24khz.safetensors")
        with patch.object(
            bark,
            "load_encodec_model",
            return_value=codec,
        ) as loader:
            bark.load_bark_encodec(
                _config(ENCODEC_CHECKPOINT=str(checkpoint)),
            )

        loader.assert_called_once_with(
            "encodec_24khz",
            checkpoint=str(checkpoint),
            cache_dir=None,
            local_files_only=False,
            trust_official_pickle=False,
        )

    def test_pickle_trust_rejects_truthy_non_boolean_values(self):
        with patch.object(bark, "load_encodec_model") as loader:
            with self.assertRaisesRegex(TypeError, "must be a boolean"):
                bark.load_bark_encodec(
                    _config(TRUST_OFFICIAL_ENCODEC_PICKLE="false"),
                )
        loader.assert_not_called()

    def test_pcm_wave_is_downmixed_and_resampled_for_native_codec(self):
        codec = _RuntimeCodec()
        model = SimpleNamespace(
            encodec=codec,
            device=torch.device("cpu"),
        )
        stereo = torch.stack(
            (
                torch.linspace(-0.75, 0.75, 120),
                torch.linspace(0.75, -0.75, 120),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            source = save_pcm_wave(
                Path(directory) / "reference.wav",
                stereo,
                12_000,
            )
            prepared = inference_funcs.prepare_codec_audio(source, model)

        self.assertEqual(prepared.shape, (1, 1, 240))
        self.assertEqual(prepared.dtype, codec.weight.dtype)
        self.assertTrue(torch.isfinite(prepared).all())
        self.assertLess(prepared.abs().max().item(), 1e-4)

    def test_codec_decode_uses_public_native_frame_contract(self):
        codec = _RuntimeCodec()
        model = SimpleNamespace(
            encodec=codec,
            device=torch.device("cpu"),
        )
        fine_tokens = np.arange(24, dtype=np.int64).reshape(8, 3) % 1_024

        audio = inference_funcs.codec_decode(fine_tokens, model)

        codes, scale = codec.decoded_frames[0]
        self.assertEqual(codes.shape, (1, 8, 3))
        self.assertEqual(codes.dtype, torch.long)
        self.assertIsNone(scale)
        np.testing.assert_array_equal(audio, fine_tokens[0])

    def test_codec_decode_rejects_ambiguous_token_shapes(self):
        model = SimpleNamespace(
            encodec=_RuntimeCodec(),
            device=torch.device("cpu"),
        )
        with self.assertRaisesRegex(ValueError, "codebooks, frames"):
            inference_funcs.codec_decode(np.zeros(8, dtype=np.int64), model)


if __name__ == "__main__":
    unittest.main()
