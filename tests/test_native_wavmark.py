import ast
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn

from voicehub.components.audio.watermarking import wavmark
from voicehub.components.audio.watermarking.wavmark.utils import file_reader, metric_util

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _DeterministicWatermarkModel(nn.Module):

    def __init__(self, message):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "decoded_message",
            torch.as_tensor(message, dtype=torch.float32),
        )

    def encode(self, signal, message):
        del message
        return signal + 0.003 + self.anchor * 0.0

    def decode(self, signal):
        return self.decoded_message.unsqueeze(0).expand(signal.shape[0], -1)


class NativeWavMarkTests(unittest.TestCase):

    def test_runtime_uses_only_torch_stdlib_and_voicehub(self):
        root = (PROJECT_ROOT / "voicehub" / "components" / "audio" / "watermarking" / "wavmark")
        forbidden = {
            "librosa",
            "numpy",
            "resampy",
            "safetensors",
            "soundfile",
            "torchaudio",
            "tqdm",
            "transformers",
        }
        violations = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module):
                    names = [node.module]
                else:
                    names = []
                for name in names:
                    if name.partition(".")[0] in forbidden:
                        violations.append((path.name, name))
        self.assertEqual(violations, [])

    def test_encode_and_decode_return_native_tensors(self):
        pattern = wavmark.wm_add_util.fix_pattern[:16]
        payload = [index % 2 for index in range(16)]
        model = _DeterministicWatermarkModel(pattern + payload)
        waveform = torch.full((17_600, ), 0.1)

        watermarked, encode_info = wavmark.encode_watermark(
            model,
            waveform,
            payload,
        )
        recovered, decode_info = wavmark.decode_watermark(
            model,
            watermarked,
        )

        self.assertIsInstance(watermarked, torch.Tensor)
        self.assertEqual(watermarked.shape, waveform.shape)
        self.assertEqual(encode_info["encoded_sections"], 1)
        self.assertGreater(encode_info["snr"], 20.0)
        self.assertLess(encode_info["snr"], 38.0)
        torch.testing.assert_close(
            recovered,
            torch.tensor(payload, dtype=torch.int32),
        )
        self.assertGreaterEqual(len(decode_info["results"]), 1)

    def test_short_audio_and_invalid_payload_fail_before_model_execution(self):
        model = _DeterministicWatermarkModel([0] * 32)
        with self.assertRaisesRegex(ValueError, "payload must contain 16"):
            wavmark.encode_watermark(
                model,
                torch.zeros(17_600),
                [0, 1],
            )
        with self.assertRaisesRegex(ValueError, "at least 17600 samples"):
            wavmark.encode_watermark(
                model,
                torch.zeros(16_000),
                [0] * 16,
            )

    def test_snr_and_resampling_remain_tensor_native(self):
        waveform = torch.linspace(-0.5, 0.5, 320)
        self.assertEqual(
            metric_util.signal_noise_ratio(waveform, waveform),
            float("inf"),
        )
        resampled = metric_util.resample_to16k(waveform, 32_000)
        self.assertIsInstance(resampled, torch.Tensor)
        self.assertEqual(resampled.numel(), 160)

    def test_pcm_wave_reader_downmixes_and_resamples_without_audio_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "stereo.wav"
            frames = []
            for index in range(800):
                left = round(8_000 * torch.sin(torch.tensor(index / 20)).item())
                frames.append(struct.pack("<hh", left, -left))
            with wave.open(str(path), "wb") as stream:
                stream.setnchannels(2)
                stream.setsampwidth(2)
                stream.setframerate(8_000)
                stream.writeframes(b"".join(frames))

            waveform, rate, duration = file_reader.read_as_single_channel_16k(path, )

        self.assertEqual(rate, 16_000)
        self.assertEqual(waveform.numel(), 1_600)
        self.assertAlmostEqual(duration, 0.1)
        self.assertEqual(float(waveform.abs().max()), 0.0)

    def test_checkpoint_loading_uses_restricted_pytorch_reader(self):
        fake_model = Mock()
        fake_model.load_state_dict = Mock()
        fake_model.eval.return_value = fake_model
        with tempfile.NamedTemporaryFile(suffix=".pkl") as stream:
            with (
                    patch.object(wavmark.my_model, "Model", return_value=fake_model),
                    patch.object(
                        torch,
                        "load",
                        return_value={"weight": torch.ones(1)},
                    ) as load,
            ):
                loaded = wavmark.load_model(stream.name)

        self.assertIs(loaded, fake_model)
        load.assert_called_once_with(
            Path(stream.name),
            map_location="cpu",
            weights_only=True,
        )
        fake_model.load_state_dict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
