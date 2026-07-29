from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from voicehub.models.parlertts.source.parler_tts.dac_wrapper.modeling_dac import (
    DACModel,
    EncodecDecoderOutput,
    EncodecEncoderOutput,
)
from voicehub.models.parlertts.source.parler_tts.dac_wrapper.modeling_outputs import DACDecoderOutput, DACEncoderOutput


class _FakeQuantizer:

    @staticmethod
    def from_codes(codes):
        return (codes.float(), )


class _FakeDAC:
    quantizer = _FakeQuantizer()

    @staticmethod
    def preprocess(audio, sample_rate):
        if sample_rate != 44_100:
            raise ValueError("unexpected test sample rate")
        return audio

    @staticmethod
    def encode(frame, n_quantizers=None):
        codebooks = 2 if n_quantizers is None else n_quantizers
        codes = torch.arange(
            frame.shape[0] * codebooks * 2,
            device=frame.device,
        ).reshape(frame.shape[0], codebooks, 2)
        return frame, codes, frame, frame.mean(), frame.mean()

    @staticmethod
    def decode(latents):
        return latents + 0.25


class ParlerDACOutputTests(unittest.TestCase):

    @staticmethod
    def _wrapper() -> DACModel:
        wrapper = DACModel.__new__(DACModel)
        nn.Module.__init__(wrapper)
        wrapper.config = SimpleNamespace(return_dict=True)
        wrapper.model = _FakeDAC()
        return wrapper

    def test_outputs_preserve_mapping_and_positional_access(self):
        codes = torch.tensor([[[[1, 2], [3, 4]]]])
        scales = [None]
        encoded = DACEncoderOutput(codes, scales)
        decoded = DACDecoderOutput(torch.tensor([0.25]))

        self.assertEqual(list(encoded), ["audio_codes", "audio_scales"])
        self.assertIs(encoded["audio_codes"], codes)
        self.assertIs(encoded[0], codes)
        self.assertEqual(encoded.get("audio_scales"), scales)
        self.assertEqual(encoded.to_tuple(), (codes, scales))
        self.assertNotIn("last_frame_pad_length", encoded)
        self.assertEqual(list(decoded), ["audio_values"])
        torch.testing.assert_close(decoded[0], torch.tensor([0.25]))
        self.assertIs(EncodecEncoderOutput, DACEncoderOutput)
        self.assertIs(EncodecDecoderOutput, DACDecoderOutput)

    def test_wrapper_encode_and_decode_use_local_output_contracts(self):
        wrapper = self._wrapper()
        waveform = torch.randn(2, 1, 8)

        encoded = wrapper.encode(
            waveform,
            sample_rate=44_100,
            n_quantizers=2,
        )
        decoded = wrapper.decode(encoded.audio_codes, encoded.audio_scales)

        self.assertIsInstance(encoded, DACEncoderOutput)
        self.assertEqual(tuple(encoded.audio_codes.shape), (1, 2, 2, 2))
        self.assertEqual(encoded.audio_scales, [None])
        self.assertIsInstance(decoded, DACDecoderOutput)
        torch.testing.assert_close(
            decoded.audio_values,
            encoded.audio_codes.squeeze(0).float() + 0.25,
        )

    def test_wrapper_has_no_model_specific_transformers_import(self):
        wrapper_directory = (
            Path(__file__).resolve().parents[1] / "voicehub" / "models" / "parlertts" / "source" /
            "parler_tts" / "dac_wrapper")
        imports = []
        for source_path in wrapper_directory.glob("*.py"):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=str(source_path),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                elif isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)

        self.assertNotIn(
            "transformers.models.encodec.modeling_encodec",
            imports,
        )
        self.assertFalse(
            any(name.startswith("transformers.models.") for name in imports),
            imports,
        )


if __name__ == "__main__":
    unittest.main()
