from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from voicehub.processing import LogMelSpectrogram, ModelBatch, PadOrTrimAudio, ProcessorGraph


class ProcessingImportTests(unittest.TestCase):

    def test_package_discovery_keeps_dsp_modules_lazy(self):
        code = (
            "import sys; "
            "import voicehub.processing as processing; "
            "assert 'torch' not in sys.modules; "
            "assert 'KaldiFbank' in processing.__all__; "
            "assert 'load_native_audio' in processing.__all__")
        subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


@unittest.skipUnless(torch is not None, "Native processing uses PyTorch")
class ProcessorGraphTests(unittest.TestCase):

    def test_graph_round_trip_runs_identical_audio_operations(self):
        graph = ProcessorGraph(
            inputs=("waveform", ),
            outputs=("input_features", "waveform_length"),
            operations=(
                PadOrTrimAudio(length=1600),
                LogMelSpectrogram(
                    input_key="padded_waveform",
                    n_mels=16,
                    whisper_scaling=True,
                ),
            ),
            metadata={"architecture": "test"},
        )
        restored = ProcessorGraph.from_dict(graph.to_dict())
        waveform = torch.linspace(-0.5, 0.5, 800)

        expected = graph.run({"waveform": waveform})
        actual = restored.run({"waveform": waveform})

        torch.testing.assert_close(
            actual["input_features"],
            expected["input_features"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(actual["waveform_length"].item(), 800)

    def test_graph_rejects_implicit_overwrites(self):
        with self.assertRaisesRegex(ValueError, "overwrites"):
            ProcessorGraph(
                inputs=("waveform", ),
                outputs=("waveform", ),
                operations=(PadOrTrimAudio(
                    length=100,
                    output_key="waveform",
                ), ),
            )

    def test_graph_requires_an_exact_input_contract(self):
        graph = ProcessorGraph(
            inputs=("waveform", ),
            outputs=("padded_waveform", ),
            operations=(PadOrTrimAudio(length=100), ),
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            graph.run({"waveform": torch.zeros(100), "secret": "ignored"})

    def test_model_batch_recursively_moves_tensor_values(self):
        batch = ModelBatch(
            data={
                "input": torch.ones(1, dtype=torch.float32),
                "nested": (torch.ones(1, dtype=torch.float32), ),
            },
            batch_size=1,
        )

        converted = batch.to(dtype=torch.float64)

        self.assertEqual(converted["input"].dtype, torch.float64)
        self.assertEqual(converted["nested"][0].dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
