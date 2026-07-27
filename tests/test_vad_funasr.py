import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.models.vad_funasr import FunASRVADConfig, FunASRVADForVoiceActivityDetection


class _FakeFunASRRuntime:

    def __init__(self, result=None):
        self.result = ([{
            "key": "audio",
            "value": [[100, 400], [450, 900], [1200, 1600]],
        }] if result is None else result)
        self.calls = []

    def generate(self, input, **kwargs):
        self.calls.append({
            "input": input,
            **kwargs,
        })
        return self.result


class FunASRVADTests(unittest.TestCase):

    def test_auto_device_uses_cpu_instead_of_mps(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(
                FunASRVADForVoiceActivityDetection._resolve_device("auto"),
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "CPU, CUDA, XPU, NPU"):
            FunASRVADForVoiceActivityDetection._resolve_device("mps")

    def test_import_is_lazy_and_does_not_import_funasr(self):
        code = """
import json
import sys
from voicehub.models.vad_funasr import FunASRVADForVoiceActivityDetection
print(json.dumps({"funasr": "funasr" in sys.modules}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )

        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {"funasr": False},
        )

    def test_config_is_serializable_and_protects_managed_options(self):
        config = FunASRVADConfig(
            name_or_path="FunAudioLLM/FSMN-VAD",
            hub="hf",
            revision="main",
            ncpu=4,
            generate_kwargs={
                "chunk_size": 30_000,
            },
        )

        self.assertEqual(config.model_type, "vad_funasr")
        self.assertEqual(config.to_dict()["hub"], "hf")
        self.assertEqual(config.to_dict()["revision"], "main")
        with self.assertRaisesRegex(ValueError, "VoiceHub-managed"):
            FunASRVADConfig(model_kwargs={"model": "other"})
        with self.assertRaisesRegex(ValueError, "VoiceHub-managed"):
            FunASRVADConfig(generate_kwargs={"is_streaming_input": True})
        with self.assertRaisesRegex(ValueError, "credentials"):
            FunASRVADConfig(model_kwargs={"download": {"token": "secret"}})
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            FunASRVADConfig(sample_rate=8_000)

    def test_loader_supports_hub_and_local_model_sources(self):
        constructor_calls = []
        runtimes = []

        def auto_model(**kwargs):
            constructor_calls.append(kwargs)
            runtime = _FakeFunASRRuntime(result=[{"value": []}])
            runtimes.append(runtime)
            return runtime

        funasr = ModuleType("funasr")
        funasr.AutoModel = auto_model
        with tempfile.TemporaryDirectory() as directory:
            model_sources = (
                ("FunAudioLLM/FSMN-VAD", "hf"),
                (Path(directory), "ms"),
            )
            with patch.dict(sys.modules, {"funasr": funasr}):
                models = []
                for source, hub in model_sources:
                    model = FunASRVADForVoiceActivityDetection(
                        FunASRVADConfig(
                            name_or_path=source,
                            hub=hub,
                            revision="v1",
                            ncpu=3,
                        ),
                        device="cpu",
                    )
                    model._load_pretrained_model()
                    models.append(model)

        self.assertEqual(
            [call["model"] for call in constructor_calls],
            [
                "FunAudioLLM/FSMN-VAD",
                str(Path(model_sources[1][0]).resolve()),
            ],
        )
        self.assertEqual(
            [call["hub"] for call in constructor_calls],
            ["hf", "ms"],
        )
        self.assertTrue(all(call["model_revision"] == "v1" for call in constructor_calls))
        self.assertTrue(all(call["ncpu"] == 3 for call in constructor_calls))
        self.assertEqual(
            [model.model for model in models],
            runtimes,
        )

    def test_public_detect_lazily_loads_the_runtime(self):
        runtime = _FakeFunASRRuntime(result=[{"value": [[100, 600]]}])
        constructor_calls = []

        def auto_model(**kwargs):
            constructor_calls.append(kwargs)
            return runtime

        funasr = ModuleType("funasr")
        funasr.AutoModel = auto_model
        model = FunASRVADForVoiceActivityDetection(
            FunASRVADConfig(inference_config={
                "min_speech_duration_ms": 0,
                "speech_pad_ms": 0,
            }),
            device="cpu",
        )

        with patch.dict(sys.modules, {"funasr": funasr}):
            output = model.detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
            )

        self.assertEqual(len(constructor_calls), 1)
        self.assertIs(model.model, runtime)
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.1, 0.6)],
        )

    def test_array_inference_maps_runtime_options_and_normalizes_seconds(self):
        runtime = _FakeFunASRRuntime()
        model = FunASRVADForVoiceActivityDetection(
            FunASRVADConfig(name_or_path="fsmn-vad"),
            device="cpu",
        )
        model.model = runtime

        output = model._detect(
            np.zeros(32_000, dtype=np.float32),
            sampling_rate=16_000,
            threshold=0.7,
            onset=0.65,
            offset=0.65,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            speech_pad_ms=50,
            max_speech_duration_s=0.5,
        )

        call = runtime.calls[0]
        self.assertEqual(call["fs"], 16_000)
        self.assertTrue(call["is_final"])
        self.assertEqual(call["cache"], {})
        self.assertEqual(call["speech_noise_thres"], 0.65)
        self.assertEqual(call["max_end_silence_time"], 100)
        self.assertEqual(call["max_single_segment_time"], 500)
        self.assertEqual(call["input"].shape, (32_000, ))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.05, 0.55), (0.55, 0.95), (1.15, 1.65)],
        )
        self.assertEqual(output.duration, 2.0)
        self.assertEqual(output.sample_rate, 16_000)
        self.assertEqual(output.metadata["backend"], "funasr")
        self.assertEqual(output.metadata["native_timestamp_unit"], "milliseconds")
        self.assertFalse(output.metadata["frame_scores_available"])

    def test_path_input_is_loaded_and_resampled_before_native_inference(self):
        runtime = _FakeFunASRRuntime(result=[{"value": [[0, 500]]}])
        model = FunASRVADForVoiceActivityDetection(
            FunASRVADConfig(),
            device="cpu",
        )
        model.model = runtime
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "audio.wav"
            sf.write(
                audio_path,
                np.zeros(8_000, dtype=np.float32),
                8_000,
            )
            output = model._detect(
                audio_path,
                min_speech_duration_ms=0,
                speech_pad_ms=0,
            )

        self.assertEqual(runtime.calls[0]["input"].shape, (16_000, ))
        self.assertEqual(runtime.calls[0]["fs"], 16_000)
        self.assertEqual(output.duration, 1.0)
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.0, 0.5)],
        )

    def test_provider_rejects_unavailable_native_controls(self):
        model = FunASRVADForVoiceActivityDetection(FunASRVADConfig())
        model.model = _FakeFunASRRuntime(result=[{"value": []}])
        audio = np.zeros(16_000, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "frame scores"):
            model._detect(
                audio,
                sampling_rate=16_000,
                return_frames=True,
            )
        with self.assertRaisesRegex(ValueError, "analysis geometry"):
            model._detect(
                audio,
                sampling_rate=16_000,
                window_size_samples=512,
            )
        with self.assertRaisesRegex(ValueError, "independent"):
            model._detect(
                audio,
                sampling_rate=16_000,
                onset=0.6,
                offset=0.4,
            )

    def test_incomplete_streaming_boundaries_fail_in_offline_mode(self):
        model = FunASRVADForVoiceActivityDetection(FunASRVADConfig())
        model.model = _FakeFunASRRuntime(result=[{"value": [[100, -1]]}])

        with self.assertRaisesRegex(RuntimeError, "incomplete streaming"):
            model._detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
            )

    def test_training_boundary_is_explicit(self):
        model = FunASRVADForVoiceActivityDetection(FunASRVADConfig())

        self.assertIsInstance(model, PreTrainedVADModel)
        self.assertIsNone(model.model)
        self.assertEqual(model.training_support, "upstream-custom")
        self.assertFalse(model.supports_generic_finetuning)
        with self.assertRaisesRegex(ValueError, "upstream-custom"):
            model._validate_training_runtime()
