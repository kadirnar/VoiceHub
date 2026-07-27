import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.models.vad_nemo import NeMoVADConfig, NeMoVADForVoiceActivityDetection
from voicehub.models.vad_pyannote import PyannoteVADConfig, PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_speechbrain import SpeechBrainVADConfig, SpeechBrainVADForVoiceActivityDetection


class _FakeTensor:

    def __init__(self, values):
        self.values = np.asarray(values)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    def unsqueeze(self, dimension):
        return _FakeTensor(np.expand_dims(self.values, axis=dimension))

    def to(self, device):
        del device
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def _fake_torch():
    module = ModuleType("torch")
    module.as_tensor = lambda values: _FakeTensor(values)
    return module


class NativeVADProviderTests(unittest.TestCase):

    def test_importing_wrappers_does_not_import_native_providers(self):
        code = """
import json
import sys
from voicehub.models.vad_nemo import NeMoVADForVoiceActivityDetection
from voicehub.models.vad_pyannote import PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_speechbrain import SpeechBrainVADForVoiceActivityDetection
print(json.dumps({
    "nemo": "nemo" in sys.modules,
    "pyannote": "pyannote.audio" in sys.modules,
    "speechbrain": "speechbrain" in sys.modules,
}))
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
            {
                "nemo": False,
                "pyannote": False,
                "speechbrain": False,
            },
        )

    def test_public_classes_are_lazy_task_specific_wrappers(self):
        cases = (
            (PyannoteVADConfig, PyannoteVADForVoiceActivityDetection),
            (SpeechBrainVADConfig, SpeechBrainVADForVoiceActivityDetection),
            (NeMoVADConfig, NeMoVADForVoiceActivityDetection),
        )
        for config_class, model_class in cases:
            with self.subTest(model=model_class.__name__):
                model = model_class(config_class())
                self.assertIsInstance(model, PreTrainedVADModel)
                self.assertIsNone(model.model)
                self.assertEqual(model.training_support, "upstream-custom")
                self.assertFalse(model.supports_generic_finetuning)
                with self.assertRaisesRegex(ValueError, "upstream-custom"):
                    model._validate_training_runtime()

    def test_native_vad_runtimes_never_auto_select_mps(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        model_classes = (
            PyannoteVADForVoiceActivityDetection,
            SpeechBrainVADForVoiceActivityDetection,
            NeMoVADForVoiceActivityDetection,
        )

        with patch.dict(sys.modules, {"torch": fake_torch}):
            for model_class in model_classes:
                with self.subTest(model=model_class.__name__):
                    self.assertEqual(model_class._resolve_device("auto"), "cpu")
                    with self.assertRaisesRegex(ValueError, "CPU and CUDA"):
                        model_class._resolve_device("mps")

    def test_provider_configs_refuse_to_serialize_credentials(self):
        cases = (
            (PyannoteVADConfig, {
                "pipeline_kwargs": {
                    "token": "secret"
                }
            }),
            (SpeechBrainVADConfig, {
                "loader_kwargs": {
                    "fetch_config": object()
                }
            }),
            (NeMoVADConfig, {
                "model_kwargs": {
                    "hub_kwargs": {
                        "use_auth_token": "secret"
                    }
                }
            }),
        )
        for config_class, kwargs in cases:
            with self.subTest(config=config_class.__name__):
                with self.assertRaisesRegex(ValueError, "token|auth"):
                    config_class(**kwargs)

        model = PyannoteVADForVoiceActivityDetection(
            PyannoteVADConfig(),
            token="runtime-secret",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = model.config.save_pretrained(directory)
            serialized = Path(config_path).read_text(encoding="utf-8")
        self.assertNotIn("runtime-secret", serialized)
        self.assertNotIn("token", json.loads(serialized))

    def test_segment_splitters_do_not_emit_floating_point_slivers(self):
        from voicehub.models.vad_pyannote.modeling_vad_pyannote import _finalize_segments as finalize_pyannote
        from voicehub.models.vad_speechbrain.modeling_vad_speechbrain import _finalize_segments as finalize_speechbrain

        values = ({"start": 0.0, "end": 0.30000000000000004}, )
        for finalize in (finalize_pyannote, finalize_speechbrain):
            with self.subTest(finalize=finalize.__module__):
                segments = finalize(
                    values,
                    duration=1.0,
                    sample_rate=16_000,
                    speech_pad_ms=0,
                    max_speech_duration_s=0.1,
                )
                self.assertEqual(
                    [(segment.start, segment.end) for segment in segments],
                    [(0.0, 0.1), (0.1, 0.2), (0.2, 0.30000000000000004)],
                )

    def test_pyannote_loads_hub_pipeline_and_normalizes_array_input(self):
        captured = {}

        class FakePipeline:

            def instantiate(self, parameters):
                captured["parameters"] = parameters

            def __call__(self, payload):
                captured["payload"] = payload
                timeline = [
                    SimpleNamespace(start=0.2, end=0.5),
                    SimpleNamespace(start=0.7, end=0.9),
                ]
                return SimpleNamespace(get_timeline=lambda: SimpleNamespace(support=lambda: timeline))

        pipeline = FakePipeline()

        class PipelineFactory:

            @staticmethod
            def from_pretrained(
                checkpoint,
                revision=None,
                token=None,
                cache_dir=None,
            ):
                captured["load"] = {
                    "checkpoint": checkpoint,
                    "revision": revision,
                    "token": token,
                    "cache_dir": cache_dir,
                }
                return pipeline

        pyannote = ModuleType("pyannote")
        pyannote.__path__ = []
        pyannote_audio = ModuleType("pyannote.audio")
        pyannote_audio.Pipeline = PipelineFactory
        modules = {
            "pyannote": pyannote,
            "pyannote.audio": pyannote_audio,
            "torch": _fake_torch(),
        }
        config = PyannoteVADConfig(
            revision="main",
            inference_config={
                "speech_pad_ms": 0,
                "min_speech_duration_ms": 0,
            },
        )
        model = PyannoteVADForVoiceActivityDetection(
            config,
            device="cpu",
            token="gated-token",
        )
        with patch.dict(sys.modules, modules):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                speech_pad_ms=0,
                min_speech_duration_ms=0,
            )

        self.assertIs(model.model, pipeline)
        self.assertEqual(
            captured["load"]["checkpoint"],
            "pyannote/voice-activity-detection",
        )
        self.assertEqual(captured["load"]["token"], "gated-token")
        self.assertEqual(captured["parameters"]["onset"], 0.5)
        self.assertEqual(captured["payload"]["waveform"].shape, (1, 16_000))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.2, 0.5), (0.7, 0.9)],
        )

    def test_speechbrain_uses_temporary_wav_and_native_boundaries(self):
        captured = {}

        class FakeVADRuntime:

            def get_speech_segments(self, audio_file, **kwargs):
                captured["audio_file"] = audio_file
                captured["path_existed"] = Path(audio_file).is_file()
                captured["detect_options"] = kwargs
                return _FakeTensor([[0.1, 0.4], [0.6, 0.9]])

        runtime = FakeVADRuntime()

        class VADFactory:

            @staticmethod
            def from_hparams(source, **kwargs):
                captured["source"] = source
                captured["load_options"] = kwargs
                return runtime

        speechbrain = ModuleType("speechbrain")
        speechbrain.__path__ = []
        inference = ModuleType("speechbrain.inference")
        inference.__path__ = []
        vad_module = ModuleType("speechbrain.inference.VAD")
        vad_module.VAD = VADFactory
        modules = {
            "speechbrain": speechbrain,
            "speechbrain.inference": inference,
            "speechbrain.inference.VAD": vad_module,
        }
        model = SpeechBrainVADForVoiceActivityDetection(
            SpeechBrainVADConfig(),
            device="cpu",
        )
        with patch.dict(sys.modules, modules):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                speech_pad_ms=0,
            )

        self.assertIs(model.model, runtime)
        self.assertEqual(
            captured["source"],
            "speechbrain/vad-crdnn-libriparty",
        )
        self.assertTrue(captured["path_existed"])
        self.assertFalse(Path(captured["audio_file"]).exists())
        self.assertEqual(captured["detect_options"]["activation_th"], 0.5)
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.1, 0.4), (0.6, 0.9)],
        )

    def test_nemo_selects_native_local_and_catalog_loaders(self):
        calls = []

        class FakeRuntime:

            def to(self, device):
                calls.append(("to", device))
                return self

            def eval(self):
                calls.append(("eval", ))

        runtime = FakeRuntime()

        class WindowModel:

            @classmethod
            def restore_from(cls, **kwargs):
                calls.append(("restore", kwargs))
                return runtime

            @classmethod
            def load_from_checkpoint(cls, **kwargs):
                calls.append(("checkpoint", kwargs))
                return runtime

            @classmethod
            def from_pretrained(cls, **kwargs):
                calls.append(("catalog", kwargs))
                return runtime

        nemo = ModuleType("nemo")
        nemo.__path__ = []
        collections = ModuleType("nemo.collections")
        collections.__path__ = []
        asr = ModuleType("nemo.collections.asr")
        asr.__path__ = []
        models = ModuleType("nemo.collections.asr.models")
        models.EncDecClassificationModel = WindowModel
        models.EncDecFrameClassificationModel = WindowModel
        modules = {
            "nemo": nemo,
            "nemo.collections": collections,
            "nemo.collections.asr": asr,
            "nemo.collections.asr.models": models,
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "vad.nemo"
            checkpoint.touch()
            local_model = NeMoVADForVoiceActivityDetection(
                NeMoVADConfig(name_or_path=checkpoint),
                device="cpu",
            )
            catalog_model = NeMoVADForVoiceActivityDetection(
                NeMoVADConfig(name_or_path="vad_multilingual_marblenet"),
                device="cpu",
            )
            with patch.dict(sys.modules, modules):
                local_model._load_pretrained_model()
                catalog_model._load_pretrained_model()

        self.assertIs(local_model.model, runtime)
        self.assertIs(catalog_model.model, runtime)
        self.assertEqual(calls[0][0], "restore")
        catalog_calls = [call for call in calls if call[0] == "catalog"]
        self.assertEqual(
            catalog_calls[0][1]["model_name"],
            "vad_multilingual_marblenet",
        )

    def test_nemo_window_inference_returns_normalized_frame_scores(self):

        class FakeRuntime:

            def __init__(self):
                self.window_index = 0

            def __call__(self, *, input_signal, input_signal_length):
                del input_signal_length
                logits = []
                for _ in range(input_signal.shape[0]):
                    logits.append([0.0, 4.0] if self.window_index < 2 else [4.0, 0.0])
                    self.window_index += 1
                return _FakeTensor(logits)

        model = NeMoVADForVoiceActivityDetection(
            NeMoVADConfig(
                sample_rate=10,
                architecture_family="window",
                window_duration_s=0.4,
                hop_duration_s=0.2,
                batch_size=2,
            ),
            device="cpu",
        )
        model.model = FakeRuntime()
        model.architecture_family = "window"
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            output = model._detect(
                np.zeros(8, dtype=np.float32),
                sampling_rate=10,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )

        self.assertEqual(len(output.probabilities), 3)
        self.assertGreater(output.probabilities[0], 0.9)
        self.assertLess(output.probabilities[-1], 0.1)
        self.assertEqual(len(output.segments), 1)
        self.assertAlmostEqual(output.segments[0].start, 0.0)
        # The first below-threshold window starts at 0.4 s. It must not be
        # counted as speech; the previous positive window ends at 0.6 s.
        self.assertAlmostEqual(output.segments[0].end, 0.6)

    def test_nemo_binary_logits_support_scalar_window_outputs(self):

        class FakeRuntime:

            def __call__(self, *, input_signal, input_signal_length):
                del input_signal_length
                return _FakeTensor([4.0] * input_signal.shape[0])

        model = NeMoVADForVoiceActivityDetection(
            NeMoVADConfig(
                sample_rate=10,
                architecture_family="window",
                window_duration_s=0.4,
                hop_duration_s=0.2,
            ),
            device="cpu",
        )
        model.model = FakeRuntime()
        model.architecture_family = "window"
        with patch.dict(sys.modules, {"torch": _fake_torch()}):
            output = model._detect(
                np.zeros(8, dtype=np.float32),
                sampling_rate=10,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )

        self.assertTrue(output.probabilities)
        self.assertTrue(all(value > 0.9 for value in output.probabilities))


if __name__ == "__main__":
    unittest.main()
