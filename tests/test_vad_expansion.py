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
from voicehub.models.vad_auditok import AuditokVADConfig, AuditokVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_brouhaha import (
    PyannoteBrouhahaVADConfig,
    PyannoteBrouhahaVADForVoiceActivityDetection,
)
from voicehub.models.vad_pyannote_segmentation import (
    PyannoteSegmentationVADConfig,
    PyannoteSegmentationVADForVoiceActivityDetection,
)
from voicehub.models.vad_sherpa_onnx import (
    SherpaONNXVADConfig,
    SherpaONNXVADForVoiceActivityDetection,
    SherpaONNXVADSession,
)


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
        return _FakeTensor(np.expand_dims(self.values, dimension))


def _fake_torch():
    module = ModuleType("torch")
    module.as_tensor = lambda value: _FakeTensor(value)
    module.device = lambda value: f"device:{value}"
    return module


class VADExpansionConfigTests(unittest.TestCase):

    def test_wrappers_are_lazy_and_do_not_import_provider_runtimes(self):
        code = """
import json
import sys
from voicehub.models.vad_auditok import AuditokVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_brouhaha import PyannoteBrouhahaVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_segmentation import PyannoteSegmentationVADForVoiceActivityDetection
from voicehub.models.vad_sherpa_onnx import SherpaONNXVADForVoiceActivityDetection
print(json.dumps({
    "auditok": "auditok" in sys.modules,
    "pyannote": "pyannote.audio" in sys.modules,
    "sherpa": "sherpa_onnx" in sys.modules,
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
                "auditok": False,
                "pyannote": False,
                "sherpa": False,
            },
        )

    def test_public_wrappers_follow_the_vad_lifecycle(self):
        cases = (
            (
                AuditokVADForVoiceActivityDetection,
                AuditokVADConfig(),
                "inference-only",
            ),
            (
                SherpaONNXVADForVoiceActivityDetection,
                SherpaONNXVADConfig(),
                "inference-only",
            ),
            (
                PyannoteSegmentationVADForVoiceActivityDetection,
                PyannoteSegmentationVADConfig(),
                "upstream-custom",
            ),
            (
                PyannoteBrouhahaVADForVoiceActivityDetection,
                PyannoteBrouhahaVADConfig(),
                "upstream-custom",
            ),
        )
        for model_class, config, support in cases:
            with self.subTest(model=model_class.__name__):
                model = model_class(config)
                self.assertIsInstance(model, PreTrainedVADModel)
                self.assertIsNone(model.model)
                self.assertEqual(model.training_support, support)
                self.assertFalse(model.supports_generic_finetuning)
                with self.assertRaises(ValueError):
                    model._validate_training_runtime()

    def test_auditok_calibration_config_is_validated_and_serializable(self):
        config = AuditokVADConfig(
            threshold_method="p20",
            energy_threshold_db=47.5,
            analysis_window_s=0.02,
            calibration_duration_s=2.5,
            minimum_energy_threshold_db=35,
            strict_min_duration=True,
        )
        restored = AuditokVADConfig.from_dict(config.to_dict())
        self.assertEqual(restored.threshold_method, "p20")
        self.assertEqual(restored.analysis_window_s, 0.02)
        self.assertTrue(restored.strict_min_duration)
        for kwargs in (
            {"threshold_method": "p0"},
            {"threshold_method": "median"},
            {"analysis_window_s": 0.2},
            {"calibration_duration_s": 0},
            {"inference_config": {"token": "secret"}},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    AuditokVADConfig(**kwargs)

    def test_sherpa_config_rejects_unsafe_assets_and_credentials(self):
        config = SherpaONNXVADConfig(
            model_family="ten",
            subfolder="onnx/vad",
            local_files_only=True,
            provider="coreml",
        )
        self.assertEqual(config.model_filename, "ten-vad.onnx")
        self.assertEqual(config.window_size_samples, 256)
        self.assertEqual(config.subfolder, "onnx/vad")
        self.assertEqual(
            SherpaONNXVADForVoiceActivityDetection(config).device,
            "cpu",
        )
        for kwargs in (
            {"model_filename": "../model.onnx"},
            {"model_filename": "/tmp/model.onnx"},
            {"model_filename": "model.pt"},
            {"subfolder": "../weights"},
            {"model_family": "unknown"},
            {"provider": "tensorrt"},
            {"kwargs": {"token": "secret"}},
        ):
            with self.subTest(kwargs=kwargs):
                if "kwargs" in kwargs:
                    kwargs = kwargs["kwargs"]
                with self.assertRaises((TypeError, ValueError)):
                    SherpaONNXVADConfig(**kwargs)

    def test_runtime_tokens_are_not_persisted(self):
        model = SherpaONNXVADForVoiceActivityDetection(
            SherpaONNXVADConfig(),
            token="runtime-only",
        )
        with tempfile.TemporaryDirectory() as directory:
            config_path = model.config.save_pretrained(directory)
            serialized = Path(config_path).read_text(encoding="utf-8")
        self.assertNotIn("runtime-only", serialized)
        self.assertNotIn("token", json.loads(serialized))


class AuditokVADRuntimeTests(unittest.TestCase):

    def test_energy_detection_uses_raw_pcm_and_normalizes_seconds(self):
        captured = {}

        class Region:

            def __init__(self, start, end):
                self.start = start
                self.end = end

        module = ModuleType("auditok")

        def split(audio, **kwargs):
            captured["audio"] = audio
            captured["kwargs"] = kwargs
            return iter((Region(0.1, 0.4), Region(0.7, 1.5)))

        module.split = split
        model = AuditokVADForVoiceActivityDetection(
            AuditokVADConfig(
                threshold_method="otsu",
                analysis_window_s=0.02,
                calibration_duration_s=2,
                minimum_energy_threshold_db=38,
            ))
        with patch.dict(sys.modules, {"auditok": module}):
            output = model.detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=120,
                speech_pad_ms=20,
            )

        self.assertIsInstance(captured["audio"], bytes)
        self.assertEqual(len(captured["audio"]), 32_000)
        self.assertEqual(captured["kwargs"]["validator"], "otsu")
        self.assertEqual(captured["kwargs"]["calibration_dur"], 2)
        self.assertEqual(captured["kwargs"]["min_energy_threshold"], 38)
        self.assertEqual(captured["kwargs"]["analysis_window"], 0.02)
        self.assertEqual(captured["kwargs"]["min_dur"], 0.02)
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.1, 0.4), (0.7, 1.0)],
        )
        self.assertEqual(output.metadata["threshold_method"], "otsu")
        self.assertIsNone(output.probabilities)

    def test_probability_options_are_rejected_instead_of_misrepresented(self):
        model = AuditokVADForVoiceActivityDetection(AuditokVADConfig())
        model.model = SimpleNamespace(split=lambda *args, **kwargs: ())
        for kwargs in (
            {"threshold": 0.8},
            {"onset": 0.6},
            {"offset": 0.4},
            {"return_frames": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "calibrated speech probabilities"):
                    model._detect(
                        np.zeros(160, dtype=np.float32),
                        sampling_rate=16_000,
                        **kwargs,
                    )


class _FakeSherpaConfig:

    def __init__(self):
        self.silero_vad = SimpleNamespace()
        self.ten_vad = SimpleNamespace()
        self.sample_rate = None
        self.num_threads = None
        self.provider = None
        self.debug = None

    def validate(self):
        return True


class _FakeSherpaDetector:
    instances = []

    def __init__(self, config, buffer_size_in_seconds):
        self.config = config
        self.buffer_size_in_seconds = buffer_size_in_seconds
        self.queue = []
        self.accepted = []
        self.was_reset = False
        self.__class__.instances.append(self)

    def accept_waveform(self, samples):
        self.accepted.append(np.asarray(samples))
        if len(self.accepted) == 1:
            self.queue.append(SimpleNamespace(
                start=160,
                samples=np.zeros(320, dtype=np.float32),
            ))

    def flush(self):
        self.queue.append(SimpleNamespace(
            start=800,
            samples=np.zeros(160, dtype=np.float32),
        ))

    def empty(self):
        return not self.queue

    @property
    def front(self):
        return self.queue[0]

    def pop(self):
        self.queue.pop(0)

    def reset(self):
        self.queue.clear()
        self.accepted.clear()
        self.was_reset = True


def _fake_sherpa():
    _FakeSherpaDetector.instances.clear()
    module = ModuleType("sherpa_onnx")
    module.VadModelConfig = _FakeSherpaConfig
    module.VoiceActivityDetector = _FakeSherpaDetector
    return module


class SherpaONNXVADRuntimeTests(unittest.TestCase):

    def test_hub_asset_and_native_options_are_applied(self):
        module = _fake_sherpa()
        captured = {}
        model = SherpaONNXVADForVoiceActivityDetection(
            SherpaONNXVADConfig(
                name_or_path="org/vad",
                revision="v1",
                subfolder="runtime",
                cache_dir="cache",
                local_files_only=True,
                num_threads=3,
                buffer_size_s=12,
            ),
            token="hub-token",
        )

        def resolve(source, filename, **kwargs):
            captured["source"] = source
            captured["filename"] = filename
            captured["resolve_kwargs"] = kwargs
            return Path("/tmp/silero_vad.onnx")

        with (patch.dict(sys.modules, {"sherpa_onnx": module}), patch(
                "voicehub.models.vad_sherpa_onnx."
                "modeling_vad_sherpa_onnx.resolve_pretrained_file",
                side_effect=resolve,
        )):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(1_600, dtype=np.float32),
                sampling_rate=16_000,
                threshold=0.4,
                offset=0.2,
                min_speech_duration_ms=100,
                min_silence_duration_ms=80,
                speech_pad_ms=0,
                max_speech_duration_s=2,
                window_size_samples=400,
            )

        detector = _FakeSherpaDetector.instances[0]
        native = detector.config.silero_vad
        self.assertEqual(captured["source"], "org/vad")
        self.assertEqual(captured["filename"], "silero_vad.onnx")
        self.assertEqual(captured["resolve_kwargs"]["revision"], "v1")
        self.assertEqual(captured["resolve_kwargs"]["token"], "hub-token")
        self.assertTrue(captured["resolve_kwargs"]["local_files_only"])
        self.assertEqual(native.model, "/tmp/silero_vad.onnx")
        self.assertEqual(native.threshold, 0.4)
        self.assertEqual(native.neg_threshold, 0.2)
        self.assertEqual(native.min_speech_duration, 0.1)
        self.assertEqual(native.min_silence_duration, 0.08)
        self.assertEqual(native.max_speech_duration, 2)
        self.assertEqual(native.window_size, 400)
        self.assertEqual(detector.config.num_threads, 3)
        np.testing.assert_allclose(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.01, 0.03), (0.05, 0.06)],
        )
        self.assertEqual(output.metadata["model_family"], "silero")

    def test_streaming_session_is_incremental_idempotent_and_resettable(self):
        module = _fake_sherpa()
        model = SherpaONNXVADForVoiceActivityDetection(
            SherpaONNXVADConfig(
                name_or_path="local",
                window_size_samples=256,
            ))
        with (patch.dict(sys.modules, {"sherpa_onnx": module}), patch(
                "voicehub.models.vad_sherpa_onnx."
                "modeling_vad_sherpa_onnx.resolve_pretrained_file",
                return_value=Path("/tmp/silero_vad.onnx"),
        )):
            model._load_pretrained_model()
            session = model.stream(
                sampling_rate=16_000,
                speech_pad_ms=0,
            )
            self.assertIsInstance(session, SherpaONNXVADSession)
            self.assertEqual(
                [(item.start, item.end) for item in session.push(np.zeros(300, dtype=np.float32))],
                [(0.01, 0.03)],
            )
            self.assertEqual(session.push(np.zeros(100, dtype=np.float32)), ())
            first = session.flush()
            second = session.flush()
            self.assertIs(first, second)
            self.assertAlmostEqual(first.duration, 0.025)
            self.assertTrue(all(segment.end <= first.duration for segment in first.segments))

            session.reset()
            self.assertTrue(_FakeSherpaDetector.instances[0].was_reset)
            session.push(np.zeros(256, dtype=np.float32))
            self.assertGreater(len(session.flush().segments), 0)
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.push(np.zeros(256, dtype=np.float32))

    def test_ten_vad_rejects_an_unsupported_offset_threshold(self):
        module = _fake_sherpa()
        model = SherpaONNXVADForVoiceActivityDetection(
            SherpaONNXVADConfig(
                name_or_path="local",
                model_family="ten",
            ))
        with (patch.dict(sys.modules, {"sherpa_onnx": module}), patch(
                "voicehub.models.vad_sherpa_onnx."
                "modeling_vad_sherpa_onnx.resolve_pretrained_file",
                return_value=Path("/tmp/ten-vad.onnx"),
        )):
            model._load_pretrained_model()
            with self.assertRaisesRegex(ValueError, "TEN VAD"):
                model._detect(
                    np.zeros(512, dtype=np.float32),
                    sampling_rate=16_000,
                    offset=0.3,
                )

    def test_direct_local_onnx_files_do_not_require_a_matching_default_name(self):
        module = _fake_sherpa()
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "custom-vad.onnx"
            model_path.touch()
            model = SherpaONNXVADForVoiceActivityDetection(SherpaONNXVADConfig(name_or_path=model_path))
            with (patch.dict(sys.modules, {"sherpa_onnx": module}),
                  patch("voicehub.models.vad_sherpa_onnx."
                        "modeling_vad_sherpa_onnx.resolve_pretrained_file", ) as resolve):
                model._load_pretrained_model()

        resolve.assert_not_called()
        self.assertEqual(model.model.model_path, model_path.resolve())

    def test_streaming_validates_common_inference_values_and_closed_state(self):
        module = _fake_sherpa()
        model = SherpaONNXVADForVoiceActivityDetection(SherpaONNXVADConfig(name_or_path="local"))
        with (patch.dict(sys.modules, {"sherpa_onnx": module}), patch(
                "voicehub.models.vad_sherpa_onnx."
                "modeling_vad_sherpa_onnx.resolve_pretrained_file",
                return_value=Path("/tmp/silero_vad.onnx"),
        )):
            model._load_pretrained_model()
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                model.stream(sampling_rate=16_000, threshold=2)
            session = model.stream(sampling_rate=16_000)
            session.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                session.flush()


class PyannotePresetRuntimeTests(unittest.TestCase):

    def test_presets_protect_their_managed_runtime_options(self):
        with self.assertRaisesRegex(ValueError, "managed segmentation"):
            PyannoteSegmentationVADConfig(pipeline_kwargs={"segmentation": "other/model"})
        with self.assertRaisesRegex(ValueError, "not supported"):
            PyannoteBrouhahaVADConfig(pipeline_kwargs={"batch_size": 1})

    def test_segmentation_checkpoint_is_wrapped_in_the_public_vad_pipeline(self):
        captured = {}

        class SegmentationFactory:

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
                return "segmentation-model"

        class Pipeline:

            def __init__(self, segmentation, **kwargs):
                captured["segmentation"] = segmentation
                captured["pipeline_kwargs"] = kwargs

            def instantiate(self, parameters):
                captured["parameters"] = parameters

            def __call__(self, payload):
                captured["payload"] = payload
                timeline = (SimpleNamespace(start=0.2, end=0.5), )
                return SimpleNamespace(get_timeline=lambda: SimpleNamespace(support=lambda: timeline))

        pyannote = ModuleType("pyannote")
        pyannote.__path__ = []
        audio = ModuleType("pyannote.audio")
        audio.__path__ = []
        audio.Model = SegmentationFactory
        pipelines = ModuleType("pyannote.audio.pipelines")
        pipelines.VoiceActivityDetection = Pipeline
        modules = {
            "pyannote": pyannote,
            "pyannote.audio": audio,
            "pyannote.audio.pipelines": pipelines,
            "torch": _fake_torch(),
        }
        model = PyannoteSegmentationVADForVoiceActivityDetection(
            PyannoteSegmentationVADConfig(
                revision="main",
                cache_dir="cache",
                pipeline_kwargs={"batch_size": 8},
            ),
            device="cpu",
            token="gated-token",
        )
        with patch.dict(sys.modules, modules):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                speech_pad_ms=0,
            )

        self.assertEqual(
            captured["load"],
            {
                "checkpoint": "pyannote/segmentation-3.0",
                "revision": "main",
                "token": "gated-token",
                "cache_dir": "cache",
            },
        )
        self.assertEqual(captured["segmentation"], "segmentation-model")
        self.assertEqual(captured["pipeline_kwargs"], {"batch_size": 8})
        self.assertEqual(captured["parameters"]["onset"], 0.5)
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.2, 0.5)],
        )

    def test_brouhaha_returns_vad_frames_and_auxiliary_summaries(self):
        captured = {}

        class ModelFactory:

            @staticmethod
            def from_pretrained(checkpoint, token=None):
                captured["load"] = {
                    "checkpoint": checkpoint,
                    "token": token,
                }
                return "brouhaha-model"

        class Inference:

            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["inference_kwargs"] = kwargs

            def __call__(self, payload):
                captured["payload"] = payload
                return SimpleNamespace(
                    data=np.asarray([
                        [0.1, 10.0, 2.0],
                        [0.9, 20.0, 4.0],
                        [0.8, 30.0, 6.0],
                        [0.1, 40.0, 8.0],
                    ]),
                    sliding_window=SimpleNamespace(
                        start=0.0,
                        step=0.1,
                        duration=0.1,
                    ),
                )

        pyannote = ModuleType("pyannote")
        pyannote.__path__ = []
        audio = ModuleType("pyannote.audio")
        audio.Model = ModelFactory
        audio.Inference = Inference
        brouhaha = ModuleType("brouhaha")
        brouhaha.__path__ = []
        brouhaha_models = ModuleType("brouhaha.models")
        modules = {
            "brouhaha": brouhaha,
            "brouhaha.models": brouhaha_models,
            "pyannote": pyannote,
            "pyannote.audio": audio,
            "torch": _fake_torch(),
        }
        model = PyannoteBrouhahaVADForVoiceActivityDetection(
            PyannoteBrouhahaVADConfig(batch_size=4),
            device="cpu",
            token="gated-token",
        )
        with patch.dict(sys.modules, modules):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(16_000, dtype=np.float32),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )

        self.assertEqual(
            captured["load"],
            {
                "checkpoint": "pyannote/brouhaha",
                "token": "gated-token",
            },
        )
        self.assertEqual(captured["model"], "brouhaha-model")
        self.assertEqual(captured["inference_kwargs"]["batch_size"], 4)
        self.assertEqual(captured["payload"]["waveform"].shape, (1, 16_000))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.1, 0.3)],
        )
        np.testing.assert_allclose(output.probabilities, [0.1, 0.9, 0.8, 0.1])
        self.assertEqual(output.metadata["mean_snr_db"], 25.0)
        self.assertEqual(output.metadata["mean_c50_db"], 5.0)
        self.assertEqual(output.metadata["auxiliary_outputs"], ("snr_db", "c50_db"))

    def test_brouhaha_provides_its_checkpoint_architecture_during_loading(self):
        captured = {}
        architecture = ModuleType("voicehub_brouhaha_architecture")
        architecture.CustomPyanNetModel = type("CustomPyanNetModel", (), {})

        def import_architecture(name):
            if name == "brouhaha.models":
                error = ModuleNotFoundError("No module named 'brouhaha'")
                error.name = "brouhaha"
                raise error
            if name == "voicehub.models.vad_pyannote_brouhaha._architecture":
                return architecture
            raise AssertionError(f"unexpected import: {name}")

        class ModelFactory:

            @staticmethod
            def from_pretrained(checkpoint):
                captured["checkpoint"] = checkpoint
                from brouhaha.models import CustomPyanNetModel
                captured["architecture"] = CustomPyanNetModel
                return "brouhaha-model"

        class Inference:

            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["inference_kwargs"] = kwargs

        pyannote = ModuleType("pyannote")
        pyannote.__path__ = []
        audio = ModuleType("pyannote.audio")
        audio.Model = ModelFactory
        audio.Inference = Inference
        model = PyannoteBrouhahaVADForVoiceActivityDetection(
            PyannoteBrouhahaVADConfig(),
            device="cpu",
        )
        with patch.dict(sys.modules, {
                "pyannote": pyannote,
                "pyannote.audio": audio,
                "torch": _fake_torch(),
        }), patch(
                "voicehub.models.vad_pyannote_brouhaha."
                "modeling_vad_pyannote_brouhaha.import_module",
                side_effect=import_architecture,
        ):
            self.assertNotIn("brouhaha.models", sys.modules)
            model._load_pretrained_model()
            self.assertNotIn("brouhaha.models", sys.modules)

        self.assertEqual(captured["checkpoint"], "pyannote/brouhaha")
        self.assertEqual(captured["architecture"].__name__, "CustomPyanNetModel")
        self.assertEqual(captured["model"], "brouhaha-model")

    def test_brouhaha_does_not_mask_nested_architecture_import_errors(self):

        class ModelFactory:

            @staticmethod
            def from_pretrained(checkpoint):
                del checkpoint
                raise AssertionError("loader should not be reached")

        pyannote = ModuleType("pyannote")
        pyannote.__path__ = []
        audio = ModuleType("pyannote.audio")
        audio.Model = ModelFactory
        audio.Inference = object
        model = PyannoteBrouhahaVADForVoiceActivityDetection(
            PyannoteBrouhahaVADConfig(),
            device="cpu",
        )
        error = ModuleNotFoundError("No module named 'pyannote.audio.models'")
        error.name = "pyannote.audio.models"
        with (
                patch.dict(sys.modules, {
                    "pyannote": pyannote,
                    "pyannote.audio": audio,
                }),
                patch(
                    "voicehub.models.vad_pyannote_brouhaha."
                    "modeling_vad_pyannote_brouhaha.import_module",
                    side_effect=error,
                ),
                self.assertRaises(ModuleNotFoundError),
        ):
            model._load_pretrained_model()


if __name__ == "__main__":
    unittest.main()
