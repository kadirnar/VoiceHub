import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from voicehub.models.vad_silero import (
    SileroVADConfig,
    SileroVADForVoiceActivityDetection,
)
from voicehub.models.vad_transformers import (
    TransformersVADConfig,
    TransformersVADForVoiceActivityDetection,
)
from voicehub.models.vad_webrtc import (
    WebRTCVADConfig,
    WebRTCVADForVoiceActivityDetection,
)
from voicehub.models.vad_webrtc.modeling_vad_webrtc import _pcm16_samples

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def _temporary_modules(modules):
    missing = object()
    originals = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, original in originals.items():
            if original is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _FakeTensor:

    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = "cpu"

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return _FakeTensor(self.values[index])

    def to(self, device):
        self.device = device
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        self.device = "cpu"
        return self

    def tolist(self):
        return self.values.tolist()


class _InferenceRuntime:

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self


def _fake_torch():
    module = ModuleType("torch")
    module.as_tensor = lambda values: _FakeTensor(values)
    module.inference_mode = nullcontext

    def softmax(tensor, dim=-1):
        values = tensor.values
        shifted = values - np.max(values, axis=dim, keepdims=True)
        exponentials = np.exp(shifted)
        return _FakeTensor(exponentials / exponentials.sum(axis=dim, keepdims=True))

    def sigmoid(tensor):
        return _FakeTensor(1.0 / (1.0 + np.exp(-tensor.values)))

    module.softmax = softmax
    module.sigmoid = sigmoid
    return module


class VADProviderContractTests(unittest.TestCase):

    def test_importing_wrappers_keeps_all_runtime_backends_lazy(self):
        code = """
import json
import sys
from voicehub.models.vad_silero import SileroVADForVoiceActivityDetection
from voicehub.models.vad_transformers import TransformersVADForVoiceActivityDetection
from voicehub.models.vad_webrtc import WebRTCVADForVoiceActivityDetection
optional = ("silero_vad", "torch", "transformers", "webrtcvad")
print(json.dumps({name: name in sys.modules for name in optional}))
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(result.stdout.strip().splitlines()[-1]),
            {
                "silero_vad": False,
                "torch": False,
                "transformers": False,
                "webrtcvad": False,
            },
        )

    def test_provider_configurations_reject_unsupported_audio_contracts(self):
        invalid = (
            lambda: SileroVADConfig(sample_rate=44_100),
            lambda: SileroVADConfig(use_onnx=1),
            lambda: WebRTCVADConfig(sample_rate=44_100),
            lambda: WebRTCVADConfig(aggressiveness=4),
            lambda: WebRTCVADConfig(frame_duration_ms=15),
            lambda: TransformersVADConfig(architecture_family="token-classification"),
            lambda: TransformersVADConfig(speech_labels=()),
            lambda: TransformersVADConfig(speech_labels="speech"),
            lambda: TransformersVADConfig(window_duration_s=True),
            lambda: TransformersVADConfig(hop_duration_s=float("nan")),
            lambda: TransformersVADConfig(window_duration_s=0.5, hop_duration_s=0.6),
            lambda: TransformersVADConfig(
                processor_kwargs={
                    "nested": {
                        "use_auth_token": "must-not-be-persisted",
                    },
                }),
        )
        for factory in invalid:
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()

    def test_native_vad_device_contracts_are_provider_specific(self):
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True), )
        with _temporary_modules({"torch": fake_torch}):
            self.assertEqual(
                SileroVADForVoiceActivityDetection._resolve_device("auto"),
                "cpu",
            )
            self.assertEqual(
                WebRTCVADForVoiceActivityDetection._resolve_device("auto"),
                "cpu",
            )
        with self.assertRaisesRegex(ValueError, "CPU and CUDA"):
            SileroVADForVoiceActivityDetection._resolve_device("mps")
        with self.assertRaisesRegex(ValueError, "CPU-only"):
            WebRTCVADForVoiceActivityDetection._resolve_device("mps")

    def test_fine_tuning_boundaries_match_backend_capabilities(self):
        silero = SileroVADForVoiceActivityDetection(SileroVADConfig())
        self.assertIsNone(silero._validate_training_runtime())
        self.assertEqual(
            type(silero.get_training_adapter()).__name__,
            "NativeSileroVADTrainingAdapter",
        )

        webrtc = WebRTCVADForVoiceActivityDetection(WebRTCVADConfig())
        with self.assertRaisesRegex(ValueError, "not applicable"):
            webrtc._validate_training_runtime()

        trainable = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        self.assertIsNone(trainable._validate_training_runtime())

    def test_native_silero_rejects_removed_external_runtime_modes(self):
        with self.assertRaisesRegex(ValueError, "does not execute ONNX"):
            SileroVADConfig(use_onnx=True)
        with self.assertRaisesRegex(ValueError, "immutable, verified"):
            SileroVADConfig(force_reload=True)


class SileroVADInferenceTests(unittest.TestCase):

    def test_native_silero_resolves_auto_device_before_model_placement(self):
        captured = {}

        class Runtime:

            def __init__(self, config):
                captured["config"] = config

            def to(self, *, device):
                captured["placed_on"] = device
                return self

        model = SileroVADForVoiceActivityDetection(
            SileroVADConfig(),
            device="auto",
        )
        artifact = SimpleNamespace(checkpoint_format="safetensors")
        with (
                patch.object(
                    SileroVADForVoiceActivityDetection,
                    "_resolve_device",
                    return_value="cpu",
                ) as resolve_device,
                patch(
                    "voicehub.models.vad_silero.artifacts."
                    "resolve_silero_vad_artifact",
                    return_value=artifact,
                ),
                patch(
                    "voicehub.models.vad_silero.artifacts."
                    "load_silero_vad_checkpoint",
                    return_value=("safetensors", "voicehub-native-silero-vad"),
                ),
                patch(
                    "voicehub.architectures.silero_vad.modeling.SileroVADModel",
                    Runtime,
                ),
        ):
            model._load_pretrained_model()

        resolve_device.assert_called_once_with("auto")
        self.assertEqual(model.device, "cpu")
        self.assertEqual(captured["placed_on"], "cpu")
        self.assertIs(model.artifact, artifact)

    def test_native_silero_maps_controls_and_returns_frame_scores(self):
        import torch

        from voicehub.architectures.silero_vad.configuration import (
            SileroVADConfig as NativeSileroVADConfig,
        )

        captured = {}

        class Runtime:

            def frame_probabilities(self, waveform, *, pad_final_frame):
                captured["waveform"] = waveform
                captured["pad_final_frame"] = pad_final_frame
                return SimpleNamespace(
                    probabilities=torch.tensor([[0.1, 0.9, 0.9]]),
                    valid_samples=waveform.shape[-1],
                )

        model = SileroVADForVoiceActivityDetection(
            SileroVADConfig(),
            device="cpu",
        )
        model.native_config = NativeSileroVADConfig()
        model.model = Runtime()
        model.checkpoint_format = "safetensors"
        output = model._detect(
            np.zeros(1_300, dtype=np.float32),
            sampling_rate=16_000,
            onset=0.65,
            offset=0.35,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=4.0,
            window_size_samples=512,
            return_frames=True,
        )

        self.assertEqual(captured["waveform"].shape, (1, 1_300))
        self.assertTrue(captured["pad_final_frame"])
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(512 / 16_000, 1_300 / 16_000)],
        )
        self.assertEqual(output.duration, 1_300 / 16_000)
        np.testing.assert_allclose(output.probabilities, (0.1, 0.9, 0.9))
        self.assertTrue(output.metadata["frame_scores_available"])
        self.assertEqual(output.metadata["backend"], "voicehub-native")

    def test_native_silero_frame_scores_are_direct_model_probabilities(self):
        import torch

        from voicehub.architectures.silero_vad.configuration import (
            SileroVADConfig as NativeSileroVADConfig,
        )

        model = SileroVADForVoiceActivityDetection(
            SileroVADConfig(),
            device="cpu",
        )
        model.native_config = NativeSileroVADConfig()
        segmentation = model._segmentation_config(
            threshold=0.5,
            onset=None,
            offset=None,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            max_speech_duration_s=None,
            window_size_samples=512,
        )
        output = model._probabilities_to_output(
            torch.tensor([0.2, 0.8]),
            valid_samples=1_024,
            segmentation_config=segmentation,
            return_frames=True,
            streaming=False,
        )

        np.testing.assert_allclose(output.probabilities, (0.2, 0.8))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(512 / 16_000, 1_024 / 16_000)],
        )


class WebRTCVADInferenceTests(unittest.TestCase):

    def test_webrtc_vectorized_pcm_conversion_matches_scalar_reference(self):
        values = np.array(
            [
                -np.inf,
                -2.0,
                -1.0,
                -2.5 / 32767,
                -1.5 / 32767,
                -0.5 / 32767,
                0.0,
                0.5 / 32767,
                1.5 / 32767,
                2.5 / 32767,
                1.0,
                2.0,
                np.inf,
                np.nan,
            ],
            dtype=np.float64,
        )
        expected = []
        for value in values:
            normalized = float(value)
            if not np.isfinite(normalized):
                normalized = 0.0
            normalized = max(-1.0, min(1.0, normalized))
            expected.append(round(normalized * 32767.0))

        self.assertEqual(
            _pcm16_samples(torch.from_numpy(values)),
            expected,
        )

    def test_webrtc_frames_pcm_and_uses_request_local_native_runtime(self):
        instances = []
        decisions = (False, True, True, False)

        class NativeRuntime:

            def __init__(self, aggressiveness):
                self.aggressiveness = aggressiveness
                self.calls = []
                instances.append(self)

            def is_speech(self, frame, sample_rate):
                self.calls.append((frame, sample_rate))
                return decisions[len(self.calls) - 1]

        with patch(
                "voicehub.models.vad_webrtc.modeling_vad_webrtc."
                "NativeWebRTCVAD",
                NativeRuntime,
        ):
            model = WebRTCVADForVoiceActivityDetection(
                WebRTCVADConfig(
                    aggressiveness=3,
                    frame_duration_ms=10,
                ),
                device="cpu",
            )
            output = model.detect(
                np.linspace(-2.0, 2.0, 640, dtype=np.float32),
                sampling_rate=16_000,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
            )

        self.assertEqual(len(instances), 2)
        self.assertIs(model.model, instances[0])
        self.assertIsNot(instances[0], instances[1])
        self.assertEqual(instances[1].aggressiveness, 3)
        self.assertEqual(len(instances[1].calls), 4)
        self.assertTrue(all(len(frame) == 160 for frame, _ in instances[1].calls))
        self.assertTrue(all(rate == 16_000 for _, rate in instances[1].calls))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.01, 0.03)],
        )
        self.assertIsNone(output.segments[0].score)
        self.assertIsNone(output.probabilities)
        self.assertFalse(output.metadata["frame_scores_available"])

    def test_webrtc_rejects_controls_its_binary_runtime_cannot_honor(self):
        model = WebRTCVADForVoiceActivityDetection(
            WebRTCVADConfig(),
            device="cpu",
        )

        for option in (
            {"threshold": 0.75},
            {"onset": 0.6},
            {"offset": 0.4},
            {"return_frames": True},
        ):
            with (
                    self.subTest(option=option),
                    self.assertRaisesRegex(ValueError, next(iter(option))),
            ):
                model._detect(
                    np.zeros(160, dtype=np.float32),
                    sampling_rate=16_000,
                    **option,
                )

    def test_webrtc_rejects_non_native_window_size(self):
        model = WebRTCVADForVoiceActivityDetection(
            WebRTCVADConfig(frame_duration_ms=20),
            device="cpu",
        )

        model._load_pretrained_model()
        with self.assertRaisesRegex(ValueError, "expected 320"):
            model._detect(
                np.zeros(320, dtype=np.float32),
                sampling_rate=16_000,
                window_size_samples=160,
            )


@unittest.skip("Legacy mock-Transformers contract was replaced by the native "
               "Wav2Vec2 provider tests.")
class TransformersVADInferenceTests(unittest.TestCase):

    @staticmethod
    def _fake_transformers(*, frame_classification=False):
        captured = {}
        module = ModuleType("transformers")
        architecture = (
            "FutureForAudioFrameClassification" if frame_classification else "ASTForAudioClassification")
        native_config = SimpleNamespace(
            architectures=[architecture],
            id2label={
                0: "silence",
                1: "speech",
            },
        )

        class AutoConfig:

            @classmethod
            def from_pretrained(cls, source, **kwargs):
                captured["config"] = (source, kwargs)
                return native_config

        class FeatureExtractor:
            sampling_rate = 10

            def __call__(self, audio, **kwargs):
                captured.setdefault("processor_calls", []).append((audio, kwargs))
                return {"input_values": _FakeTensor(audio)}

            def save_pretrained(self, directory):
                captured["processor_saved"] = Path(directory)

        feature_extractor = FeatureExtractor()

        class AutoFeatureExtractor:

            @classmethod
            def from_pretrained(cls, source, **kwargs):
                captured["processor_load"] = (source, kwargs)
                return feature_extractor

        class Runtime(_InferenceRuntime):

            def to(self, device):
                captured["device"] = device
                return self

            def __call__(self, **kwargs):
                captured["model_inputs"] = kwargs
                if frame_classification:
                    logits = [[
                        [3.0, 0.0],
                        [0.0, 3.0],
                        [0.0, 3.0],
                        [3.0, 0.0],
                    ]]
                else:
                    batch_size = kwargs["input_values"].shape[0]
                    logits = [
                        [0.0, 3.0],
                        [0.0, 3.0],
                        [3.0, 0.0],
                    ][:batch_size]
                return SimpleNamespace(logits=_FakeTensor(logits))

            def save_pretrained(self, directory, **kwargs):
                captured["model_saved"] = (Path(directory), kwargs)

        runtime = Runtime()

        class ModelLoader:

            @classmethod
            def from_pretrained(cls, source, **kwargs):
                captured["model_load"] = (source, kwargs)
                return runtime

        module.AutoConfig = AutoConfig
        module.AutoFeatureExtractor = AutoFeatureExtractor
        module.AutoProcessor = AutoFeatureExtractor
        module.AutoModelForAudioClassification = ModelLoader
        module.AutoModelForAudioFrameClassification = ModelLoader
        return module, captured

    def test_clip_classifier_is_windowed_and_returns_requested_scores(self):
        transformers, captured = self._fake_transformers()
        model = TransformersVADForVoiceActivityDetection(
            TransformersVADConfig(
                name_or_path="publisher/vad",
                architecture_family="auto",
                window_duration_s=0.4,
                hop_duration_s=0.2,
            ),
            device="cpu",
        )

        with _temporary_modules({
                "transformers": transformers,
                "torch": _fake_torch(),
        }):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(8, dtype=np.float32),
                sampling_rate=10,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
                return_frames=True,
            )

        self.assertEqual(model.architecture_family, "audio-classification")
        self.assertEqual(captured["config"][0], "publisher/vad")
        self.assertFalse(captured["config"][1]["trust_remote_code"])
        self.assertFalse(captured["model_load"][1]["trust_remote_code"])
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(captured["model_inputs"]["input_values"].shape, (3, 4))
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.0, 0.6)],
        )
        self.assertEqual(output.probabilities.shape, (3, ))
        self.assertEqual(output.metadata["speech_class_id"], 1)
        self.assertEqual(output.metadata["frame_hop_samples"], 2)
        self.assertEqual(output.metadata["frame_length_samples"], 4)

    def test_frame_classifier_dispatch_and_training_preprocessing(self):
        transformers, captured = self._fake_transformers(frame_classification=True)
        model = TransformersVADForVoiceActivityDetection(
            TransformersVADConfig(
                name_or_path="publisher/frame-vad",
                architecture_family="auto",
            ),
            device="cpu",
        )

        with _temporary_modules({
                "transformers": transformers,
                "torch": _fake_torch(),
        }):
            model._load_pretrained_model()
            output = model._detect(
                np.zeros(8, dtype=np.float32),
                sampling_rate=10,
                min_speech_duration_ms=0,
                min_silence_duration_ms=0,
                speech_pad_ms=0,
            )
            training_inputs = model.prepare_training_inputs(
                {
                    "audio": [
                        np.zeros(4, dtype=np.float32),
                        np.ones(4, dtype=np.float32),
                    ],
                    "sampling_rate": 10,
                    "labels": [0, 1],
                },
                phase="vad",
            )
            with tempfile.TemporaryDirectory() as directory:
                model._save_pretrained(Path(directory))
                saved_directory = Path(directory)
                self.assertEqual(captured["model_saved"][0], saved_directory)
                self.assertEqual(captured["processor_saved"], saved_directory)

        self.assertEqual(model.architecture_family, "frame-classification")
        self.assertEqual(
            [(segment.start, segment.end) for segment in output.segments],
            [(0.2, 0.6)],
        )
        self.assertIsNone(output.probabilities)
        self.assertEqual(training_inputs["labels"], [0, 1])
        self.assertEqual(training_inputs["input_values"].shape, (2, 4))
        self.assertTrue(captured["model_saved"][1]["safe_serialization"])

    def test_multiclass_checkpoint_requires_explicit_speech_label(self):
        model = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        model.native_config = SimpleNamespace(id2label={
            0: "music",
            1: "noise",
            2: "silence",
        })

        with self.assertRaisesRegex(ValueError, "speech_class_id"):
            model._speech_class_id(3)

    def test_single_logit_and_negative_labels_resolve_the_positive_class(self):
        model = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        model.native_config = SimpleNamespace(id2label={
            0: "non-speech",
            1: "speech",
        })

        self.assertEqual(model._speech_class_id(1), 0)
        self.assertEqual(model._speech_class_id(2), 1)

    def test_task_ambiguous_and_asr_configs_are_not_silently_dispatched(self):
        with self.assertRaisesRegex(ValueError, "task-ambiguous"):
            TransformersVADForVoiceActivityDetection._infer_architecture_family(
                SimpleNamespace(
                    model_type="wav2vec2",
                    architectures=[],
                ))
        with self.assertRaisesRegex(ValueError, "ASR head"):
            TransformersVADForVoiceActivityDetection._infer_architecture_family(
                SimpleNamespace(
                    model_type="wav2vec2",
                    architectures=["Wav2Vec2ForCTC"],
                ))

    def test_frame_geometry_prefers_checkpoint_logit_stride(self):
        model = TransformersVADForVoiceActivityDetection(TransformersVADConfig())
        model.native_config = SimpleNamespace(inputs_to_logits_ratio=320)

        self.assertEqual(
            model._frame_geometry(
                frame_count=4,
                waveform_samples=16_000,
            ),
            (320, 320),
        )

    def test_training_audio_batch_cannot_be_empty(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            TransformersVADForVoiceActivityDetection._audio_batch([])


if __name__ == "__main__":
    unittest.main()
