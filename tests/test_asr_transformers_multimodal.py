import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from voicehub.models.asr_transformers_multimodal import (
    Qwen3ASRConfig,
    Qwen3ASRForSpeechRecognition,
    VibeVoiceASRConfig,
    VibeVoiceASRForSpeechRecognition,
)
from voicehub.trainer_utils import NATIVE_EXPORT_DIR

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


class FakeBatchFeature(dict):

    def __init__(self, **values):
        super().__init__(values)
        self.to_calls = []

    def to(self, *args):
        self.to_calls.append(args)
        return self


class FakeQwenTokenizer:

    def __call__(self, text, **kwargs):
        self.last_call = (text, kwargs)
        return {
            "input_ids": [[12, 13]],
        }


class FakeProcessor:

    def __init__(self, *, sampling_rate, decoded):
        self.feature_extractor = SimpleNamespace(sampling_rate=sampling_rate, )
        self.decoded = decoded
        self.request_calls = []
        self.template_calls = []
        self.decode_calls = []
        self.request_output = None
        self.template_output = None
        self.saved_to = None
        self.tokenizer = FakeQwenTokenizer()

    def apply_transcription_request(self, **kwargs):
        self.request_calls.append(kwargs)
        self.request_output = FakeBatchFeature(
            input_ids=np.asarray([[10, 11]], dtype=np.int64),
            input_features="native-audio-features",
            processor_state="must-be-preserved",
        )
        return self.request_output

    def apply_chat_template(self, conversations, **kwargs):
        self.template_calls.append((conversations, kwargs))
        if kwargs.get("add_generation_prompt") or kwargs.get("continue_final_message"):
            self.template_output = FakeBatchFeature(
                input_ids=np.asarray([[10, 11]], dtype=np.int64),
                input_features="native-audio-features",
                processor_state="must-be-preserved",
            )
        else:
            if kwargs.get("output_labels"):
                self.template_output = FakeBatchFeature(
                    input_ids="native-input-ids",
                    attention_mask="native-attention-mask",
                    input_features="native-input-features",
                    input_features_mask="native-feature-mask",
                    labels="native-labels",
                    processor_state="must-be-preserved",
                )
            else:
                self.template_output = FakeBatchFeature(
                    input_ids=np.asarray(
                        [[10, 11, 12, 13, 90, 91]],
                        dtype=np.int64,
                    ),
                    attention_mask=np.asarray(
                        [[1, 1, 1, 1, 1, 1]],
                        dtype=np.int64,
                    ),
                    input_features="native-input-features",
                    input_features_mask="native-feature-mask",
                    processor_state="must-be-preserved",
                )
        return self.template_output

    def decode(self, tokens, **kwargs):
        self.decode_calls.append((tokens, kwargs))
        return self.decoded

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class FakeNativeModel:

    def __init__(self):
        self.device = "cpu"
        self.dtype = "float32"
        self.hf_device_map = None
        self.training = True
        self.generate_calls = []
        self.forward_calls = []
        self.saved_to = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return np.asarray([[10, 11, 90, 91]], dtype=np.int64)

    def __call__(self, **kwargs):
        self.forward_calls.append(kwargs)
        return SimpleNamespace(loss="native-loss")

    def save_pretrained(self, directory, **kwargs):
        self.saved_to = (Path(directory), kwargs)


def _loader(model=None, *, error=None):

    class Loader:
        calls = []
        loaded_model = model or FakeNativeModel()

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            if error is not None:
                raise error
            return cls.loaded_model

    return Loader


def _fake_transformers(
    *,
    model_type,
    processor,
    multimodal_loader=None,
    seq2seq_loader=None,
    include_multimodal=True,
):
    module = ModuleType("transformers")
    native_config = SimpleNamespace(
        model_type=model_type,
        architectures=["NativeForConditionalGeneration"],
    )

    class AutoConfig:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return native_config

    class AutoProcessor:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return processor

    module.AutoConfig = AutoConfig
    module.AutoProcessor = AutoProcessor
    if include_multimodal:
        module.AutoModelForMultimodalLM = (multimodal_loader or _loader())
    module.AutoModelForSpeechSeq2Seq = seq2seq_loader or _loader()
    return module


class MultimodalASRConfigTests(unittest.TestCase):

    def test_package_import_is_dependency_free(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_transformers_multimodal; "
            "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")

    def test_family_configs_publish_current_native_defaults(self):
        qwen = Qwen3ASRConfig()
        vibe = VibeVoiceASRConfig()

        self.assertEqual(qwen.model_type, "asr_qwen3")
        self.assertEqual(qwen.architecture_family, "speech-seq2seq")
        self.assertEqual(qwen.training_language, "English")
        self.assertEqual(vibe.model_type, "asr_vibevoice")
        self.assertEqual(vibe.sample_rate, 24_000)
        with self.assertRaisesRegex(ValueError, "speech-seq2seq"):
            Qwen3ASRConfig(architecture_family="ctc")
        with self.assertRaisesRegex(ValueError, "language-conditioning"):
            VibeVoiceASRConfig(training_language="English")


class MultimodalASRInferenceTests(unittest.TestCase):

    def test_qwen_routes_the_full_processor_request_and_slices_prompt(self):
        processor = FakeProcessor(
            sampling_rate=16_000,
            decoded=[{
                "language": "English",
                "transcription": "hello VoiceHub",
            }],
        )
        loader = _loader()
        transformers = _fake_transformers(
            model_type="qwen3_asr",
            processor=processor,
            multimodal_loader=loader,
        )
        model = Qwen3ASRForSpeechRecognition(
            Qwen3ASRConfig(name_or_path="Qwen/Qwen3-ASR-0.6B-hf"),
            device="cpu",
        )

        with _temporary_modules({"transformers": transformers}):
            output = model.transcribe(
                np.zeros(320, dtype=np.float32),
                sampling_rate=16_000,
                language="en",
                hotwords=["VoiceHub", "Codex"],
                max_new_tokens=32,
                processor_kwargs={
                    "audio_kwargs": {
                        "padding": True,
                    },
                },
                temperature=0.2,
            )

        conversations, template_options = processor.template_calls[0]
        request = conversations[0]
        self.assertEqual(
            request[0]["content"][0]["text"],
            "Vocabulary: VoiceHub, Codex",
        )
        self.assertIsInstance(
            request[1]["content"][0]["audio"],
            np.ndarray,
        )
        self.assertEqual(
            request[2]["content"][0]["text"],
            "language English<asr_text>",
        )
        self.assertEqual(
            template_options,
            {
                "tokenize": True,
                "return_dict": True,
                "processor_kwargs": {
                    "audio_kwargs": {
                        "padding": True,
                    },
                },
                "continue_final_message": True,
            },
        )
        generation = loader.loaded_model.generate_calls[0]
        self.assertEqual(
            generation["processor_state"],
            "must-be-preserved",
        )
        self.assertEqual(generation["max_new_tokens"], 32)
        self.assertEqual(generation["temperature"], 0.2)
        np.testing.assert_array_equal(
            processor.decode_calls[0][0],
            np.asarray([[90, 91]], dtype=np.int64),
        )
        self.assertEqual(
            processor.decode_calls[0][1],
            {
                "return_format": "parsed",
            },
        )
        self.assertEqual(output.text, "hello VoiceHub")
        self.assertEqual(output.language, "English")

    def test_qwen_rejects_an_unsupported_language_before_rendering(self):
        model = Qwen3ASRForSpeechRecognition(device="cpu")
        model.transformers_processor = FakeProcessor(
            sampling_rate=16_000,
            decoded=[],
        )

        with self.assertRaisesRegex(ValueError, "Unsupported Qwen3-ASR language"):
            model._apply_transcription_request(
                waveform=np.zeros(160, dtype=np.float32),
                language="Klingon",
                prompt=None,
                processor_kwargs={},
            )

    def test_vibe_normalizes_speakers_and_timestamps(self):
        processor = FakeProcessor(
            sampling_rate=24_000,
            decoded=[[
                {
                    "Start": 0,
                    "End": 1.25,
                    "Speaker": 0,
                    "Content": "hello",
                },
                {
                    "Start": 1.25,
                    "End": 2.5,
                    "Speaker": 1,
                    "Content": "world",
                },
            ]],
        )
        loader = _loader()
        transformers = _fake_transformers(
            model_type="vibevoice_asr",
            processor=processor,
            multimodal_loader=loader,
        )
        model = VibeVoiceASRForSpeechRecognition(
            VibeVoiceASRConfig(name_or_path="microsoft/VibeVoice-ASR-HF", ),
            device="cpu",
        )

        with _temporary_modules({"transformers": transformers}):
            output = model.transcribe(
                np.zeros(2_400, dtype=np.float32),
                sampling_rate=24_000,
                prompt="Project context",
                return_timestamps=True,
            )

        self.assertEqual(
            processor.request_calls[0]["prompt"],
            "Project context",
        )
        self.assertNotIn("language", processor.request_calls[0])
        self.assertEqual(output.text, "hello world")
        self.assertEqual(len(output.segments), 2)
        self.assertEqual(output.segments[0].speaker, "0")
        self.assertEqual(output.segments[1].start, 1.25)
        self.assertEqual(output.segments[1].end, 2.5)
        self.assertTrue(output.metadata["structured_output"])

    def test_vibe_rejects_word_timestamps_without_an_aligner(self):
        model = VibeVoiceASRForSpeechRecognition(device="cpu")

        with self.assertRaisesRegex(
                ValueError,
                "speaker-segment timestamps",
        ):
            model._transcribe(
                np.zeros(240, dtype=np.float32),
                sampling_rate=24_000,
                return_timestamps="word",
            )

    def test_vibe_rejects_language_forcing(self):
        model = VibeVoiceASRForSpeechRecognition(device="cpu")

        with self.assertRaisesRegex(ValueError, "does not expose language forcing"):
            model._request_options(
                language="en",
                prompt=None,
                processor_kwargs={},
            )

    def test_loader_falls_back_to_speech_seq2seq_on_auto_mismatch(self):
        processor = FakeProcessor(
            sampling_rate=16_000,
            decoded=[{
                "language": None,
                "transcription": "fallback",
            }],
        )
        primary = _loader(
            error=ValueError("configuration class is not supported for this kind of "
                             "AutoModel"), )
        fallback = _loader()
        transformers = _fake_transformers(
            model_type="qwen3_asr",
            processor=processor,
            multimodal_loader=primary,
            seq2seq_loader=fallback,
        )
        model = Qwen3ASRForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": transformers}):
            model.load()

        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(fallback.calls), 1)
        self.assertIs(model.model, fallback.loaded_model)


class MultimodalASRTrainingTests(unittest.TestCase):

    def test_qwen_uses_assistant_target_and_nested_output_labels(self):
        processor = FakeProcessor(
            sampling_rate=16_000,
            decoded=[],
        )
        loader = _loader()
        transformers = _fake_transformers(
            model_type="qwen3_asr",
            processor=processor,
            multimodal_loader=loader,
        )
        model = Qwen3ASRForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": transformers}):
            model.load_for_training()
            batch = model.prepare_training_inputs(
                {
                    "audio": np.zeros(160, dtype=np.float32),
                    "sampling_rate": 16_000,
                    "text": "Merhaba dünya",
                    "language": "Turkish",
                },
                phase="main",
            )
            native_output = model.model(**batch)

        conversations, options = processor.template_calls[0]
        self.assertEqual(
            conversations[0][1]["content"][0]["text"],
            "language Turkish<asr_text>Merhaba dünya",
        )
        self.assertIsInstance(
            conversations[0][0]["content"][0]["audio"],
            np.ndarray,
        )
        self.assertEqual(
            options,
            {
                "tokenize": True,
                "return_dict": True,
            },
        )
        np.testing.assert_array_equal(
            batch["labels"],
            np.asarray([[-100, -100, -100, -100, 90, 91]]),
        )
        self.assertEqual(
            processor.tokenizer.last_call,
            (
                "<|im_start|>assistant\n",
                {
                    "add_special_tokens": False,
                },
            ),
        )
        self.assertEqual(
            batch["processor_state"],
            "must-be-preserved",
        )
        self.assertEqual(native_output.loss, "native-loss")
        self.assertEqual(
            loader.loaded_model.forward_calls[0],
            batch,
        )

    def test_qwen_rebuilds_invalid_cached_processor_labels(self):
        processor = FakeProcessor(
            sampling_rate=16_000,
            decoded=[],
        )
        model = Qwen3ASRForSpeechRecognition(device="cpu")
        model.transformers_processor = processor

        batch = model.prepare_training_inputs(
            {
                "input_ids": np.asarray(
                    [[10, 11, 12, 13, 90, 91]],
                    dtype=np.int64,
                ),
                "attention_mask": np.ones((1, 6), dtype=np.int64),
                "labels": np.asarray(
                    [[0, 0, 3, 3, 3, 3]],
                    dtype=np.int64,
                ),
            },
            phase="main",
        )

        np.testing.assert_array_equal(
            batch["labels"],
            np.asarray([[-100, -100, -100, -100, 90, 91]]),
        )

    def test_vibe_uses_user_text_target_and_direct_output_labels(self):
        processor = FakeProcessor(
            sampling_rate=24_000,
            decoded=[],
        )
        loader = _loader()
        transformers = _fake_transformers(
            model_type="vibevoice_asr",
            processor=processor,
            multimodal_loader=loader,
        )
        model = VibeVoiceASRForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": transformers}):
            model.load_for_training()
            batch = model.prepare_training_inputs(
                {
                    "audio": [
                        np.zeros(240, dtype=np.float32),
                        np.zeros(480, dtype=np.float32),
                    ],
                    "sampling_rate": 24_000,
                    "transcription": ["speaker one", "speaker two"],
                },
                phase="main",
            )

        conversations, options = processor.template_calls[0]
        self.assertEqual(
            conversations[0][0]["content"][0],
            {
                "type": "text",
                "text": "speaker one",
            },
        )
        self.assertEqual(
            conversations[1][0]["content"][0]["text"],
            "speaker two",
        )
        self.assertEqual(
            options,
            {
                "tokenize": True,
                "return_dict": True,
                "output_labels": True,
            },
        )
        self.assertEqual(batch["labels"], "native-labels")
        self.assertEqual(
            batch["input_features_mask"],
            "native-feature-mask",
        )

    def test_vibe_training_rejects_ignored_language_metadata(self):
        model = VibeVoiceASRForSpeechRecognition(device="cpu")

        with self.assertRaisesRegex(ValueError, "does not expose language"):
            model._training_conversation(
                waveform=np.zeros(240, dtype=np.float32),
                transcription="hello",
                language="English",
            )

    def test_training_rejects_serving_and_quantized_checkpoints(self):
        serving = Qwen3ASRForSpeechRecognition(
            Qwen3ASRConfig(name_or_path="publisher/model.gguf"),
            device="cpu",
        )
        quantized = VibeVoiceASRForSpeechRecognition(
            VibeVoiceASRConfig(model_kwargs={
                "load_in_4bit": True,
            }, ),
            device="cpu",
        )

        with self.assertRaisesRegex(ValueError, "inference-only"):
            serving.load_for_training()
        with self.assertRaisesRegex(ValueError, "unquantized"):
            quantized.load_for_training()

    def test_native_export_saves_safetensors_and_processor(self):
        processor = FakeProcessor(
            sampling_rate=16_000,
            decoded=[],
        )
        loader = _loader()
        transformers = _fake_transformers(
            model_type="qwen3_asr",
            processor=processor,
            multimodal_loader=loader,
        )
        model = Qwen3ASRForSpeechRecognition(device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            with _temporary_modules({"transformers": transformers}):
                model.load()
                model.save_pretrained(directory)
            native_directory = Path(directory) / NATIVE_EXPORT_DIR

        self.assertEqual(
            loader.loaded_model.saved_to,
            (
                native_directory,
                {
                    "safe_serialization": True,
                },
            ),
        )
        self.assertEqual(processor.saved_to, native_directory)


if __name__ == "__main__":
    unittest.main()
