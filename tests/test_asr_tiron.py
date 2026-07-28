import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_tiron import TironASRConfig, TironForSpeechRecognition

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


class FakeTensor:

    def __init__(self, values):
        self.values = values
        self.to_calls = []

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def tolist(self):
        return self.values


class RecordingFeatureExtractor:
    sampling_rate = 16_000

    def __init__(self):
        self.calls = []
        self.features = FakeTensor("features")

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return SimpleNamespace(input_features=self.features)


class TironTokenizer:

    def __init__(self):
        self.vocabulary = {
            "<|startoftranscript|>": 100,
            "<|en|>": 101,
            "<|zh|>": 102,
            "<|transcribe|>": 103,
            "<|endoftext|>": 104,
            "<|notimestamps|>": 200,
            "<|30.00|>": 1701,
            "<|speaker1|>": 3000,
            "<|speaker2|>": 3001,
            "<|nospeech|>": 3002,
        }
        self.reverse_vocabulary = {value: key for key, value in self.vocabulary.items()}
        self.text = {
            10: " Thanks",
            11: " everyone.",
            12: " Let's start.",
            13: " Morning!",
            14: " trailing words",
        }
        self.unk_token_id = -1
        self.eos_token_id = 104
        self.pad_token_id = 104
        self.calls = []
        self.pad_calls = []
        self.saved_to = None

    def convert_tokens_to_ids(self, token):
        return self.vocabulary.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self.reverse_vocabulary.get(token_id, f"text-{token_id}")

    def get_decoder_prompt_ids(
        self,
        *,
        language,
        task,
        no_timestamps=False,
    ):
        del no_timestamps
        if language == "english" and task == "transcribe":
            return [(1, self.vocabulary["<|en|>"])]
        return []

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(self.text.get(token_id, "") for token_id in token_ids)

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        values = [text] if isinstance(text, str) else list(text)
        return {
            "input_ids": [[3000, 201, 10, 251] for _ in values],
        }

    def pad(self, encoded, **kwargs):
        self.pad_calls.append((encoded, kwargs))
        sequences = encoded["input_ids"]
        width = max(len(row) for row in sequences)
        return {
            "input_ids": [[*row, *([self.pad_token_id] * (width - len(row)))] for row in sequences],
            "attention_mask": [[*([1] * len(row)), *([0] * (width - len(row)))] for row in sequences],
        }

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class RecordingProcessor:

    def __init__(self):
        self.feature_extractor = RecordingFeatureExtractor()
        self.tokenizer = TironTokenizer()
        self.calls = []
        self.saved_to = None

    def __call__(self, audio=None, **kwargs):
        self.calls.append((audio, kwargs))
        return {
            "input_features": "training-features",
        }

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class FakeNativeModel:

    def __init__(self, generated=None):
        self.config = SimpleNamespace(
            forced_decoder_ids=[(1, 100)],
            suppress_tokens=[1],
            begin_suppress_tokens=[2],
        )
        self.generation_config = SimpleNamespace(
            forced_decoder_ids=[(1, 100)],
            language="en",
            task="transcribe",
            suppress_tokens=[1],
            begin_suppress_tokens=[2],
            no_timestamps_token_id=200,
            no_speech_threshold=0.6,
        )
        self.device = "cpu"
        self.dtype = "float32"
        self.generated = generated or [[]]
        self.generate_calls = []
        self.training = True
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
        return FakeTensor(self.generated)

    def save_pretrained(self, directory, **kwargs):
        self.saved_to = (Path(directory), kwargs)


def _fake_transformers(*, model_type="whisper"):
    module = ModuleType("transformers")
    processor = RecordingProcessor()
    native_model = FakeNativeModel()
    native_config = SimpleNamespace(
        model_type=model_type,
        architectures=["WhisperForConditionalGeneration"],
    )

    class AutoConfig:

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            del cls, source, kwargs
            return native_config

    class AutoProcessor:

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            del cls, source, kwargs
            return processor

    class AutoModelForSpeechSeq2Seq:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return native_model

    module.AutoConfig = AutoConfig
    module.AutoProcessor = AutoProcessor
    module.AutoModelForSpeechSeq2Seq = AutoModelForSpeechSeq2Seq
    module.processor = processor
    module.native_model = native_model
    return module


def _fake_torch():
    module = ModuleType("torch")
    module.long = "long"
    module.tensor_calls = []

    def tensor(values, **kwargs):
        module.tensor_calls.append((values, kwargs))
        return FakeTensor(values)

    class InferenceMode:

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            del exc_type, exc, traceback
            return False

    module.tensor = tensor
    module.inference_mode = InferenceMode
    return module


class TironConfigurationTests(unittest.TestCase):

    def test_config_locks_whisper_sequence_to_sequence_runtime(self):
        model = TironForSpeechRecognition()

        self.assertEqual(model.config.model_type, "asr_tiron")
        self.assertEqual(model.config.architecture_family, "speech-seq2seq")
        self.assertEqual(model.config.default_language, "en")
        self.assertEqual(model.config.name_or_path, "Trelis/tiron")
        self.assertEqual(
            model.config.to_dict()["default_language"],
            "en",
        )
        with self.assertRaisesRegex(ValueError, "speech-seq2seq"):
            TironASRConfig(architecture_family="auto")
        with self.assertRaisesRegex(ValueError, "default_language"):
            TironASRConfig(default_language="")
        with self.assertRaisesRegex(ValueError, "pipeline_kwargs"):
            TironASRConfig(pipeline_kwargs={"batch_size": 2})

    def test_package_import_is_dependency_light(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_tiron; "
            "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")


class TironLoadingTests(unittest.TestCase):

    def test_loading_uses_native_whisper_and_applies_published_generation_setup(self, ):
        fake_transformers = _fake_transformers()
        model = TironForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": fake_transformers}):
            model._load_pretrained_model()

        native_model = fake_transformers.native_model
        self.assertEqual(model.architecture_family, "speech-seq2seq")
        self.assertIs(model.model, native_model)
        self.assertEqual(
            fake_transformers.AutoModelForSpeechSeq2Seq.calls[0][0],
            "Trelis/tiron",
        )
        self.assertFalse(fake_transformers.AutoModelForSpeechSeq2Seq.calls[0][1]["trust_remote_code"], )
        self.assertIsNone(native_model.config.forced_decoder_ids)
        self.assertEqual(native_model.config.suppress_tokens, [])
        self.assertEqual(native_model.config.begin_suppress_tokens, [])
        self.assertIsNone(native_model.generation_config.forced_decoder_ids)
        self.assertIsNone(native_model.generation_config.language)
        self.assertIsNone(native_model.generation_config.task)
        self.assertIsNone(native_model.generation_config.suppress_tokens)
        self.assertIsNone(native_model.generation_config.begin_suppress_tokens, )
        self.assertIsNone(native_model.generation_config.no_speech_threshold, )
        self.assertFalse(hasattr(
            native_model.generation_config,
            "no_timestamps_token_id",
        ))

    def test_loading_rejects_non_whisper_checkpoints(self):
        fake_transformers = _fake_transformers(model_type="moonshine")
        model = TironForSpeechRecognition(
            TironASRConfig(name_or_path="publisher/not-tiron"),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            with self.assertRaisesRegex(ValueError, "native Whisper"):
                model._load_pretrained_model()


class TironInferenceTests(unittest.TestCase):

    @staticmethod
    def _loaded_model(generated):
        model = TironForSpeechRecognition(device="cpu")
        model.model = FakeNativeModel(generated=generated)
        model.transformers_processor = RecordingProcessor()
        model.architecture_family = "speech-seq2seq"
        return model

    def test_direct_generation_preserves_speakers_timestamps_and_all_text(self):
        timestamp_begin = 201
        generated = [[
            100,
            101,
            103,
            3000,
            timestamp_begin,
            10,
            11,
            timestamp_begin + 148,
            timestamp_begin + 176,
            12,
            timestamp_begin + 240,
            3001,
            timestamp_begin + 149,
            13,
            timestamp_begin + 170,
            104,
        ]]
        model = self._loaded_model(generated)
        fake_torch = _fake_torch()

        with _temporary_modules({"torch": fake_torch}):
            output = model._transcribe(
                np.zeros(8_000, dtype=np.float32),
                sampling_rate=16_000,
                return_timestamps=True,
            )

        self.assertIsInstance(output, ASROutput)
        self.assertEqual(
            output.text,
            "Thanks everyone. Let's start. Morning!",
        )
        self.assertEqual(
            [(
                segment.speaker,
                segment.start,
                segment.end,
                segment.text,
            ) for segment in output.segments],
            [
                ("SPEAKER_00", 0.0, 2.96, "Thanks everyone."),
                ("SPEAKER_00", 3.52, 4.8, "Let's start."),
                ("SPEAKER_01", 2.98, 3.4, "Morning!"),
            ],
        )
        self.assertEqual(
            output.segments[0].metadata["local_speaker_index"],
            1,
        )
        self.assertEqual(output.metadata["backend"], "tiron")
        self.assertTrue(output.metadata["native_segment_timestamps"])

        generation = model.model.generate_calls[0]
        self.assertEqual(generation["max_new_tokens"], 444)
        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 1)
        self.assertEqual(
            generation["decoder_input_ids"].tolist(),
            [[100, 101, 103]],
        )
        feature_call = model.transformers_processor.feature_extractor.calls[0]
        self.assertEqual(feature_call[1]["sampling_rate"], 16_000)
        self.assertEqual(feature_call[1]["return_tensors"], "pt")
        self.assertEqual(
            fake_torch.tensor_calls[0][1],
            {
                "device": "cpu",
                "dtype": "long",
            },
        )

    def test_text_without_timestamps_is_not_dropped(self):
        model = self._loaded_model([[
            100,
            101,
            103,
            3000,
            10,
            11,
            14,
            104,
        ]])
        fake_torch = _fake_torch()

        with _temporary_modules({"torch": fake_torch}):
            output = model._transcribe(
                np.zeros(800, dtype=np.float32),
                sampling_rate=16_000,
                language="english",
            )

        self.assertEqual(
            output.text,
            "Thanks everyone. trailing words",
        )
        self.assertEqual(len(output.segments), 1)
        self.assertIsNone(output.segments[0].start)
        self.assertIsNone(output.segments[0].end)
        self.assertEqual(output.language, "english")

    def test_native_nospeech_token_produces_an_empty_transcript(self):
        model = self._loaded_model([[
            100,
            101,
            103,
            3002,
            104,
        ]])
        fake_torch = _fake_torch()

        with _temporary_modules({"torch": fake_torch}):
            output = model._transcribe(
                np.zeros(800, dtype=np.float32),
                sampling_rate=16_000,
            )

        self.assertEqual(output.text, "")
        self.assertEqual(output.segments, ())

    def test_invalid_controls_fail_before_generation(self):
        model = self._loaded_model([[]])
        cases = (
            ({
                "task": "translate",
            }, "translation"),
            ({
                "return_timestamps": "word",
            }, "word-level"),
            ({
                "chunk_length_s": 15.0,
            }, "meeting harness"),
            ({
                "batch_size": 2,
            }, "one audio window"),
            ({
                "num_beams": 2,
            }, "num_beams=1"),
            ({
                "hotwords": ("VoiceHub", ),
            }, "hotword"),
            ({
                "language": "auto",
            }, "explicit"),
        )
        fake_torch = _fake_torch()

        with _temporary_modules({"torch": fake_torch}):
            for kwargs, message in cases:
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        model._transcribe(
                            np.zeros(800, dtype=np.float32),
                            sampling_rate=16_000,
                            **kwargs,
                        )

    def test_audio_longer_than_native_window_is_rejected(self):
        model = self._loaded_model([[]])
        fake_torch = _fake_torch()

        with _temporary_modules({"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "at most 30 seconds"):
                model._transcribe(
                    np.zeros(480_100, dtype=np.float32),
                    sampling_rate=16_000,
                )


class TironTrainingAndExportTests(unittest.TestCase):

    def test_training_uses_checkpoint_tokenizer_without_stripping_tiron_tokens(self, ):
        model = TironForSpeechRecognition(device="cpu")
        processor = RecordingProcessor()
        model.transformers_processor = processor
        transcript = "<|speaker1|><|0.00|> hello<|1.00|>"

        batch = model.prepare_training_inputs(
            {
                "audio": np.zeros(800, dtype=np.float32),
                "sampling_rate": 16_000,
                "text": transcript,
            },
            phase="asr",
        )

        self.assertEqual(batch["input_features"], "training-features")
        self.assertEqual(
            processor.calls[0][1],
            {
                "sampling_rate": 16_000,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(
            batch["labels"],
            [[101, 103, 3000, 201, 10, 251, 104]],
        )
        self.assertNotIn(200, batch["labels"][0])
        self.assertEqual(processor.tokenizer.calls[0][0], [transcript])
        self.assertEqual(
            processor.tokenizer.calls[0][1],
            {
                "add_special_tokens": False,
                "padding": False,
            },
        )
        self.assertEqual(
            processor.tokenizer.pad_calls[0],
            (
                {
                    "input_ids": [[
                        101,
                        103,
                        3000,
                        201,
                        10,
                        251,
                        104,
                    ]],
                },
                {
                    "padding": True,
                    "return_attention_mask": True,
                    "return_tensors": "pt",
                },
            ),
        )

    def test_training_accepts_safetensors_and_rejects_serving_artifacts(self):
        TironForSpeechRecognition(
            TironASRConfig(name_or_path="publisher/tiron.safetensors"), )._validate_training_runtime()

        with self.assertRaisesRegex(ValueError, "inference-only"):
            TironForSpeechRecognition(
                TironASRConfig(name_or_path="publisher/tiron.gguf"), )._validate_training_runtime()

    def test_native_export_saves_safe_weights_and_checkpoint_processor(self):
        model = TironForSpeechRecognition(device="cpu")
        model.model = FakeNativeModel()
        model.transformers_processor = RecordingProcessor()

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "native"
            model._save_pretrained(destination)

        self.assertEqual(
            model.model.saved_to,
            (destination, {
                "safe_serialization": True,
            }),
        )
        self.assertEqual(
            model.transformers_processor.saved_to,
            destination,
        )


if __name__ == "__main__":
    unittest.main()
