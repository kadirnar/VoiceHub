import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from voicehub.errors import OptionalDependencyError
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_transformers import TransformersASRConfig, TransformersASRForSpeechRecognition

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


class FakeNativeModel:

    def __init__(self):
        self.device = None
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

    def save_pretrained(self, directory, **kwargs):
        self.saved_to = (Path(directory), kwargs)


class FakeTokenizer:

    def __init__(self):
        self.calls = []
        self.target_languages = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {
            "input_ids": [[4, 5]],
            "attention_mask": [[1, 1]],
        }

    def set_target_lang(self, language):
        self.target_languages.append(language)


class FakeProcessor:

    def __init__(self):
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.saved_to = None

    def __call__(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return {
            "input_values": "processed-audio",
        }

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


def _loader(*, error=None):

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            if error is not None:
                raise error
            return FakeNativeModel()

    return Loader


def _fake_transformers(
    *,
    native_config=None,
    ctc_loader=None,
    seq2seq_loader=None,
    rnnt_loader=None,
    tdt_loader=None,
    include_rnnt=True,
    include_tdt=True,
):
    module = ModuleType("transformers")
    config = native_config or SimpleNamespace(
        model_type="whisper",
        architectures=["WhisperForConditionalGeneration"],
    )

    class AutoConfig:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return config

    class AutoProcessor:
        calls = []
        processor = FakeProcessor()

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return cls.processor

    module.AutoConfig = AutoConfig
    module.AutoProcessor = AutoProcessor
    module.AutoModelForCTC = ctc_loader or _loader()
    module.AutoModelForSpeechSeq2Seq = seq2seq_loader or _loader()
    if include_rnnt:
        module.AutoModelForRNNT = rnnt_loader or _loader()
    if include_tdt:
        module.AutoModelForTDT = tdt_loader or _loader()
    module.pipeline_calls = []

    def pipeline(**kwargs):
        module.pipeline_calls.append(kwargs)
        return lambda audio, **options: {
            "text": "pipeline result",
            "options": options,
        }

    module.pipeline = pipeline
    return module


class TransformersASRConfigTests(unittest.TestCase):

    def test_configuration_is_dependency_free_and_validated(self):
        config = TransformersASRConfig(
            architecture_family="RNNT",
            trust_remote_code=False,
            model_kwargs={
                "low_cpu_mem_usage": True,
            },
        )

        self.assertEqual(config.model_type, "asr_transformers")
        self.assertEqual(config.architecture_family, "rnnt")
        self.assertFalse(config.trust_remote_code)
        with self.assertRaisesRegex(ValueError, "architecture_family"):
            TransformersASRConfig(architecture_family="decoder-only")
        with self.assertRaisesRegex(ValueError, "provider-owned"):
            TransformersASRConfig(model_kwargs={
                "trust_remote_code": True,
            })
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TransformersASRConfig(token="must-not-be-serialized")
        with self.assertRaisesRegex(ValueError, "runtime secrets"):
            TransformersASRConfig(pipeline_kwargs={
                "nested": {
                    "api_key": "must-not-be-serialized",
                },
            })
        with self.assertRaisesRegex(ValueError, "dedicated provider"):
            TransformersASRConfig(architecture_family="audio-text-to-text")

    def test_import_does_not_load_transformers_or_torch(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_transformers; "
            "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")


class TransformersASRLoadingTests(unittest.TestCase):

    def test_known_checkpoint_dispatches_and_keeps_remote_code_disabled(self):
        seq2seq_loader = _loader()
        fake_transformers = _fake_transformers(seq2seq_loader=seq2seq_loader)
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(
                name_or_path="publisher/whisper",
                architecture_family="auto",
                revision="release",
                local_files_only=True,
            ),
            device="cpu",
            token="private-token",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            model._load_pretrained_model()

        self.assertIsInstance(model.model, FakeNativeModel)
        self.assertIs(model.training_processor, model.transformers_processor)
        self.assertEqual(model.architecture_family, "speech-seq2seq")
        self.assertEqual(model.model.device, "cpu")
        config_call = fake_transformers.AutoConfig.calls[0]
        processor_call = fake_transformers.AutoProcessor.calls[0]
        model_call = seq2seq_loader.calls[0]
        self.assertFalse(config_call[1]["trust_remote_code"])
        self.assertFalse(processor_call[1]["trust_remote_code"])
        self.assertFalse(model_call[1]["trust_remote_code"])
        self.assertEqual(model_call[1]["config"].model_type, "whisper")
        self.assertEqual(model_call[1]["revision"], "release")
        self.assertTrue(model_call[1]["local_files_only"])
        self.assertEqual(model_call[1]["token"], "private-token")
        self.assertNotIn("token", model.config.to_dict())
        self.assertIsNone(model._pipeline)

    def test_explicit_rnnt_uses_feature_detected_auto_class(self):
        rnnt_loader = _loader()
        fake_transformers = _fake_transformers(
            native_config=SimpleNamespace(
                model_type="future-transducer",
                architectures=[],
            ),
            rnnt_loader=rnnt_loader,
        )
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(
                name_or_path="publisher/transducer",
                architecture_family="rnnt",
            ),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            model._load_pretrained_model()

        self.assertEqual(model.architecture_family, "rnnt")
        self.assertEqual(len(rnnt_loader.calls), 1)

    def test_missing_new_auto_class_has_actionable_error(self):
        fake_transformers = _fake_transformers(include_tdt=False)
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(
                name_or_path="publisher/tdt",
                architecture_family="tdt",
            ),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            with self.assertRaisesRegex(
                    OptionalDependencyError,
                    "AutoModelForTDT",
            ):
                model._load_pretrained_model()

    def test_unknown_future_config_probes_public_auto_classes(self):
        mismatch = ValueError(
            "Unrecognized configuration class FutureASRConfig for this kind "
            "of AutoModel: AutoModelForCTC.")
        ctc_loader = _loader(error=mismatch)
        seq2seq_loader = _loader()
        fake_transformers = _fake_transformers(
            native_config=SimpleNamespace(
                model_type="future-asr",
                architectures=[],
                is_encoder_decoder=False,
            ),
            ctc_loader=ctc_loader,
            seq2seq_loader=seq2seq_loader,
        )
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/future-asr"),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            model._load_pretrained_model()

        self.assertEqual(model.architecture_family, "speech-seq2seq")
        self.assertEqual(len(ctc_loader.calls), 1)
        self.assertEqual(len(seq2seq_loader.calls), 1)

    def test_direct_safetensors_uses_sibling_layout_and_explicit_state(self):
        seq2seq_loader = _loader()
        fake_transformers = _fake_transformers(seq2seq_loader=seq2seq_loader)
        fake_safetensors_package = ModuleType("safetensors")
        fake_safetensors = ModuleType("safetensors.torch")
        state_dict = {
            "encoder.weight": object(),
        }
        fake_safetensors.load_file = (lambda path, device: state_dict)
        with tempfile.TemporaryDirectory() as directory:
            weight_file = Path(directory) / "fine-tuned.safetensors"
            weight_file.touch()
            model = TransformersASRForSpeechRecognition(
                TransformersASRConfig(
                    name_or_path=weight_file,
                    config_name_or_path="publisher/base",
                    processor_name_or_path="publisher/base",
                ),
                device="cpu",
            )

            with _temporary_modules({
                    "transformers": fake_transformers,
                    "safetensors": fake_safetensors_package,
                    "safetensors.torch": fake_safetensors,
            }):
                model._load_pretrained_model()

        source, kwargs = seq2seq_loader.calls[0]
        self.assertEqual(source, str(weight_file.parent.resolve()))
        self.assertIs(kwargs["state_dict"], state_dict)
        self.assertEqual(
            fake_transformers.AutoConfig.calls[0][0],
            "publisher/base",
        )

    def test_direct_bin_checkpoint_is_loaded_as_weights_only(self):
        seq2seq_loader = _loader()
        fake_transformers = _fake_transformers(seq2seq_loader=seq2seq_loader)
        fake_torch = ModuleType("torch")
        state_dict = {
            "encoder.weight": object(),
        }
        load_calls = []

        def load(path, **kwargs):
            load_calls.append((path, kwargs))
            return state_dict

        fake_torch.load = load
        with tempfile.TemporaryDirectory() as directory:
            weight_file = Path(directory) / "fine-tuned.bin"
            weight_file.touch()
            model = TransformersASRForSpeechRecognition(
                TransformersASRConfig(
                    name_or_path=weight_file,
                    config_name_or_path="publisher/base",
                    processor_name_or_path="publisher/base",
                ),
                device="cpu",
            )

            with _temporary_modules({
                    "transformers": fake_transformers,
                    "torch": fake_torch,
            }):
                model._load_pretrained_model()

        self.assertIs(
            seq2seq_loader.calls[0][1]["state_dict"],
            state_dict,
        )
        self.assertEqual(load_calls[0][0], str(weight_file.resolve()))
        self.assertEqual(
            load_calls[0][1],
            {
                "map_location": "cpu",
                "weights_only": True,
            },
        )

    def test_task_incompatible_heads_are_not_remapped_by_base_model_type(self):
        native_config = SimpleNamespace(
            model_type="wav2vec2",
            architectures=["Wav2Vec2ForAudioFrameClassification"],
        )
        fake_transformers = _fake_transformers(native_config=native_config)
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/vad"),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            with self.assertRaisesRegex(ValueError, "not an ASR head"):
                model._load_pretrained_model()

    def test_audio_text_to_text_auto_map_has_an_actionable_boundary(self):
        native_config = SimpleNamespace(
            model_type="future-audio-llm",
            architectures=[],
            auto_map={
                "AutoModelForAudioTextToText": "model.FutureAudioModel",
            },
        )
        fake_transformers = _fake_transformers(native_config=native_config)
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/audio-llm"),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            with self.assertRaisesRegex(ValueError, "chat-template"):
                model._load_pretrained_model()

    def test_training_load_exposes_native_model_without_a_pipeline(self):
        seq2seq_loader = _loader()
        fake_transformers = _fake_transformers(seq2seq_loader=seq2seq_loader)
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/whisper"),
            device="cpu",
        )

        with _temporary_modules({"transformers": fake_transformers}):
            model.load_for_training()

        self.assertIsInstance(model.model, FakeNativeModel)
        self.assertTrue(model.model.training)
        self.assertIsNone(model._pipeline)


class TransformersASRInferenceTests(unittest.TestCase):

    def _loaded_model(self, *, family="speech-seq2seq"):
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/asr"),
            device="cpu",
        )
        model.model = FakeNativeModel()
        model.transformers_processor = FakeProcessor()
        model.architecture_family = family
        return model

    def test_word_timestamps_are_normalized_to_public_output(self):
        model = self._loaded_model()
        pipeline_calls = []

        def pipeline(audio, **kwargs):
            pipeline_calls.append((audio, kwargs))
            return {
                "text":
                "hello world",
                "language":
                "en",
                "chunks": [
                    {
                        "text": "hello",
                        "timestamp": (0.0, 0.4),
                        "score": 0.8,
                    },
                    {
                        "word": "world",
                        "timestamp": [0.5, 1.0],
                        "confidence": 0.6,
                    },
                ],
                "model_revision":
                "test",
            }

        model._ensure_pipeline = lambda: pipeline
        output = model._transcribe(
            np.asarray([0.0, 0.1], dtype=np.float32),
            sampling_rate=16_000,
            language="en",
            task="translate",
            return_timestamps="word",
            num_beams=3,
            max_new_tokens=32,
        )

        self.assertIsInstance(output, ASROutput)
        self.assertEqual(output.text, "hello world")
        self.assertEqual(output.language, "en")
        self.assertEqual(len(output.segments), 1)
        self.assertEqual(
            [word.text for word in output.segments[0].words],
            ["hello", "world"],
        )
        self.assertAlmostEqual(output.segments[0].confidence, 0.7)
        self.assertEqual(output.metadata["backend"], "transformers")
        self.assertEqual(output.metadata["model_revision"], "test")
        pipeline_input, options = pipeline_calls[0]
        self.assertEqual(pipeline_input["sampling_rate"], 16_000)
        self.assertEqual(
            options["generate_kwargs"],
            {
                "language": "en",
                "task": "translate",
                "num_beams": 3,
                "max_new_tokens": 32,
            },
        )
        self.assertEqual(options["return_timestamps"], "word")

    def test_segment_chunks_preserve_nested_words_and_metadata(self):
        model = self._loaded_model()
        output = model._normalize_pipeline_output(
            {
                "text":
                "hello",
                "chunks": [{
                    "text": "hello",
                    "timestamp": {
                        "start": 0.25,
                        "end": 0.75,
                    },
                    "speaker": "speaker-1",
                    "words": [{
                        "word": "hello",
                        "timestamp": (0.25, 0.75),
                    }],
                    "temperature": 0.0,
                }],
            },
            duration=1.0,
            timestamp_mode=True,
        )

        segment = output.segments[0]
        self.assertEqual((segment.start, segment.end), (0.25, 0.75))
        self.assertEqual(segment.speaker, "speaker-1")
        self.assertEqual(segment.words[0].text, "hello")
        self.assertEqual(segment.metadata["temperature"], 0.0)

    def test_requested_language_is_preserved_when_pipeline_omits_it(self):
        model = self._loaded_model()

        output = model._normalize_pipeline_output(
            {
                "text": "merhaba",
            },
            duration=0.5,
            timestamp_mode=False,
            fallback_language="tr",
        )

        self.assertEqual(output.language, "tr")

    def test_ctc_controls_reject_generation_and_support_language_adapters(self):
        model = self._loaded_model(family="ctc")

        with self.assertRaisesRegex(ValueError, "not CTC"):
            model._pipeline_call_options(
                language=None,
                task="transcribe",
                return_timestamps=False,
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=2,
                max_new_tokens=None,
                hotwords=None,
                options={},
            )

        model._pipeline_call_options(
            language="eng",
            task="transcribe",
            return_timestamps=False,
            chunk_length_s=None,
            stride_length_s=None,
            batch_size=None,
            num_beams=None,
            max_new_tokens=None,
            hotwords=("VoiceHub", ),
            options={},
        )
        self.assertEqual(
            model.transformers_processor.tokenizer.target_languages,
            ["eng"],
        )

    def test_pipeline_is_built_lazily_against_current_native_model(self):
        model = self._loaded_model()
        fake_transformers = _fake_transformers()

        with _temporary_modules({"transformers": fake_transformers}):
            first = model._ensure_pipeline()
            second = model._ensure_pipeline()

        self.assertIs(first, second)
        self.assertEqual(len(fake_transformers.pipeline_calls), 1)
        pipeline_kwargs = fake_transformers.pipeline_calls[0]
        self.assertIs(pipeline_kwargs["model"], model.model)
        self.assertEqual(
            pipeline_kwargs["task"],
            "automatic-speech-recognition",
        )
        self.assertEqual(pipeline_kwargs["device"], "cpu")
        self.assertNotIn("processor", pipeline_kwargs)

    def test_opaque_callable_signatures_do_not_break_feature_detection(self):

        for exception_type in (TypeError, ValueError):

            class OpaqueCallable:

                @property
                def __signature__(self):
                    raise exception_type("signature unavailable")

                def __call__(self, **kwargs):
                    return kwargs

            with self.subTest(exception_type=exception_type):
                self.assertFalse(
                    TransformersASRForSpeechRecognition._accepts_keyword(
                        OpaqueCallable(),
                        "processor",
                    ))


class TransformersASRTrainingTests(unittest.TestCase):

    def test_training_rejects_optimized_and_quantized_artifacts(self):
        serving = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/model.gguf"))
        quantized = TransformersASRForSpeechRecognition(
            TransformersASRConfig(
                name_or_path="publisher/model",
                model_kwargs={"quantization_config": {
                    "load_in_4bit": True,
                }},
            ))

        with self.assertRaisesRegex(ValueError, "inference-only"):
            serving._validate_training_runtime()
        with self.assertRaisesRegex(ValueError, "unquantized"):
            quantized._validate_training_runtime()

    def test_training_accepts_safetensors_artifacts(self):
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/model.safetensors"))

        model._validate_training_runtime()

    def test_raw_audio_and_text_are_prepared_for_native_training(self):
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/asr"),
            device="cpu",
        )
        processor = FakeProcessor()
        model.transformers_processor = processor

        batch = model.prepare_training_inputs(
            {
                "audio": np.asarray([0.0, 0.1], dtype=np.float32),
                "sampling_rate": 16_000,
                "text": "hello",
            },
            phase="asr",
        )

        self.assertEqual(batch["input_values"], "processed-audio")
        self.assertEqual(batch["labels"], [[4, 5]])
        self.assertEqual(processor.calls[0][1]["sampling_rate"], 16_000)
        self.assertEqual(processor.tokenizer.calls[0][0], "hello")

    def test_padded_audio_batch_uses_lengths_and_vector_sample_rates(self):
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/asr"),
            device="cpu",
        )
        processor = FakeProcessor()
        model.transformers_processor = processor
        padded_audio = np.asarray(
            [
                [0.1, 0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        model.prepare_training_inputs(
            {
                "audio": padded_audio,
                "audio_lengths": np.asarray([2, 1]),
                "sampling_rate": np.asarray([16_000, 16_000]),
                "text": ["first", "second"],
            },
            phase="asr",
        )

        processed_audio, options = processor.calls[0]
        self.assertEqual(len(processed_audio), 2)
        self.assertEqual(
            [len(waveform) for waveform in processed_audio],
            [2, 1],
        )
        self.assertEqual(options["sampling_rate"], 16_000)
        self.assertEqual(
            processor.tokenizer.calls[0][0],
            ["first", "second"],
        )

    def test_training_rejects_empty_batches_and_scalar_batch_transcripts(self):
        model = TransformersASRForSpeechRecognition(
            TransformersASRConfig(name_or_path="publisher/asr"),
            device="cpu",
        )
        model.transformers_processor = FakeProcessor()

        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            model.prepare_training_inputs(
                {
                    "audio": [],
                    "sampling_rate": 16_000,
                    "text": [],
                },
                phase="asr",
            )
        with self.assertRaisesRegex(ValueError, "one transcription"):
            model.prepare_training_inputs(
                {
                    "audio": np.zeros((2, 4), dtype=np.float32),
                    "sampling_rate": 16_000,
                    "text": "one transcript",
                },
                phase="asr",
            )


if __name__ == "__main__":
    unittest.main()
