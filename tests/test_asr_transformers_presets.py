import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType, SimpleNamespace

from voicehub.models.asr_transformers.training_asr_transformers import TransformersASRTrainingAdapter
from voicehub.models.asr_transformers_presets import (
    CohereASRConfig,
    CohereForSpeechRecognition,
    HubertASRConfig,
    HubertForSpeechRecognition,
    MedASRConfig,
    MedASRForSpeechRecognition,
    MoonshineASRConfig,
    MoonshineForSpeechRecognition,
    NemotronASRConfig,
    NemotronForSpeechRecognition,
    ParakeetTDTASRConfig,
    ParakeetTDTForSpeechRecognition,
    SeamlessM4Tv2ASRConfig,
    SeamlessM4Tv2ForSpeechRecognition,
    Wav2Vec2ASRConfig,
    Wav2Vec2ForSpeechRecognition,
    WavLMASRConfig,
    WavLMForSpeechRecognition,
    WhisperASRConfig,
    WhisperForSpeechRecognition,
)
from voicehub.tasks import SpeechTask
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS_TORCH_AVAILABLE = (find_spec("torch") is not None and find_spec("transformers") is not None)
TRANSFORMERS_RNNT_AVAILABLE = (TRANSFORMERS_TORCH_AVAILABLE and find_spec("torchaudio") is not None)


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
        self.training = False
        self.saved_to = None
        self.config = SimpleNamespace(blank_token_id=0)
        self.weight = SimpleNamespace(
            requires_grad=True,
            dtype=SimpleNamespace(
                is_floating_point=True,
                is_complex=False,
            ),
        )

    def __call__(self, **kwargs):
        return {
            "loss": 0.0,
            **kwargs,
        }

    def parameters(self):
        return iter((self.weight, ))

    def named_parameters(self):
        return iter((("weight", self.weight), ))

    def to(self, device):
        self.device = device
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def save_pretrained(self, directory, **kwargs):
        self.saved_to = (Path(directory), kwargs)


class RecordingTokenizer:

    def __init__(self):
        self.calls = []
        self.saved_to = None

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
        }

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class RecordingProcessor:

    def __init__(self):
        self.feature_extractor = SimpleNamespace(sampling_rate=16_000)
        self.tokenizer = RecordingTokenizer()
        self.calls = []
        self.saved_to = None

    def __call__(self, *, audio, **kwargs):
        self.calls.append((audio, kwargs))
        return {
            "input_values": "processed-audio",
        }

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class RecordingJointProcessor(RecordingProcessor):
    """Record native processors that prepare audio and labels together."""

    def __init__(self, *, training_fields=()):
        super().__init__()
        self.training_fields = tuple(training_fields)

    def __call__(
        self,
        *,
        audio,
        text=None,
        **kwargs,
    ):
        self.calls.append({
            "audio": audio,
            "text": text,
            **kwargs,
        })
        batch = {
            "input_features": "processed-audio",
        }
        if text is not None:
            batch["labels"] = "native-labels"
            for name in self.training_fields:
                batch[name] = f"native-{name}"
        return batch


class RecordingBatch(dict):
    """Small BatchFeature stand-in that records device and dtype moves."""

    def __init__(self, **values):
        super().__init__(values)
        self.to_calls = []

    def to(self, *, device=None, dtype=None):
        self.to_calls.append({
            "device": device,
            "dtype": dtype,
        })
        return self


class RecordingGenerationModel:
    """Record native generate calls and return one configured output."""

    def __init__(self, output, *, inference_events=None):
        self.device = "fake-accelerator:0"
        self.dtype = object()
        self.output = output
        self.inference_events = inference_events
        self.generate_calls = []

    def generate(self, **kwargs):
        if self.inference_events is not None:
            self.inference_events.append("generate")
        self.generate_calls.append(kwargs)
        return self.output


class RecordingInferenceProcessor(RecordingJointProcessor):
    """Native processor stand-in for direct processor/generate inference."""

    def __init__(
        self,
        *,
        batch,
        decoded,
        decoded_with_special_tokens=None,
    ):
        super().__init__()
        self.batch = batch
        self.decoded = decoded
        self.decoded_with_special_tokens = decoded_with_special_tokens
        self.decode_calls = []

    def __call__(
        self,
        *,
        audio,
        text=None,
        **kwargs,
    ):
        if text is not None:
            return super().__call__(
                audio=audio,
                text=text,
                **kwargs,
            )
        self.calls.append({
            "audio": audio,
            **kwargs,
        })
        return self.batch

    def decode(self, sequences, **kwargs):
        self.decode_calls.append({
            "sequences": sequences,
            **kwargs,
        })
        if (kwargs.get("skip_special_tokens") is False and self.decoded_with_special_tokens is not None):
            return self.decoded_with_special_tokens
        return self.decoded


def _fake_torch(inference_events):
    module = ModuleType("torch")

    @contextmanager
    def inference_mode():
        inference_events.append("enter")
        try:
            yield
        finally:
            inference_events.append("exit")

    module.inference_mode = inference_mode
    return module


def _loader():

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return FakeNativeModel()

    return Loader


def _fake_transformers(
    *,
    model_type,
    architecture,
    processor=None,
):
    module = ModuleType("transformers")
    native_config = SimpleNamespace(
        model_type=model_type,
        architectures=[architecture],
    )
    resolved_processor = processor or RecordingProcessor()

    class AutoConfig:

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return native_config

    class AutoProcessor:

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return resolved_processor

    module.AutoConfig = AutoConfig
    module.AutoProcessor = AutoProcessor
    module.AutoModelForCTC = _loader()
    module.AutoModelForSpeechSeq2Seq = _loader()
    module.AutoModelForRNNT = _loader()
    module.AutoModelForTDT = _loader()
    return module


class TransformersASRPresetConfigurationTests(unittest.TestCase):

    CASES = (
        (
            Wav2Vec2ASRConfig,
            Wav2Vec2ForSpeechRecognition,
            "asr_wav2vec2",
            "ctc",
            "facebook/wav2vec2-base-960h",
            "wav2vec2",
            "Wav2Vec2ForCTC",
        ),
        (
            HubertASRConfig,
            HubertForSpeechRecognition,
            "asr_hubert",
            "ctc",
            "facebook/hubert-large-ls960-ft",
            "hubert",
            "HubertForCTC",
        ),
        (
            WavLMASRConfig,
            WavLMForSpeechRecognition,
            "asr_wavlm",
            "ctc",
            "patrickvonplaten/wavlm-libri-clean-100h-base-plus",
            "wavlm",
            "WavLMForCTC",
        ),
        (
            MoonshineASRConfig,
            MoonshineForSpeechRecognition,
            "asr_moonshine",
            "speech-seq2seq",
            "UsefulSensors/moonshine-tiny",
            "moonshine",
            "MoonshineForConditionalGeneration",
        ),
        (
            SeamlessM4Tv2ASRConfig,
            SeamlessM4Tv2ForSpeechRecognition,
            "asr_seamless_m4t_v2",
            "speech-seq2seq",
            "facebook/seamless-m4t-v2-large",
            "seamless_m4t_v2",
            "SeamlessM4Tv2Model",
        ),
        (
            WhisperASRConfig,
            WhisperForSpeechRecognition,
            "asr_whisper",
            "speech-seq2seq",
            "openai/whisper-large-v3-turbo",
            "whisper",
            "WhisperForConditionalGeneration",
        ),
        (
            ParakeetTDTASRConfig,
            ParakeetTDTForSpeechRecognition,
            "asr_parakeet_tdt",
            "tdt",
            "nvidia/parakeet-tdt-0.6b-v3",
            "parakeet_tdt",
            "ParakeetForTDT",
        ),
        (
            NemotronASRConfig,
            NemotronForSpeechRecognition,
            "asr_nemotron",
            "rnnt",
            "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "nemotron3_5_asr",
            "Nemotron3_5AsrForRNNT",
        ),
        (
            CohereASRConfig,
            CohereForSpeechRecognition,
            "asr_cohere",
            "speech-seq2seq",
            "CohereLabs/cohere-transcribe-03-2026",
            "cohere_asr",
            "CohereAsrForConditionalGeneration",
        ),
        (
            MedASRConfig,
            MedASRForSpeechRecognition,
            "asr_medasr",
            "ctc",
            "google/medasr",
            "lasr_ctc",
            "LasrForCTC",
        ),
    )

    def test_presets_have_real_defaults_and_locked_architecture_families(self):
        for (config_class, model_class, model_type, family, checkpoint, _native_type,
             _architecture) in self.CASES:
            with self.subTest(model_type=model_type):
                model = model_class()

                self.assertIs(model.config_class, config_class)
                self.assertEqual(model.config.model_type, model_type)
                self.assertEqual(model.config.architecture_family, family)
                self.assertEqual(
                    model.default_model_name_or_path,
                    checkpoint,
                )
                self.assertEqual(model.config.name_or_path, checkpoint)
                with self.assertRaisesRegex(
                        ValueError,
                        "requires `architecture_family",
                ):
                    config_class(architecture_family="auto")

    def test_presets_remain_dependency_free_to_import(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_transformers_presets; "
            "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")

    def test_seamless_target_language_is_validated_and_serialized(self):
        config = SeamlessM4Tv2ASRConfig(target_language=" TUR ")

        self.assertEqual(config.target_language, "tur")
        self.assertEqual(config.to_dict()["target_language"], "tur")
        with self.assertRaisesRegex(ValueError, "target_language"):
            SeamlessM4Tv2ASRConfig(target_language="")

    def test_language_conditioned_presets_validate_and_serialize_options(self):
        nemotron = NemotronASRConfig(target_language=" de-DE ")
        cohere = CohereASRConfig(
            target_language=" FR ",
            punctuation=False,
        )

        self.assertEqual(nemotron.target_language, "de-DE")
        self.assertEqual(
            nemotron.to_dict()["target_language"],
            "de-DE",
        )
        self.assertEqual(cohere.target_language, "fr")
        self.assertFalse(cohere.punctuation)
        self.assertEqual(cohere.to_dict()["target_language"], "fr")
        self.assertFalse(cohere.to_dict()["punctuation"])

        for factory in (
                lambda: NemotronASRConfig(target_language=""),
                lambda: CohereASRConfig(target_language=""),
                lambda: CohereASRConfig(punctuation=1),
        ):
            with self.subTest(factory=factory), self.assertRaises((TypeError, ValueError)):
                factory()


class TransformersASRPresetLoadingTests(unittest.TestCase):

    CASES = TransformersASRPresetConfigurationTests.CASES

    def test_each_preset_dispatches_to_its_native_auto_model_class(self):
        for (_config_class, model_class, model_type, family, checkpoint, native_type,
             architecture) in self.CASES:
            with self.subTest(model_type=model_type):
                fake_transformers = _fake_transformers(
                    model_type=native_type,
                    architecture=architecture,
                )
                model = model_class(
                    use_safetensors=True,
                    device="cpu",
                )

                with _temporary_modules({"transformers": fake_transformers}):
                    model._load_pretrained_model()

                loader = {
                    "ctc": fake_transformers.AutoModelForCTC,
                    "speech-seq2seq": fake_transformers.AutoModelForSpeechSeq2Seq,
                    "rnnt": fake_transformers.AutoModelForRNNT,
                    "tdt": fake_transformers.AutoModelForTDT,
                }[family]
                self.assertEqual(model.architecture_family, family)
                self.assertEqual(model.model.device, "cpu")
                self.assertEqual(loader.calls[0][0], checkpoint)
                self.assertTrue(loader.calls[0][1]["use_safetensors"])
                self.assertIs(
                    loader.calls[0][1]["config"],
                    model.native_config,
                )

    def test_cross_family_checkpoint_is_rejected_before_weight_loading(self):
        fake_transformers = _fake_transformers(
            model_type="hubert",
            architecture="HubertForCTC",
        )
        model = Wav2Vec2ForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": fake_transformers}):
            with self.assertRaisesRegex(
                    ValueError,
                    "requires a Transformers checkpoint",
            ):
                model._load_pretrained_model()

        self.assertEqual(
            fake_transformers.AutoModelForCTC.calls,
            [],
        )


class TransformersASRPresetInferenceAndTrainingTests(unittest.TestCase):

    def test_ctc_boolean_timestamps_map_to_word_timestamps(self):
        model = Wav2Vec2ForSpeechRecognition(device="cpu")
        model.architecture_family = "ctc"
        model.transformers_processor = SimpleNamespace(tokenizer=SimpleNamespace(), )

        options = model._pipeline_call_options(
            language=None,
            task="transcribe",
            return_timestamps=True,
            chunk_length_s=None,
            stride_length_s=None,
            batch_size=None,
            num_beams=None,
            max_new_tokens=None,
            hotwords=None,
            options={},
        )

        self.assertEqual(options["return_timestamps"], "word")

    def test_ctc_timestamp_mode_rejects_pipeline_incompatible_values(self):
        model = Wav2Vec2ForSpeechRecognition(device="cpu")
        model.architecture_family = "ctc"
        model.transformers_processor = SimpleNamespace(tokenizer=SimpleNamespace(), )

        with self.assertRaisesRegex(ValueError, "CTC timestamp mode"):
            model._pipeline_call_options(
                language=None,
                task="transcribe",
                return_timestamps="segment",
                chunk_length_s=None,
                stride_length_s=None,
                batch_size=None,
                num_beams=None,
                max_new_tokens=None,
                hotwords=None,
                options={},
            )

    def test_seamless_maps_public_language_to_native_target_language(self):
        model = SeamlessM4Tv2ForSpeechRecognition(
            target_language="eng",
            device="cpu",
        )
        model.architecture_family = "speech-seq2seq"

        transcribe_options = model._pipeline_call_options(
            language=None,
            task="transcribe",
            return_timestamps=False,
            chunk_length_s=None,
            stride_length_s=None,
            batch_size=None,
            num_beams=2,
            max_new_tokens=32,
            hotwords=None,
            options={},
        )
        translate_options = model._pipeline_call_options(
            language="tur",
            task="translate",
            return_timestamps=False,
            chunk_length_s=None,
            stride_length_s=None,
            batch_size=None,
            num_beams=None,
            max_new_tokens=None,
            hotwords=None,
            options={},
        )

        self.assertEqual(
            transcribe_options["generate_kwargs"],
            {
                "num_beams": 2,
                "max_new_tokens": 32,
                "tgt_lang": "eng",
            },
        )
        self.assertEqual(
            translate_options["generate_kwargs"],
            {
                "tgt_lang": "tur",
            },
        )
        self.assertNotIn("task", translate_options["generate_kwargs"])
        self.assertNotIn("language", translate_options["generate_kwargs"])

    def test_seamless_training_labels_use_target_language_tokens(self):
        model = SeamlessM4Tv2ForSpeechRecognition(
            target_language="deu",
            device="cpu",
        )
        processor = RecordingProcessor()
        model.transformers_processor = processor

        labels = model._tokenize_training_labels(["hallo"])

        self.assertEqual(labels["input_ids"], [[1, 2]])
        args, kwargs = processor.tokenizer.calls[0]
        self.assertEqual(args, ())
        self.assertEqual(kwargs["text_target"], ["hallo"])
        self.assertEqual(kwargs["tgt_lang"], "deu")

    def test_shared_training_audio_router_uses_named_audio_input(self):
        model = MoonshineForSpeechRecognition(device="cpu")
        processor = RecordingProcessor()
        model.transformers_processor = processor

        encoded = model._process_training_audio(
            [0.0, 0.1],
            sampling_rate=16_000,
        )

        self.assertEqual(encoded["input_values"], "processed-audio")
        self.assertEqual(processor.calls[0][0], [0.0, 0.1])
        self.assertEqual(processor.calls[0][1]["sampling_rate"], 16_000)

    def test_parakeet_native_inference_preserves_tensors_and_decodes_durations(self):
        sequences = object()
        durations = object()
        batch = RecordingBatch(
            input_features=object(),
            attention_mask=object(),
        )
        processor = RecordingInferenceProcessor(
            batch=batch,
            decoded=(
                ["hello world!"],
                [[
                    {
                        "token": " hello",
                        "start": 0.0,
                        "end": 0.4,
                    },
                    {
                        "token": " wor",
                        "start": 0.5,
                        "end": 0.8,
                    },
                    {
                        "token": "ld",
                        "start": 0.8,
                        "end": 0.95,
                    },
                    {
                        "token": "!",
                        "start": 0.95,
                        "end": 0.95,
                    },
                ]],
            ),
        )
        inference_events = []
        native_model = RecordingGenerationModel(
            SimpleNamespace(
                sequences=sequences,
                durations=durations,
            ),
            inference_events=inference_events,
        )
        model = ParakeetTDTForSpeechRecognition(device="cpu")
        model.model = native_model
        model.transformers_processor = processor
        model.architecture_family = "tdt"

        with _temporary_modules({
                "torch": _fake_torch(inference_events),
        }):
            output = model._transcribe(
                [0.0, 0.1, 0.2],
                sampling_rate=16_000,
                return_timestamps="word",
                num_beams=1,
                max_new_tokens=64,
                generate_kwargs={
                    "temperature": 0.2,
                },
                top_k=4,
            )

        self.assertEqual(inference_events, ["enter", "generate", "exit"])
        processor_call = processor.calls[0]
        for actual, expected in zip(processor_call["audio"].tolist(), [0.0, 0.1, 0.2]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(processor_call["sampling_rate"], 16_000)
        self.assertEqual(processor_call["return_tensors"], "pt")
        self.assertEqual(
            batch.to_calls,
            [{
                "device": native_model.device,
                "dtype": native_model.dtype,
            }],
        )
        generation_call = native_model.generate_calls[0]
        self.assertIs(
            generation_call["input_features"],
            batch["input_features"],
        )
        self.assertIs(
            generation_call["attention_mask"],
            batch["attention_mask"],
        )
        self.assertEqual(generation_call["num_beams"], 1)
        self.assertEqual(generation_call["max_new_tokens"], 64)
        self.assertEqual(generation_call["temperature"], 0.2)
        self.assertEqual(generation_call["top_k"], 4)
        self.assertTrue(generation_call["return_dict_in_generate"])
        self.assertEqual(
            processor.decode_calls,
            [{
                "sequences": sequences,
                "skip_special_tokens": True,
                "durations": durations,
            }],
        )
        self.assertEqual(output.text, "hello world!")
        self.assertEqual(len(output.segments), 1)
        segment = output.segments[0]
        self.assertEqual(segment.start, 0.0)
        self.assertEqual(segment.end, 0.95)
        self.assertEqual(
            [word.text for word in segment.words],
            ["hello", "world!"],
        )
        self.assertEqual(segment.words[1].start, 0.5)
        self.assertEqual(segment.words[1].end, 0.95)
        self.assertEqual(
            output.metadata["native_token_timestamps"],
            processor.decoded[1],
        )
        self.assertEqual(output.metadata["architecture_family"], "tdt")
        self.assertAlmostEqual(output.duration, 3 / 16_000)

    def test_nemotron_routes_language_and_all_processor_fields_to_generate(self):
        sequences = object()
        batch = RecordingBatch(
            input_features=object(),
            attention_mask=object(),
            prompt_ids=object(),
            num_lookahead_tokens=object(),
        )
        processor = RecordingInferenceProcessor(
            batch=batch,
            decoded=["guten tag"],
        )
        inference_events = []
        native_model = RecordingGenerationModel(
            SimpleNamespace(
                sequences=sequences,
                durations=object(),
            ),
            inference_events=inference_events,
        )
        model = NemotronForSpeechRecognition(
            target_language="auto",
            device="cpu",
        )
        model.model = native_model
        model.transformers_processor = processor
        model.architecture_family = "rnnt"

        with _temporary_modules({
                "torch": _fake_torch(inference_events),
        }):
            output = model._transcribe(
                [0.0, 0.1],
                sampling_rate=16_000,
                language=" de-DE ",
                max_new_tokens=48,
            )

        self.assertEqual(inference_events, ["enter", "generate", "exit"])
        processor_call = processor.calls[0]
        self.assertEqual(processor_call["language"], "de-DE")
        self.assertFalse(processor_call["is_streaming"])
        self.assertTrue(processor_call["is_first_audio_chunk"])
        generation_call = native_model.generate_calls[0]
        for name, value in batch.items():
            self.assertIs(generation_call[name], value)
        self.assertNotIn("language", generation_call)
        self.assertEqual(generation_call["max_new_tokens"], 48)
        self.assertEqual(
            processor.decode_calls,
            [{
                "sequences": sequences,
                "skip_special_tokens": True,
            }],
        )
        self.assertEqual(output.text, "guten tag")
        self.assertEqual(output.language, "de-DE")
        self.assertEqual(output.metadata["architecture_family"], "rnnt")

    def test_nemotron_auto_language_mode_normalizes_the_emitted_tag(self):
        sequences = object()
        processor = RecordingInferenceProcessor(
            batch=RecordingBatch(
                input_features=object(),
                prompt_ids=object(),
            ),
            decoded=["hola mundo."],
            decoded_with_special_tokens=["hola mundo.<es-ES>"],
        )
        inference_events = []
        model = NemotronForSpeechRecognition(
            target_language="auto",
            device="cpu",
        )
        model.model = RecordingGenerationModel(
            SimpleNamespace(sequences=sequences),
            inference_events=inference_events,
        )
        model.transformers_processor = processor
        model.architecture_family = "rnnt"

        with _temporary_modules({
                "torch": _fake_torch(inference_events),
        }):
            output = model._transcribe(
                [0.0, 0.1],
                sampling_rate=16_000,
            )

        self.assertEqual(output.text, "hola mundo.")
        self.assertEqual(output.language, "es-ES")
        self.assertEqual(
            output.metadata["detected_language"],
            "es-ES",
        )
        self.assertEqual(
            processor.decode_calls,
            [
                {
                    "sequences": sequences,
                    "skip_special_tokens": True,
                },
                {
                    "sequences": sequences,
                    "skip_special_tokens": False,
                },
            ],
        )

    def test_cohere_reassembles_native_chunks_with_decode_metadata(self):
        sequences = object()
        audio_chunk_index = [(0, 0), (0, 1)]
        batch = RecordingBatch(
            input_features=object(),
            attention_mask=object(),
            decoder_input_ids=object(),
            audio_chunk_index=audio_chunk_index,
        )
        processor = RecordingInferenceProcessor(
            batch=batch,
            decoded=["first second"],
        )
        inference_events = []
        native_model = RecordingGenerationModel(
            {
                "sequences": sequences,
            },
            inference_events=inference_events,
        )
        model = CohereForSpeechRecognition(
            target_language="en",
            punctuation=False,
            device="cpu",
        )
        model.model = native_model
        model.transformers_processor = processor
        model.architecture_family = "speech-seq2seq"

        with _temporary_modules({
                "torch": _fake_torch(inference_events),
        }):
            output = model._transcribe(
                [0.0, 0.1],
                sampling_rate=16_000,
                language=" JA ",
            )

        self.assertEqual(inference_events, ["enter", "generate", "exit"])
        processor_call = processor.calls[0]
        self.assertEqual(processor_call["language"], "ja")
        self.assertFalse(processor_call["punctuation"])
        generation_call = native_model.generate_calls[0]
        for name, value in batch.items():
            self.assertIs(generation_call[name], value)
        decode_call = processor.decode_calls[0]
        self.assertIs(
            decode_call["audio_chunk_index"],
            audio_chunk_index,
        )
        self.assertEqual(decode_call["language"], "ja")
        self.assertTrue(decode_call["skip_special_tokens"])
        self.assertEqual(output.text, "first second")
        self.assertEqual(output.language, "ja")
        self.assertEqual(
            output.metadata["audio_chunk_index"],
            audio_chunk_index,
        )
        self.assertEqual(output.metadata["audio_chunk_count"], 2)
        self.assertTrue(output.metadata["long_form_reassembled"])
        self.assertEqual(output.segments, ())

    def test_native_inference_rejects_unsupported_common_options(self):
        cases = (
            (
                ParakeetTDTForSpeechRecognition(device="cpu"),
                {
                    "task": "translate",
                },
                "translation",
            ),
            (
                ParakeetTDTForSpeechRecognition(device="cpu"),
                {
                    "language": "en",
                },
                "language",
            ),
            (
                ParakeetTDTForSpeechRecognition(device="cpu"),
                {
                    "chunk_length_s": 30.0,
                },
                "chunk_length_s",
            ),
            (
                NemotronForSpeechRecognition(device="cpu"),
                {
                    "stride_length_s": (2.0, 2.0),
                },
                "stride_length_s",
            ),
            (
                NemotronForSpeechRecognition(device="cpu"),
                {
                    "hotwords": ("VoiceHub", ),
                },
                "hotword",
            ),
            (
                ParakeetTDTForSpeechRecognition(device="cpu"),
                {
                    "num_beams": 2,
                },
                "greedy transducer decoding",
            ),
            (
                NemotronForSpeechRecognition(device="cpu"),
                {
                    "num_beams": 2,
                },
                "greedy transducer decoding",
            ),
            (
                CohereForSpeechRecognition(device="cpu"),
                {
                    "return_timestamps": True,
                },
                "timestamps",
            ),
        )

        for model, options, expected_message in cases:
            with self.subTest(model_type=model.config.model_type, options=options):
                processor = RecordingInferenceProcessor(
                    batch=RecordingBatch(input_features=object()),
                    decoded=["unused"],
                )
                model.model = RecordingGenerationModel(SimpleNamespace(sequences=object()))
                model.transformers_processor = processor

                with self.assertRaisesRegex(ValueError, expected_message):
                    model._transcribe(
                        [0.0],
                        sampling_rate=16_000,
                        **options,
                    )

                self.assertEqual(processor.calls, [])
                self.assertEqual(model.model.generate_calls, [])

    def test_native_generation_options_cannot_replace_processor_tensors(self):
        batch = RecordingBatch(input_features=object())
        processor = RecordingInferenceProcessor(
            batch=batch,
            decoded=["unused"],
        )
        native_model = RecordingGenerationModel(SimpleNamespace(sequences=object()))
        model = ParakeetTDTForSpeechRecognition(device="cpu")
        model.model = native_model
        model.transformers_processor = processor

        with self.assertRaisesRegex(ValueError, "cannot replace native processor tensor"):
            model._transcribe(
                [0.0],
                sampling_rate=16_000,
                generate_kwargs={
                    "input_features": "replacement",
                },
            )

        self.assertEqual(native_model.generate_calls, [])

    def test_joint_native_processors_own_training_label_construction(self):
        cases = (
            (
                ParakeetTDTForSpeechRecognition(device="cpu"),
                RecordingJointProcessor(training_fields=("decoder_input_ids", )),
                {},
                ("decoder_input_ids", ),
            ),
            (
                NemotronForSpeechRecognition(
                    target_language="de-DE",
                    device="cpu",
                ),
                RecordingJointProcessor(
                    training_fields=(
                        "decoder_input_ids",
                        "prompt_ids",
                        "num_lookahead_tokens",
                    ), ),
                {
                    "language": "de-DE",
                },
                (
                    "decoder_input_ids",
                    "prompt_ids",
                    "num_lookahead_tokens",
                ),
            ),
            (
                CohereForSpeechRecognition(
                    target_language="en",
                    punctuation=False,
                    device="cpu",
                ),
                RecordingJointProcessor(training_fields=("decoder_input_ids", )),
                {
                    "language": "fr",
                    "punctuation": False,
                },
                ("decoder_input_ids", ),
            ),
            (
                MedASRForSpeechRecognition(device="cpu"),
                RecordingJointProcessor(),
                {},
                (),
            ),
        )

        for model, processor, record_overrides, native_fields in cases:
            with self.subTest(model_type=model.config.model_type):
                model.transformers_processor = processor
                record = {
                    "audio": [0.0, 0.1],
                    "sampling_rate": 16_000,
                    "text": "native transcript",
                    **({
                        "language": "fr",
                    } if model.config.model_type == "asr_cohere" else {}),
                }

                batch = model.prepare_training_inputs(
                    record,
                    phase="speech_recognition",
                )

                self.assertEqual(batch["labels"], "native-labels")
                self.assertEqual(
                    processor.tokenizer.calls,
                    [],
                    "Joint processors must not be followed by a second "
                    "tokenizer pass.",
                )
                for name in native_fields:
                    self.assertEqual(batch[name], f"native-{name}")
                call = processor.calls[0]
                self.assertEqual(call["text"], "native transcript")
                self.assertEqual(call["sampling_rate"], 16_000)
                self.assertTrue(call["padding"])
                self.assertEqual(call["return_tensors"], "pt")
                for name, expected in record_overrides.items():
                    self.assertEqual(call[name], expected)

    def test_joint_processor_batches_trim_padded_audio_before_native_call(self):
        model = ParakeetTDTForSpeechRecognition(device="cpu")
        processor = RecordingJointProcessor(training_fields=("decoder_input_ids", ))
        model.transformers_processor = processor

        batch = model.prepare_training_inputs(
            {
                "audio": [
                    [0.1, 0.2, 0.0],
                    [0.3, 0.0, 0.0],
                ],
                "audio_lengths": [2, 1],
                "sampling_rate": [16_000, 16_000],
                "text": ["first", "second"],
            },
            phase="speech_recognition",
        )

        self.assertEqual(batch["labels"], "native-labels")
        call = processor.calls[0]
        self.assertEqual(
            [len(waveform) for waveform in call["audio"]],
            [2, 1],
        )
        self.assertEqual(call["text"], ["first", "second"])

    def test_cohere_training_collapses_one_homogeneous_batch_language(self):
        model = CohereForSpeechRecognition(device="cpu")
        processor = RecordingJointProcessor(training_fields=("decoder_input_ids", ), )
        model.transformers_processor = processor

        model.prepare_training_inputs(
            {
                "audio": [
                    [0.0, 0.1],
                    [0.2, 0.3],
                ],
                "sampling_rate": [16_000, 16_000],
                "text": ["hello", "world"],
                "language": [" EN ", "en"],
            },
            phase="speech_recognition",
        )

        self.assertEqual(processor.calls[0]["language"], "en")

    def test_cohere_training_rejects_a_mixed_language_batch(self):
        model = CohereForSpeechRecognition(device="cpu")
        model.transformers_processor = RecordingJointProcessor()

        with self.assertRaisesRegex(ValueError, "Group training records"):
            model.prepare_training_inputs(
                {
                    "audio": [
                        [0.0, 0.1],
                        [0.2, 0.3],
                    ],
                    "sampling_rate": [16_000, 16_000],
                    "text": ["hello", "bonjour"],
                    "language": ["en", "fr"],
                },
                phase="speech_recognition",
            )

    def test_cohere_training_rejects_inference_only_long_audio_chunks(self):

        class ChunkingProcessor:

            def __init__(self, chunk_index):
                self.chunk_index = chunk_index

            def __call__(self, **kwargs):
                del kwargs
                return {
                    "input_features": "features",
                    "decoder_input_ids": "decoder-inputs",
                    "labels": "labels",
                    "audio_chunk_index": self.chunk_index,
                }

        model = CohereForSpeechRecognition(device="cpu")
        model.transformers_processor = ChunkingProcessor([
            (0, 0),
            (0, 1),
        ])

        with self.assertRaisesRegex(
                ValueError,
                "Pre-segment long recordings",
        ):
            model._joint_processor_training_batch(
                audio=[0.0],
                text="one transcript",
                sampling_rate=16_000,
                inputs={
                    "language": "en",
                },
            )

        model.transformers_processor = ChunkingProcessor([(0, None)])
        batch = model._joint_processor_training_batch(
            audio=[0.0],
            text="one transcript",
            sampling_rate=16_000,
            inputs={
                "language": "en",
            },
        )
        self.assertNotIn("audio_chunk_index", batch)
        self.assertEqual(batch["labels"], "labels")

    @unittest.skipUnless(
        TRANSFORMERS_TORCH_AVAILABLE,
        "The real Cohere training smoke test requires torch and transformers.",
    )
    def test_cohere_prompt_alignment_has_a_finite_real_backward_pass(self):
        import torch
        import transformers

        native_config = transformers.CohereAsrConfig(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            pad_token_id=2,
            eos_token_id=3,
            bos_token_id=4,
            encoder_config={
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "conv_kernel_size": 3,
                "subsampling_factor": 2,
                "subsampling_conv_channels": 4,
                "num_mel_bins": 8,
                "subsampling_conv_kernel_size": 3,
                "subsampling_conv_stride": 2,
                "dropout": 0.0,
                "activation_dropout": 0.0,
                "attention_dropout": 0.0,
                "layerdrop": 0.0,
                "scale_input": False,
            },
        )
        native_model = transformers.CohereAsrForConditionalGeneration(native_config, )

        class TensorCohereProcessor:
            tokenizer = SimpleNamespace(pad_token_id=2)

            def __call__(self, **kwargs):
                del kwargs
                return {
                    "input_features": torch.randn(2, 17, 8),
                    "attention_mask": torch.ones(2, 17, dtype=torch.long),
                    # Short audio has no chunk map in the real processor.
                    "audio_chunk_index": None,
                    "decoder_input_ids": torch.tensor([
                        [10, 11, 12, 13, 14],
                        [10, 11, 12, 13, 14],
                    ]),
                    "labels": torch.tensor([
                        [7, 8, 3],
                        [9, 2, 2],
                    ]),
                }

        model = CohereForSpeechRecognition(device="cpu")
        model.model = native_model
        model.native_config = native_config
        model.transformers_processor = TensorCohereProcessor()

        batch = model._joint_processor_training_batch(
            audio=[0.0],
            text=["first", "second"],
            sampling_rate=16_000,
            inputs={"language": "en"},
        )

        torch.testing.assert_close(
            batch["decoder_input_ids"],
            torch.tensor([
                [10, 11, 12, 13, 14, 7, 8],
                [10, 11, 12, 13, 14, 9, 2],
            ]),
        )
        torch.testing.assert_close(
            batch["labels"],
            torch.tensor([
                [-100, -100, -100, -100, 7, 8, 3],
                [-100, -100, -100, -100, 9, -100, -100],
            ]),
        )
        torch.testing.assert_close(
            batch["decoder_attention_mask"],
            torch.tensor([
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0],
            ]),
        )

        cached_batch = model.prepare_training_inputs(
            batch,
            phase="speech_recognition",
        )
        torch.testing.assert_close(
            cached_batch["decoder_input_ids"],
            batch["decoder_input_ids"],
        )
        torch.testing.assert_close(
            cached_batch["labels"],
            batch["labels"],
        )

        model._prepare_for_training()
        output = native_model(**batch)

        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in native_model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    @unittest.skipUnless(
        TRANSFORMERS_RNNT_AVAILABLE,
        "The real Nemotron training smoke test requires torch, torchaudio, "
        "and transformers.",
    )
    def test_nemotron_blank_normalization_has_a_finite_real_rnnt_backward_pass(self, ):
        import torch
        import transformers

        native_config = transformers.Nemotron3_5AsrConfig(
            vocab_size=16,
            blank_token_id=15,
            pad_token_id=0,
            decoder_hidden_size=8,
            num_decoder_layers=1,
            num_prompts=4,
            prompt_intermediate_size=8,
            default_prompt_id=0,
            encoder_config={
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 16,
                "conv_kernel_size": 3,
                "subsampling_factor": 2,
                "subsampling_conv_channels": 4,
                "num_mel_bins": 8,
                "subsampling_conv_kernel_size": 3,
                "subsampling_conv_stride": 2,
                "sliding_window": 8,
                "default_num_lookahead_tokens": 0,
                "dropout": 0.0,
                "activation_dropout": 0.0,
                "attention_dropout": 0.0,
                "layerdrop": 0.0,
            },
        )
        native_model = transformers.Nemotron3_5AsrForRNNT(native_config)

        class TensorNemotronProcessor:
            # The published processor currently emits a prefix one past the
            # model's last vocabulary index.
            blank_token_id = 16

            def __call__(self, **kwargs):
                del kwargs
                return {
                    "input_features": torch.randn(1, 17, 8),
                    "attention_mask": torch.ones(1, 17, dtype=torch.long),
                    "prompt_ids": torch.tensor([0]),
                    "num_lookahead_tokens": 0,
                    "labels": torch.tensor([[2, 3, 4]]),
                    "decoder_input_ids": torch.tensor([[16, 2, 3, 4]]),
                }

        model = NemotronForSpeechRecognition(device="cpu")
        model.model = native_model
        model.native_config = native_config
        model.transformers_processor = TensorNemotronProcessor()

        batch = model._joint_processor_training_batch(
            audio=[0.0],
            text="transcript",
            sampling_rate=16_000,
            inputs={"language": "en-US"},
        )

        torch.testing.assert_close(
            batch["decoder_input_ids"],
            torch.tensor([[15, 2, 3, 4]]),
        )
        model._prepare_for_training()
        output = native_model(**batch)

        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in native_model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_each_preset_uses_the_native_transformers_training_adapter(self):
        expected_objectives = {
            "ctc": "CTC",
            "speech-seq2seq": "speech sequence-to-sequence",
            "rnnt": "RNN-T",
            "tdt": "TDT",
        }
        for (config_class, model_class, model_type, family, _checkpoint, _native_type,
             _architecture) in (TransformersASRPresetConfigurationTests.CASES):
            with self.subTest(model_type=model_type):
                model = model_class(device="cpu")
                model.model = FakeNativeModel()
                model.transformers_processor = RecordingProcessor()
                spec = ModelTrainingSpec(
                    model_type=model_type,
                    family=TrainingFamily.UPSTREAM_NATIVE,
                    task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
                    module_paths=("model", ),
                    component_paths=("model", ),
                    native_training=True,
                    support=TrainingSupport.NATIVE,
                )
                adapter = TransformersASRTrainingAdapter(model, spec)

                fake_rnnt_loss = ModuleType("transformers.loss.loss_rnnt", )
                fake_rnnt_loss.ParakeetForRNNTLoss = lambda **kwargs: kwargs
                fake_loss_utils = ModuleType("transformers.loss.loss_utils", )
                fake_loss_utils.ForMaskedLMLoss = lambda **kwargs: kwargs
                with _temporary_modules({
                        "transformers.loss.loss_rnnt": fake_rnnt_loss,
                        "transformers.loss.loss_utils": fake_loss_utils,
                }):
                    adapter.setup()

                self.assertIsInstance(model.config, config_class)
                self.assertEqual(adapter.native_family, family)
                self.assertEqual(
                    adapter.objective_name,
                    expected_objectives[family],
                )
                self.assertTrue(model.model.training)

    def test_native_export_is_safetensors_and_processor_complete(self):
        model = Wav2Vec2ForSpeechRecognition(device="cpu")
        model.model = FakeNativeModel()
        model.transformers_processor = RecordingProcessor()
        spec = ModelTrainingSpec(
            model_type="asr_wav2vec2",
            family=TrainingFamily.UPSTREAM_NATIVE,
            task=SpeechTask.AUTOMATIC_SPEECH_RECOGNITION,
            module_paths=("model", ),
            component_paths=("model", ),
            native_training=True,
            support=TrainingSupport.NATIVE,
        )
        adapter = TransformersASRTrainingAdapter(model, spec)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "export"
            adapter.save_pretrained(destination)

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
