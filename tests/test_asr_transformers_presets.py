import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

from voicehub.models.asr_transformers.training_asr_transformers import TransformersASRTrainingAdapter
from voicehub.models.asr_transformers_presets import (
    HubertASRConfig,
    HubertForSpeechRecognition,
    MoonshineASRConfig,
    MoonshineForSpeechRecognition,
    SeamlessM4Tv2ASRConfig,
    SeamlessM4Tv2ForSpeechRecognition,
    Wav2Vec2ASRConfig,
    Wav2Vec2ForSpeechRecognition,
    WavLMASRConfig,
    WavLMForSpeechRecognition,
)
from voicehub.tasks import SpeechTask
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

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
        self.training = False
        self.saved_to = None
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


def _loader():

    class Loader:
        calls = []

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            cls.calls.append((source, kwargs))
            return FakeNativeModel()

    return Loader


def _fake_transformers(*, model_type, architecture):
    module = ModuleType("transformers")
    native_config = SimpleNamespace(
        model_type=model_type,
        architectures=[architecture],
    )
    processor = RecordingProcessor()

    class AutoConfig:

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return native_config

    class AutoProcessor:

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return processor

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

                loader = (
                    fake_transformers.AutoModelForCTC
                    if family == "ctc" else fake_transformers.AutoModelForSpeechSeq2Seq)
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

    def test_each_preset_uses_the_native_transformers_training_adapter(self):
        expected_objectives = {
            "ctc": "CTC",
            "speech-seq2seq": "speech sequence-to-sequence",
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
