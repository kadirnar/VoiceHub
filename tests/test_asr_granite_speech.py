import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

from voicehub.models.asr_granite_speech import GraniteSpeechASRConfig, GraniteSpeechForSpeechRecognition
from voicehub.trainer_utils import NATIVE_EXPORT_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_RUNTIME_ENABLED = os.environ.get("VOICEHUB_FULL_RUNTIME_TEST") == "1"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


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


class FakeGraniteTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.template_calls = []
        self.target_calls = []
        self.decode_calls = []

    def apply_chat_template(self, conversation, **kwargs):
        self.template_calls.append((conversation, kwargs))
        content = conversation[0]["content"]
        return f"USER: {content}\n ASSISTANT:"

    def __call__(self, texts, **kwargs):
        self.target_calls.append((texts, kwargs))
        return {
            "input_ids": [
                [21, 22, 0],
                [31, 32, 33],
            ],
            "attention_mask": [
                [1, 1, 0],
                [1, 1, 1],
            ],
        }

    def batch_decode(self, tokens, **kwargs):
        self.decode_calls.append((tokens, kwargs))
        return ["hello Granite"]


class FakeGraniteProcessor:

    def __init__(self):
        self.audio_processor = SimpleNamespace(sampling_rate=16_000)
        self.tokenizer = FakeGraniteTokenizer()
        self.calls = []
        self.saved_to = None

    def __call__(self, prompts, audio, **kwargs):
        self.calls.append((prompts, audio, kwargs))
        batch_size = len(prompts) if isinstance(prompts, list) else 1
        return {
            "input_ids": [[10, 11, 12] for _ in range(batch_size)],
            "attention_mask": [[1, 1, 1] for _ in range(batch_size)],
            "input_features": [[[0.0] * 160 for _ in range(4)] for _ in range(batch_size)],
            "input_features_mask": [[1, 1] for _ in range(batch_size)],
            "processor_state": "preserved",
        }

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
        return np.asarray([[10, 11, 12, 90, 91]], dtype=np.int64)

    def __call__(self, **kwargs):
        self.forward_calls.append(kwargs)
        return SimpleNamespace(loss="native-loss")

    def save_pretrained(self, directory, **kwargs):
        self.saved_to = (Path(directory), kwargs)


def _fake_transformers(*, processor, model_type="granite_speech"):
    module = ModuleType("transformers")
    native_config = SimpleNamespace(
        model_type=model_type,
        architectures=["GraniteSpeechForConditionalGeneration"],
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

    class AutoModelForMultimodalLM:
        loaded_model = FakeNativeModel()

        @classmethod
        def from_pretrained(cls, source, **kwargs):
            del source, kwargs
            return cls.loaded_model

    module.AutoConfig = AutoConfig
    module.AutoProcessor = AutoProcessor
    module.AutoModelForMultimodalLM = AutoModelForMultimodalLM
    return module


class GraniteSpeechConfigTests(unittest.TestCase):

    def test_package_import_is_dependency_free(self):
        command = (
            "import sys; "
            "import voicehub.models.asr_granite_speech; "
            "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")

    def test_config_requires_the_audio_placeholder_and_prompt_conditioning(self):
        config = GraniteSpeechASRConfig()

        self.assertEqual(config.model_type, "asr_granite_speech")
        self.assertEqual(config.sample_rate, 16_000)
        self.assertIn("<|audio|>", config.transcription_prompt)
        with self.assertRaisesRegex(ValueError, "audio"):
            GraniteSpeechASRConfig(transcription_prompt="transcribe")
        with self.assertRaisesRegex(ValueError, "prompt-conditioned"):
            GraniteSpeechASRConfig(training_language="English")


class GraniteSpeechRuntimeTests(unittest.TestCase):

    def test_real_loader_uses_the_granite_processor_contract(self):
        processor = FakeGraniteProcessor()
        transformers = _fake_transformers(processor=processor)
        model = GraniteSpeechForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": transformers}):
            model.load()

        self.assertIs(model.transformers_processor, processor)
        self.assertEqual(model.architecture_family, "speech-seq2seq")

    def test_loader_fails_closed_for_structured_plus_checkpoints(self):
        processor = FakeGraniteProcessor()
        transformers = _fake_transformers(
            processor=processor,
            model_type="granite_speech_plus",
        )
        model = GraniteSpeechForSpeechRecognition(device="cpu")

        with _temporary_modules({"transformers": transformers}):
            with self.assertRaisesRegex(ValueError, "granite_speech"):
                model.load()

    def test_inference_renders_the_native_prompt_and_decodes_only_new_tokens(self):
        processor = FakeGraniteProcessor()
        native_model = FakeNativeModel()
        model = GraniteSpeechForSpeechRecognition(device="cpu")
        model.model = native_model
        model.native_config = SimpleNamespace(model_type="granite_speech")
        model.transformers_processor = processor

        output = model._transcribe(
            [0.0, 0.1],
            sampling_rate=16_000,
            prompt="transcribe with punctuation",
            hotwords=["VoiceHub", "Granite"],
            max_new_tokens=32,
        )

        rendered_prompt = processor.calls[0][0]
        self.assertIn("<|audio|>transcribe with punctuation", rendered_prompt)
        self.assertIn("Keywords: VoiceHub, Granite", rendered_prompt)
        self.assertEqual(native_model.generate_calls[0]["max_new_tokens"], 32)
        decoded_tokens = processor.tokenizer.decode_calls[0][0]
        np.testing.assert_array_equal(
            decoded_tokens,
            np.asarray([[90, 91]], dtype=np.int64),
        )
        self.assertEqual(output.text, "hello Granite")
        self.assertEqual(
            output.metadata["backend"],
            "transformers-granite-speech-asr",
        )

    def test_inference_rejects_language_forcing_and_owned_processor_options(self):
        processor = FakeGraniteProcessor()
        model = GraniteSpeechForSpeechRecognition(device="cpu")
        model.model = FakeNativeModel()
        model.native_config = SimpleNamespace(model_type="granite_speech")
        model.transformers_processor = processor

        with self.assertRaisesRegex(ValueError, "language-ID forcing"):
            model._transcribe(
                [0.0],
                sampling_rate=16_000,
                language="fr",
            )
        with self.assertRaisesRegex(ValueError, "provider-owned"):
            model._transcribe(
                [0.0],
                sampling_rate=16_000,
                processor_kwargs={"padding": False},
            )

    def test_raw_training_matches_ibm_completion_only_label_recipe(self):
        processor = FakeGraniteProcessor()
        model = GraniteSpeechForSpeechRecognition(device="cpu")
        model.transformers_processor = processor

        batch = model.prepare_training_inputs(
            {
                "audio": [
                    np.zeros(160, dtype=np.float32),
                    np.zeros(80, dtype=np.float32),
                ],
                "audio_lengths": [160, 80],
                "sampling_rate": 16_000,
                "text": ["first transcript", "second transcript"],
            },
            phase="speech_recognition",
        )

        self.assertEqual(
            batch["input_ids"],
            [
                [10, 11, 12, 21, 22, 0],
                [10, 11, 12, 31, 32, 33],
            ],
        )
        self.assertEqual(
            batch["attention_mask"],
            [
                [1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1],
            ],
        )
        self.assertEqual(
            batch["labels"],
            [
                [-100, -100, -100, 21, 22, -100],
                [-100, -100, -100, 31, 32, 33],
            ],
        )
        self.assertEqual(batch["processor_state"], "preserved")
        prompts, _audio, processor_options = processor.calls[0]
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all("<|audio|>" in prompt for prompt in prompts))
        self.assertEqual(processor_options["padding_side"], "left")
        target_texts, target_options = processor.tokenizer.target_calls[0]
        self.assertTrue(all(text.endswith("<eos>") for text in target_texts))
        self.assertEqual(target_options["padding_side"], "right")

    def test_training_rejects_an_unapplied_language_column(self):
        model = GraniteSpeechForSpeechRecognition(device="cpu")
        model.transformers_processor = FakeGraniteProcessor()

        with self.assertRaisesRegex(ValueError, "prompt-conditioned"):
            model.prepare_training_inputs(
                {
                    "audio": [0.0],
                    "sampling_rate": 16_000,
                    "text": "bonjour",
                    "language": "fr",
                },
                phase="speech_recognition",
            )

    def test_native_export_saves_safetensors_and_processor(self):
        processor = FakeGraniteProcessor()
        transformers = _fake_transformers(processor=processor)
        model = GraniteSpeechForSpeechRecognition(device="cpu")

        with tempfile.TemporaryDirectory() as directory:
            with _temporary_modules({"transformers": transformers}):
                model.load()
                model.save_pretrained(directory)
            native_directory = Path(directory) / NATIVE_EXPORT_DIR

        self.assertEqual(
            model.model.saved_to,
            (
                native_directory,
                {
                    "safe_serialization": True,
                },
            ),
        )
        self.assertEqual(processor.saved_to, native_directory)


@unittest.skipUnless(
    FULL_RUNTIME_ENABLED and TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE,
    "The real Granite Speech smoke test requires the full runtime.",
)
class GraniteSpeechFullRuntimeTests(unittest.TestCase):

    def test_real_processor_and_tiny_model_have_a_finite_backward_pass(self):
        import torch
        import transformers

        processor = transformers.AutoProcessor.from_pretrained("ibm-granite/granite-speech-4.1-2b", )
        text_config = transformers.GraniteConfig(
            vocab_size=len(processor.tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            pad_token_id=processor.tokenizer.pad_token_id,
            bos_token_id=processor.tokenizer.bos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            tie_word_embeddings=False,
        )
        encoder_config = transformers.GraniteSpeechEncoderConfig(
            input_dim=160,
            num_layers=1,
            hidden_dim=16,
            feedforward_mult=2,
            num_heads=2,
            dim_head=8,
            output_dim=16,
            context_size=20,
            max_pos_emb=64,
            dropout=0.0,
            conv_kernel_size=3,
            conv_expansion_factor=2,
        )
        projector_config = transformers.Blip2QFormerConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=64,
            encoder_hidden_size=16,
            cross_attention_frequency=1,
        )
        native_config = transformers.GraniteSpeechConfig(
            text_config=text_config,
            encoder_config=encoder_config,
            projector_config=projector_config,
            audio_token_index=processor.tokenizer.convert_tokens_to_ids("<|audio|>"),
            has_lora_adapter=False,
            downsample_rate=5,
            window_size=15,
            tie_word_embeddings=False,
        )
        native_model = transformers.GraniteSpeechForConditionalGeneration(native_config)
        model = GraniteSpeechForSpeechRecognition(device="cpu")
        model.model = native_model
        model.native_config = native_config
        model.transformers_processor = processor

        batch = model.prepare_training_inputs(
            {
                "audio": [
                    torch.zeros(1_600),
                    torch.zeros(2_400),
                ],
                "sampling_rate": 16_000,
                "text": ["hello", "world"],
            },
            phase="speech_recognition",
        )
        supervised = batch["labels"].ne(-100)
        torch.testing.assert_close(
            batch["labels"][supervised],
            batch["input_ids"][supervised],
        )
        output = native_model(**batch)

        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in native_model.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
