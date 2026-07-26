import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import voicehub.models.dia.training as dia_training
from voicehub.models.dia.inference import DiaConfig, DiaForTextToSpeech
from voicehub.models.dia.training import (
    DiaSFTDataset,
    DiaTrainingAdapter,
    DiaTrainingBackend,
    DiaTrainingCollator,
    load_dia_transformers_backend,
)
from voicehub.trainer_utils import NATIVE_EXPORT_DIR
from voicehub.training.contracts import TrainingSupport
from voicehub.training.specs import ModelTrainingSpec, TrainingFamily

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeBatch(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


class FakeProcessor:

    def __init__(self, audio_tokenizer=None):
        self.feature_extractor = SimpleNamespace(sampling_rate=44_100)
        self.audio_tokenizer = audio_tokenizer
        self.calls = []
        self.saved_to = None
        self.decode_calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return FakeBatch({
            "input_ids": "input-ids",
            "attention_mask": "attention-mask",
            "decoder_input_ids": "decoder-input-ids",
            "decoder_attention_mask": "decoder-attention-mask",
            "labels": "labels",
        })

    def get_audio_prompt_len(self, attention_mask):
        self.prompt_mask = attention_mask
        return 7

    def batch_decode(self, sequences, **kwargs):
        self.decode_calls.append((sequences, kwargs))
        return [[0.25, -0.25]]

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class DiaTrainingInputTests(unittest.TestCase):

    def test_training_module_keeps_heavy_dependencies_lazy(self):
        command = (
            "import sys; "
            "import voicehub.models.dia.training; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False")

    def test_collator_uses_official_training_processor_contract(self):
        processor = FakeProcessor()
        collator = DiaTrainingCollator(
            processor,
            processor_kwargs={
                "truncation": True,
            },
        )

        batch = collator([
            {
                "text": "[S1] Hello.",
                "audio": {
                    "array": [0.0, 0.1],
                    "sampling_rate": 44_100,
                },
            },
            {
                "text": "[S2] Hi.",
                "audio": {
                    "array": [0.2, 0.3],
                    "sampling_rate": 44_100,
                },
            },
        ])

        self.assertEqual(batch["labels"], "labels")
        self.assertEqual(
            processor.calls[0],
            {
                "text": ["[S1] Hello.", "[S2] Hi."],
                "audio": [[0.0, 0.1], [0.2, 0.3]],
                "generation": False,
                "output_labels": True,
                "padding": True,
                "return_tensors": "pt",
                "truncation": True,
            },
        )

    def test_collator_rejects_wrong_sample_rate(self):
        collator = DiaTrainingCollator(FakeProcessor())
        with self.assertRaisesRegex(ValueError, "resampled to 44100 Hz"):
            collator([{
                "text": "[S1] Wrong rate.",
                "audio": {
                    "array": [0.0],
                    "sampling_rate": 24_000,
                },
            }])

    def test_processor_controls_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError, "output_labels"):
            DiaTrainingCollator(
                FakeProcessor(),
                processor_kwargs={
                    "output_labels": False,
                },
            )

    def test_dataset_exposes_the_matching_collator(self):
        processor = FakeProcessor()
        dataset = DiaSFTDataset(
            [{
                "text": "[S1] Dataset.",
                "audio": {
                    "array": [0.0, 0.1],
                    "sampling_rate": 44_100,
                },
            }],
            processor=processor,
        )

        item = dataset[0]
        self.assertEqual(item["text"], "[S1] Dataset.")
        self.assertEqual(dataset.collate_fn([item])["labels"], "labels")


class DiaWrapperTests(unittest.TestCase):

    def test_backend_selection_preserves_legacy_inference(self):
        legacy = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B",
            device="cpu",
        )
        converted = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B-0626",
            device="cpu",
        )

        self.assertEqual(
            legacy._select_backend(for_training=False),
            "legacy",
        )
        self.assertEqual(
            converted._select_backend(for_training=False),
            "transformers",
        )
        self.assertEqual(
            converted._select_backend(for_training=True),
            "transformers",
        )

    def test_legacy_checkpoint_training_error_is_actionable(self):
        model = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B",
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "Dia-1.6B-0626"):
            model._validate_training_runtime()

    def test_backend_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            DiaConfig(backend="fused")

    def test_transformers_generation_uses_processor_decode(self):
        processor = FakeProcessor()

        class FakeModel:

            def __init__(self):
                self.config = SimpleNamespace(use_cache=True)
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return "generated-codes"

        runtime = FakeModel()
        backend = DiaTrainingBackend(
            model=runtime,
            processor=processor,
            sample_rate=44_100,
        )
        model = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B-0626",
            device="cpu",
        )
        model.model = runtime
        model._dia_backend = backend
        model._loaded_backend = "transformers"

        with patch(
                "voicehub.models.dia.inference.seeded_inference",
                return_value=nullcontext(13),
        ) as seeded:
            output = model._generate(
                "[S1] Generate.",
                max_tokens=16,
                cfg_scale=2.5,
            )

        seeded.assert_called_once_with(
            None,
            device="cpu",
            model_type="dia",
        )
        self.assertEqual(output.audio, [0.25, -0.25])
        self.assertEqual(output.metadata["backend"], "transformers")
        self.assertEqual(
            runtime.calls[0]["max_new_tokens"],
            16,
        )
        self.assertEqual(
            runtime.calls[0]["guidance_scale"],
            2.5,
        )
        self.assertEqual(
            processor.calls[0]["text"],
            ["[S1] Generate."],
        )
        self.assertEqual(
            processor.decode_calls[0],
            ("generated-codes", {
                "audio_prompt_len": None,
            }),
        )

    def test_transformers_export_cannot_overwrite_voicehub_config(self):

        class NativeBackend:

            def save_pretrained(self, directory):
                destination = Path(directory)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "config.json").write_text(
                    json.dumps({
                        "model_type": "dia-native",
                    }),
                    encoding="utf-8",
                )

        model = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B-0626",
            device="cpu",
        )
        model._loaded_backend = "transformers"
        model._dia_backend = NativeBackend()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            root_config = json.loads(Path(directory, "config.json").read_text(encoding="utf-8"))
            native_config = json.loads(
                Path(
                    directory,
                    NATIVE_EXPORT_DIR,
                    "config.json",
                ).read_text(encoding="utf-8"))

        self.assertEqual(root_config["model_type"], "dia")
        self.assertEqual(native_config["model_type"], "dia-native")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Dia extra")
class DiaTrainingBackendTests(unittest.TestCase):

    def test_loader_uses_safetensors_and_freezes_dac(self):
        import torch

        loaded = {}
        processor = FakeProcessor(audio_tokenizer=torch.nn.Linear(1, 1))

        class FakeDiaModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))
                self.config = SimpleNamespace(use_cache=True)

            def forward(self, input_ids, labels, **kwargs):
                del input_ids, kwargs
                return SimpleNamespace(loss=(self.scale - labels.float().mean()).square(), )

        class FakeModelFactory:

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                loaded["model_name"] = name
                loaded["model_kwargs"] = kwargs
                loaded["model"] = FakeDiaModel()
                return loaded["model"]

        class FakeProcessorFactory:

            @classmethod
            def from_pretrained(cls, name):
                loaded["processor_name"] = name
                return processor

        transformers = SimpleNamespace(
            __version__="4.57.0",
            DiaForConditionalGeneration=FakeModelFactory,
            AutoProcessor=FakeProcessorFactory,
        )

        def optional_dependency(name, **kwargs):
            del kwargs
            if name == "torch":
                return torch
            if name == "transformers":
                return transformers
            raise AssertionError(name)

        with patch.object(
                dia_training,
                "import_optional",
                side_effect=optional_dependency,
        ):
            backend = load_dia_transformers_backend(
                "nari-labs/Dia-1.6B-0626",
                device="cpu",
                compute_dtype="bfloat16",
                for_training=True,
            )

        self.assertEqual(
            loaded["model_name"],
            "nari-labs/Dia-1.6B-0626",
        )
        self.assertEqual(
            loaded["processor_name"],
            "nari-labs/Dia-1.6B-0626",
        )
        self.assertTrue(loaded["model_kwargs"]["use_safetensors"])
        self.assertIs(loaded["model_kwargs"]["torch_dtype"], torch.float32)
        self.assertFalse(backend.model.config.use_cache)
        self.assertFalse(processor.audio_tokenizer.training)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in processor.audio_tokenizer.parameters()))

        loss = backend.forward_loss(
            input_ids=torch.tensor([[1, 2]]),
            labels=torch.tensor([[1, 1]]),
        )
        loss.backward()
        self.assertIsNotNone(backend.model.scale.grad)
        self.assertTrue(all(parameter.grad is None for parameter in processor.audio_tokenizer.parameters()))

    def test_native_adapter_backpropagates_official_loss(self):
        import torch

        class NativeDia(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))
                self.config = SimpleNamespace(use_cache=False)

            def forward(self, input_ids, labels, **kwargs):
                del input_ids, kwargs
                logits = self.scale.expand(1, 1, 2)
                loss = (self.scale - labels.float().mean()).square()
                return SimpleNamespace(loss=loss, logits=logits)

        processor = FakeProcessor(audio_tokenizer=torch.nn.Linear(1, 1))
        runtime = NativeDia()
        backend = DiaTrainingBackend(
            model=runtime,
            processor=processor,
            sample_rate=44_100,
        )
        wrapper = DiaForTextToSpeech(
            model_path="nari-labs/Dia-1.6B-0626",
            device="cpu",
        )
        wrapper.model = runtime
        wrapper._dia_backend = backend
        wrapper._loaded_backend = "transformers"
        spec = ModelTrainingSpec(
            model_type="dia",
            family=TrainingFamily.SEQ2SEQ,
            module_paths=("model", ),
            support=TrainingSupport.NATIVE,
            native_training=True,
        )
        adapter = DiaTrainingAdapter(wrapper, spec)
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "decoder_input_ids": torch.tensor([[[1026] * 9]]),
            "decoder_attention_mask": torch.tensor([[1]]),
            "labels": torch.tensor([[1]]),
        }

        output = adapter(**inputs)
        output.loss.backward()

        self.assertEqual(output.loss.ndim, 0)
        self.assertIsNotNone(runtime.scale.grad)
        self.assertTrue(all(parameter.grad is None for parameter in processor.audio_tokenizer.parameters()))

    def test_backend_exports_transformers_safetensors(self):
        import torch

        class SaveableModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.saved = None

            def save_pretrained(
                self,
                directory,
                *,
                safe_serialization=False,
            ):
                self.saved = (Path(directory), safe_serialization)

        model = SaveableModel()
        processor = FakeProcessor(audio_tokenizer=torch.nn.Linear(1, 1), )
        backend = DiaTrainingBackend(
            model=model,
            processor=processor,
            sample_rate=44_100,
            transformers_major_version=4,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = backend.save_pretrained(directory)
            self.assertEqual(output, Path(directory))
            self.assertEqual(
                model.saved,
                (Path(directory), True),
            )
            self.assertEqual(processor.saved_to, Path(directory))


if __name__ == "__main__":
    unittest.main()
