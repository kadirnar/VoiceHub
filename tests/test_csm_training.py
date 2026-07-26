import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import voicehub.models.csm.training as csm_training
from voicehub.models.csm.inference import CSMForTextToSpeech
from voicehub.models.csm.training import CSMTrainingBackend, CSMTrainingCollator, prepare_csm_training_inputs
from voicehub.trainer import Trainer
from voicehub.training_args import TrainingArguments

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeProcessor:

    def __init__(self, sample_rate=24_000):
        self.feature_extractor = SimpleNamespace(sampling_rate=sample_rate)
        self.calls = []
        self.saved_to = None

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        return {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
            "input_values": [[[0.1, 0.2]]],
            "input_values_cutoffs": [[2]],
            "labels": [[-100, 1]],
        }

    def save_pretrained(self, directory):
        self.saved_to = Path(directory)


class FakeGenerationBatch(dict):

    def __init__(self, values):
        super().__init__(values)
        self.device = None

    def to(self, device):
        self.device = device
        return self


class FakeGenerationProcessor(FakeProcessor):

    def __init__(self, sample_rate=24_000):
        super().__init__(sample_rate=sample_rate)
        self.generation_calls = []
        self.last_batch = None

    def apply_chat_template(self, conversation, **kwargs):
        self.generation_calls.append((conversation, kwargs))
        self.last_batch = FakeGenerationBatch({
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
        })
        return self.last_batch


class CSMTrainingInputTests(unittest.TestCase):

    def test_training_module_keeps_heavy_dependencies_lazy(self):
        command = (
            "import sys; "
            "import voicehub.models.csm.training; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False False")

    def test_collator_matches_official_grouped_conversation_recipe(self):
        processor = FakeProcessor()
        collator = CSMTrainingCollator(processor)
        concatenated = np.arange(12, dtype=np.float32)

        batch = collator([
            {
                "speaker_ids": [0, 1],
                "texts": ["first", "second"],
                "audio": {
                    "array": concatenated,
                    "sampling_rate": 24_000,
                },
                "audio_cut_idxs": [(0, 5), (5, 12)],
            },
            {
                "speaker_id": 2,
                "text": "third",
                "audio": {
                    "array": np.arange(4, dtype=np.float32),
                    "sampling_rate": 24_000,
                },
            },
        ])

        self.assertIn("labels", batch)
        conversations, kwargs = processor.calls[0]
        self.assertEqual(len(conversations), 2)
        self.assertEqual(
            [message["role"] for message in conversations[0]],
            ["0", "1"],
        )
        np.testing.assert_array_equal(
            conversations[0][0]["content"][1]["audio"],
            concatenated[:5],
        )
        np.testing.assert_array_equal(
            conversations[0][1]["content"][1]["audio"],
            concatenated[5:],
        )
        self.assertEqual(conversations[1][0]["role"], "2")
        self.assertEqual(
            kwargs,
            {
                "tokenize": True,
                "return_dict": True,
                "output_labels": True,
                "depth_decoder_labels_ratio": 1.0,
            },
        )

    def test_prepared_inputs_pass_through_without_reprocessing(self):
        processor = FakeProcessor()
        prepared = {
            "input_ids": object(),
            "labels": object(),
            "input_values": object(),
        }

        output = prepare_csm_training_inputs(processor, prepared)

        self.assertEqual(output, prepared)
        self.assertIsNot(output, prepared)
        self.assertEqual(processor.calls, [])

    def test_columnar_inputs_are_converted_to_batched_conversations(self):
        processor = FakeProcessor()

        prepare_csm_training_inputs(
            processor,
            {
                "text": ["alpha", "beta"],
                "speaker_id": [3, 4],
                "audio": [
                    np.ones(3, dtype=np.float32),
                    np.ones(5, dtype=np.float32),
                ],
            },
        )

        conversations, _ = processor.calls[0]
        self.assertEqual(
            [conversation[0]["role"] for conversation in conversations],
            ["3", "4"],
        )
        self.assertEqual(
            [conversation[0]["content"][0]["text"] for conversation in conversations],
            ["alpha", "beta"],
        )

    def test_audio_sample_rate_mismatch_is_actionable(self):
        processor = FakeProcessor()
        collator = CSMTrainingCollator(processor)

        with self.assertRaisesRegex(ValueError, "resampled to 24000 Hz"):
            collator([{
                "text": "wrong rate",
                "audio": {
                    "array": np.ones(4, dtype=np.float32),
                    "sampling_rate": 16_000,
                },
            }])

    def test_depth_decoder_ratio_is_validated(self):
        with self.assertRaisesRegex(
                ValueError,
                "depth_decoder_labels_ratio",
        ):
            CSMTrainingCollator(
                FakeProcessor(),
                depth_decoder_labels_ratio=1.1,
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional CSM extra")
class CSMTrainingBackendTests(unittest.TestCase):

    def test_loader_uses_safetensors_freezes_codec_and_returns_native_loss(self):
        import torch

        processor = FakeProcessor()

        class FakeCsmModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))
                self.codec_model = torch.nn.Linear(1, 1)
                self.config = SimpleNamespace(use_cache=True)
                self.codec_training_during_forward = None

            def forward(self, input_ids, labels, **kwargs):
                del input_ids, kwargs
                self.codec_training_during_forward = self.codec_model.training
                target = labels.float().mean()
                return SimpleNamespace(loss=(self.scale - target).square())

        loaded = {}

        class FakeModelFactory:

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                loaded["model_name"] = name
                loaded["model_kwargs"] = kwargs
                loaded["model"] = FakeCsmModel()
                return loaded["model"]

        class FakeProcessorFactory:

            @classmethod
            def from_pretrained(cls, name):
                loaded["processor_name"] = name
                return processor

        transformers = SimpleNamespace(
            __version__="5.1.0",
            CsmForConditionalGeneration=FakeModelFactory,
            CsmProcessor=FakeProcessorFactory,
        )

        def optional_dependency(name, **kwargs):
            del kwargs
            if name == "torch":
                return torch
            if name == "transformers":
                return transformers
            raise AssertionError(name)

        with patch.object(
                csm_training,
                "import_optional",
                side_effect=optional_dependency,
        ):
            backend = csm_training.load_csm_training_backend(
                "sesame/csm-1b",
                device="cpu",
                torch_dtype="bfloat16",
            )

        self.assertEqual(loaded["model_name"], "sesame/csm-1b")
        self.assertEqual(loaded["processor_name"], "sesame/csm-1b")
        self.assertTrue(loaded["model_kwargs"]["use_safetensors"])
        self.assertIs(loaded["model_kwargs"]["dtype"], torch.float32)
        self.assertNotIn("torch_dtype", loaded["model_kwargs"])
        self.assertFalse(backend.model.config.use_cache)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in backend.model.codec_model.parameters()))
        self.assertFalse(backend.model.codec_model.training)

        backend.model.train()
        self.assertTrue(backend.model.codec_model.training)
        loss = backend.forward_loss(
            input_ids=torch.tensor([[1, 2]]),
            labels=torch.tensor([[2, 2]]),
        )

        self.assertEqual(loss.ndim, 0)
        self.assertFalse(backend.model.codec_training_during_forward)
        self.assertFalse(backend.model.codec_model.training)
        loss.backward()
        self.assertIsNotNone(backend.model.scale.grad)
        self.assertTrue(all(parameter.grad is None for parameter in backend.model.codec_model.parameters()))

    def test_scalar_loss_requires_exactly_one_native_value(self):
        import torch

        with self.assertRaisesRegex(RuntimeError, "returned no loss"):
            CSMTrainingBackend.scalar_loss({})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            CSMTrainingBackend.scalar_loss({
                "loss": torch.ones(2),
            })

    def test_backend_exports_safe_serialization_for_transformers_v4(self):
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
        processor = FakeProcessor()
        backend = CSMTrainingBackend(
            model=model,
            processor=processor,
            sample_rate=24_000,
            transformers_major_version=4,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = backend.save_pretrained(directory)
            self.assertEqual(output, Path(directory))
            self.assertEqual(model.saved, (Path(directory), True))
            self.assertEqual(processor.saved_to, Path(directory))

    def test_trainer_artifact_reload_generates_with_transformers_backend(self):
        import torch

        class ArtifactCsmModel(torch.nn.Module):

            def __init__(self, scale):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(float(scale)))
                self.codec_model = torch.nn.Linear(1, 1)
                self.config = SimpleNamespace(use_cache=False)
                self.generation_calls = []
                self.saved_to = None

            def forward(self, input_ids, labels=None, **kwargs):
                del input_ids, kwargs
                target = (
                    torch.zeros((), dtype=self.scale.dtype) if labels is None else labels.float().mean())
                return SimpleNamespace(
                    loss=(self.scale - target).square(),
                    logits=self.scale.expand(1, 1, 1),
                )

            def generate(self, **kwargs):
                self.generation_calls.append(kwargs)
                return [self.scale.detach().reshape(1)]

            def save_pretrained(
                self,
                directory,
                *,
                safe_serialization=False,
            ):
                self.saved_to = (
                    Path(directory),
                    safe_serialization,
                )

        source_processor = FakeGenerationProcessor()
        source_runtime = ArtifactCsmModel(0.75)
        source_backend = CSMTrainingBackend(
            model=source_runtime,
            processor=source_processor,
            sample_rate=24_000,
            transformers_major_version=4,
        )
        source_wrapper = CSMForTextToSpeech(device="cpu")
        source_wrapper.model = source_runtime
        source_wrapper._training_backend = source_backend
        source_wrapper.config.sample_rate = source_backend.sample_rate

        restored_processor = FakeGenerationProcessor()
        restored_runtime = ArtifactCsmModel(-1.0)
        restored_backend = CSMTrainingBackend(
            model=restored_runtime,
            processor=restored_processor,
            sample_rate=24_000,
            transformers_major_version=4,
        )

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=source_wrapper,
                args=TrainingArguments(
                    output_dir=directory,
                    use_cpu=True,
                ),
            )
            trainer._ensure_model_loaded()
            trainer._move_model_to_device()
            trainer.save_model(directory)
            self.assertTrue((Path(directory) / "model_state.pt").is_file())

            restored = CSMForTextToSpeech.from_pretrained(
                directory,
                device="cpu",
            )
            with patch(
                    "voicehub.models.csm.training.load_csm_training_backend",
                    return_value=restored_backend,
            ) as loader:
                output = restored.generate(
                    "Portable CSM.",
                    speaker=3,
                    max_audio_length_ms=800,
                    temperature=0.8,
                    top_k=25,
                )

        loader.assert_called_once_with(
            "sesame/csm-1b",
            device="cpu",
            torch_dtype="bfloat16",
        )
        self.assertAlmostEqual(
            restored_runtime.scale.item(),
            0.75,
        )
        self.assertAlmostEqual(
            output.audio.item(),
            0.75,
        )
        self.assertEqual(output.metadata["backend"], "transformers")
        self.assertEqual(output.metadata["speaker"], 3)
        self.assertEqual(output.metadata["context_segments"], 0)
        conversation, processor_kwargs = (restored_processor.generation_calls[0])
        self.assertEqual(
            conversation,
            [{
                "role": "3",
                "content": [{
                    "type": "text",
                    "text": "Portable CSM.",
                }],
            }],
        )
        self.assertEqual(
            processor_kwargs,
            {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(restored_processor.last_batch.device, "cpu")
        generation_call = restored_runtime.generation_calls[0]
        self.assertTrue(generation_call["output_audio"])
        self.assertEqual(generation_call["max_new_tokens"], 10)
        self.assertEqual(generation_call["temperature"], 0.8)
        self.assertEqual(generation_call["top_k"], 25)
        self.assertEqual(
            generation_call["depth_decoder_temperature"],
            0.8,
        )
        self.assertEqual(generation_call["depth_decoder_top_k"], 25)


class CSMWrapperBackendSelectionTests(unittest.TestCase):

    def test_training_load_selects_transformers_backend(self):
        backend = SimpleNamespace(
            model=object(),
            processor=object(),
            sample_rate=24_000,
            prepare_inputs=Mock(return_value={"labels": "prepared"}),
        )
        model = CSMForTextToSpeech(device="cpu")

        with patch(
                "voicehub.models.csm.training.load_csm_training_backend",
                return_value=backend,
        ) as loader:
            model._loading_for_training = True
            try:
                model._load_pretrained_model()
            finally:
                model._loading_for_training = False

        loader.assert_called_once_with(
            "sesame/csm-1b",
            device="cpu",
            torch_dtype="bfloat16",
        )
        self.assertIs(model.model, backend.model)
        self.assertIs(model.training_backend, backend)
        self.assertEqual(
            model.prepare_training_inputs({}, phase="default"),
            {"labels": "prepared"},
        )

    def test_existing_inference_runtime_is_reloaded_for_training(self):
        backend = SimpleNamespace(model=object())
        model = CSMForTextToSpeech(device="cpu")
        model.model = object()

        def load_training_runtime():
            self.assertTrue(model.is_training_load)
            model.model = backend.model
            model._training_backend = backend
            return model

        with patch.object(
                model,
                "load",
                side_effect=load_training_runtime,
        ) as load:
            model._prepare_for_training()

        load.assert_called_once_with()
        self.assertIs(model.model, backend.model)

    def test_normal_load_keeps_vendored_inference_generator(self):
        half = object()
        bfloat = object()
        full = object()
        fake_torch = SimpleNamespace(
            float16=half,
            bfloat16=bfloat,
            float32=full,
        )

        class SourceModel:

            def __init__(self):
                self.to_call = None

            def to(self, **kwargs):
                self.to_call = kwargs

        source_model = SourceModel()

        class SourceModelFactory:

            @classmethod
            def from_pretrained(cls, name):
                self.assertEqual(name, "sesame/csm-1b")
                return source_model

        generator = SimpleNamespace(sample_rate=24_000)
        runtime = SimpleNamespace(Generator=Mock(return_value=generator))
        modules = {
            "torch": fake_torch,
            "torchaudio": object(),
            "voicehub.models.csm.source.csm.generator": runtime,
            "voicehub.models.csm.source.csm.models": SimpleNamespace(Model=SourceModelFactory),
        }

        model = CSMForTextToSpeech(device="cpu")
        with patch(
                "voicehub.models.csm.inference.import_optional",
                side_effect=lambda name, **kwargs: modules[name],
        ):
            model._load_pretrained_model()

        runtime.Generator.assert_called_once_with(source_model)
        self.assertIs(model.model, generator)
        self.assertIsNone(model.training_backend)
        self.assertEqual(
            source_model.to_call,
            {
                "device": "cpu",
                "dtype": full,
            },
        )


if __name__ == "__main__":
    unittest.main()
