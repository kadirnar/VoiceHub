import importlib.util
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub import Trainer, TrainingArguments
from voicehub.models.conversationtts.inference import (
    ConversationTTSConfig,
    ConversationTTSForTextToSpeech,
)
from voicehub.models.conversationtts.runtime import resume_for_inference


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class ConversationTTSCheckpointTests(unittest.TestCase):

    def test_checkpoint_uses_restricted_torch_loader(self):
        torch = SimpleNamespace(
            load=Mock(return_value={
                "model": {
                    "module.weight": 3,
                },
            }),
        )
        model = Mock()
        with patch(
            "voicehub.models.conversationtts.runtime.import_optional",
            return_value=torch,
        ):
            resume_for_inference("/checkpoint.pt", None, model, "cpu")

        torch.load.assert_called_once_with(
            "/checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict.assert_called_once_with({"weight": 3})

    def test_checkpoint_only_falls_back_for_unsupported_weights_only(self):
        torch = SimpleNamespace(
            load=Mock(side_effect=[
                TypeError("weights_only is unsupported"),
                {
                    "model": {
                        "weight": 4,
                    },
                },
            ]),
        )
        model = Mock()
        with patch(
            "voicehub.models.conversationtts.runtime.import_optional",
            return_value=torch,
        ):
            resume_for_inference("/checkpoint.pt", None, model, "cpu")

        self.assertEqual(
            torch.load.call_args_list,
            [
                unittest.mock.call(
                    "/checkpoint.pt",
                    map_location="cpu",
                    weights_only=True,
                ),
                unittest.mock.call(
                    "/checkpoint.pt",
                    map_location="cpu",
                ),
            ],
        )
        model.load_state_dict.assert_called_once_with({"weight": 4})

    def test_checkpoint_does_not_retry_unsafe_content_failures(self):
        torch = SimpleNamespace(
            load=Mock(side_effect=RuntimeError("unsafe checkpoint")),
        )
        with (
            patch(
                "voicehub.models.conversationtts.runtime.import_optional",
                return_value=torch,
            ),
            self.assertRaisesRegex(RuntimeError, "unsafe checkpoint"),
        ):
            resume_for_inference(
                "/checkpoint.pt",
                None,
                Mock(),
                "cpu",
            )
        self.assertEqual(torch.load.call_count, 1)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class ConversationTTSTrainingRuntimeTests(unittest.TestCase):

    def setUp(self):
        import torch

        self.torch = torch
        self.events = []
        events = self.events

        class FakeCache(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.register_buffer("values", torch.ones(1))

        class FakeAttention(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.kv_cache = None
                self.cache_enabled = False

        class FakeLayer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.attn = FakeAttention()

        class FakeTransformer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([FakeLayer()])

            def setup_caches(self, *_args, **_kwargs):
                self.layers[0].attn.kv_cache = FakeCache()
                self.layers[0].attn.cache_enabled = True

            def caches_are_setup(self):
                return self.layers[0].attn.kv_cache is not None

            def caches_are_enabled(self):
                return self.layers[0].attn.cache_enabled

        class FakeModel(torch.nn.Module):

            def __init__(self, config):
                super().__init__()
                self.config = config
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.backbone = FakeTransformer()
                self.decoder = FakeTransformer()

            def setup_caches(self, max_batch_size):
                events.append(("setup_caches", max_batch_size))
                self.backbone.setup_caches(max_batch_size)
                self.decoder.setup_caches(max_batch_size)
                self.register_buffer(
                    "backbone_causal_mask",
                    torch.ones(1, dtype=torch.bool),
                )
                self.register_buffer(
                    "decoder_causal_mask",
                    torch.ones(1, dtype=torch.bool),
                )

            def forward(
                self,
                tokens,
                labels,
                tokens_mask,
                input_pos=None,
            ):
                del tokens_mask, input_pos
                batch_size, sequence_length, _ = tokens.shape
                c0_logits = self.weight.expand(
                    batch_size,
                    sequence_length - 1,
                    4,
                )
                residual_labels = labels[..., 1:].reshape(-1, 1)
                residual_logits = self.weight.expand(
                    residual_labels.shape[0],
                    1,
                    4,
                )
                return c0_logits, residual_logits, residual_labels

        class FakeGenerator:

            def __init__(
                self,
                model,
                text_tokenizer_path,
                audio_tokenizer_path,
            ):
                events.append((
                    "generator",
                    text_tokenizer_path,
                    audio_tokenizer_path,
                ))
                self._model = model
                model.setup_caches(1)
                self._text_tokenizer = object()
                self._audio_tokenizer = object()

            def generate_v1(self, **_kwargs):
                return self._model.weight.detach().reshape(1)

        self.model_module = SimpleNamespace(
            Model=FakeModel,
            ModelArgs=lambda **values: SimpleNamespace(**values),
        )
        self.generator_module = SimpleNamespace(
            Generator=FakeGenerator,
            prepare_prompt=Mock(),
        )
        self.loss_module = SimpleNamespace(
            CrossEntropyAndAccuracy_zero=(
                lambda logits, labels, mask, ignore_id=0: (
                    logits.mean(),
                    {
                        "zero_loss": logits.detach().mean(),
                    },
                )
            ),
            CrossEntropyAndAccuracy_residual=(
                lambda logits, labels, loss_weights, ignore_id=0: (
                    logits.mean(),
                    {
                        "residual_loss": logits.detach().mean(),
                    },
                )
            ),
        )

    def _model(self):
        return ConversationTTSForTextToSpeech(
            ConversationTTSConfig(
                name_or_path="test/conversationtts",
                model_args={
                    "backbone_flavor": "test",
                    "decoder_flavor": "test",
                    "text_vocab_size": 8,
                    "audio_vocab_size": 4,
                    "audio_num_codebooks": 2,
                },
                torch_dtype="float32",
            ),
            device="cpu",
        )

    @contextmanager
    def _patched_runtime(self):
        imported = []

        def import_runtime(name, **_kwargs):
            imported.append(name)
            if name == "torch":
                return self.torch
            if name.endswith("models.model_new"):
                return self.model_module
            if name.endswith("inference.generator"):
                return self.generator_module
            raise AssertionError(f"Unexpected ConversationTTS import: {name}")

        def restore_checkpoint(_checkpoint, _experiment, model, _device):
            model.weight.data.fill_(1.0)

        with (
            patch(
                "voicehub.models.conversationtts."
                "modeling_conversationtts.import_optional",
                side_effect=import_runtime,
            ),
            patch(
                "voicehub.models.conversationtts."
                "modeling_conversationtts.resume_for_inference",
                side_effect=restore_checkpoint,
            ),
            patch.object(
                ConversationTTSForTextToSpeech,
                "_checkpoint_path",
                return_value=Path("/unused/checkpoint.pt"),
            ),
            patch.object(
                ConversationTTSForTextToSpeech,
                "_text_tokenizer_path",
                return_value=Path("/unused/text-tokenizer"),
            ) as text_tokenizer,
            patch.object(
                ConversationTTSForTextToSpeech,
                "_audio_tokenizer_path",
                return_value=Path("/unused/audio-tokenizer.safetensors"),
            ) as audio_tokenizer,
        ):
            yield SimpleNamespace(
                imported=imported,
                text_tokenizer=text_tokenizer,
                audio_tokenizer=audio_tokenizer,
            )

    @contextmanager
    def _patched_training_loss(self):

        def import_source(name, **_kwargs):
            if name.endswith("models.model_new"):
                return self.loss_module
            raise AssertionError(
                f"Unexpected ConversationTTS recipe import: {name}"
            )

        with patch(
            "voicehub.training.recipes.import_optional",
            side_effect=import_source,
        ):
            yield

    def _batch(self):
        return {
            "tokens": self.torch.zeros((1, 4, 3), dtype=self.torch.long),
            "labels": self.torch.ones((1, 3, 2), dtype=self.torch.long),
            "tokens_mask": self.torch.ones((1, 4, 3), dtype=self.torch.bool),
        }

    def assert_training_graph_is_cache_free(self, model):
        self.assertFalse(model.backbone.caches_are_setup())
        self.assertFalse(model.backbone.caches_are_enabled())
        self.assertFalse(model.decoder.caches_are_setup())
        self.assertFalse(model.decoder.caches_are_enabled())
        self.assertFalse(hasattr(model, "backbone_causal_mask"))
        self.assertFalse(hasattr(model, "decoder_causal_mask"))

    def test_cold_training_load_skips_all_serving_allocations(self):
        model = self._model()
        with self._patched_runtime() as runtime:
            model.load_for_training()

        self.assertTrue(model._loaded_for_training)
        self.assertTrue(model.model.training)
        self.assertIsNone(model._generator)
        self.assertIsNone(model._generator_module)
        self.assert_training_graph_is_cache_free(model.model)
        self.assertNotIn(
            "voicehub.models.conversationtts.source.conversationtts."
            "inference.generator",
            runtime.imported,
        )
        runtime.text_tokenizer.assert_not_called()
        runtime.audio_tokenizer.assert_not_called()
        self.assertNotIn(("setup_caches", 1), self.events)

    def test_lifecycle_preserves_weights_gradients_and_adapter_identity(self):
        model = self._model()
        with self._patched_runtime(), self._patched_training_loss():
            adapter = model.get_training_adapter()
            first_output = adapter(**self._batch())
            first_output.loss.backward()
            self.assertIsNotNone(model.model.weight.grad)

            training_model = model.model
            model.model.weight.data.fill_(7.0)
            model.model.weight.grad = None

            model.load()
            self.assertIs(model.model, training_model)
            self.assertFalse(model._loaded_for_training)
            self.assertIsNotNone(model._generator)
            self.assertTrue(model.model.backbone.caches_are_setup())
            self.assertTrue(model.model.decoder.caches_are_setup())
            self.assertEqual(model.model.weight.item(), 7.0)

            model.load_for_training()
            self.assertIs(model.model, training_model)
            self.assertIs(adapter.primary_model, training_model)
            self.assertTrue(model._loaded_for_training)
            self.assertIsNone(model._generator)
            self.assertTrue(model.model.training)
            self.assert_training_graph_is_cache_free(model.model)
            self.assertEqual(model.model.weight.item(), 7.0)

            second_output = adapter(**self._batch())
            second_output.loss.backward()
            self.assertIsNotNone(model.model.weight.grad)

            generated = model.generate("preserve my weights")

        self.assertEqual(float(generated.audio[0]), 7.0)
        self.assertEqual(
            [event for event in self.events if event[0] == "setup_caches"],
            [
                ("setup_caches", 1),
                ("setup_caches", 1),
            ],
        )

    def test_portable_artifact_restores_then_builds_inference_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._model()
            with self._patched_runtime():
                trainer = Trainer(
                    model=source,
                    args=TrainingArguments(
                        output_dir=directory,
                        use_cpu=True,
                    ),
                )
                trainer._ensure_model_loaded()
                trainer._move_model_to_device()
                source.model.weight.data.fill_(5.0)
                trainer.save_model(directory)

                restored = ConversationTTSForTextToSpeech.from_pretrained(
                    directory,
                    device="cpu",
                    lazy_load=False,
                )
                generated = restored.generate("portable artifact")

                self.assertFalse(restored._loaded_for_training)
                self.assertIsNotNone(restored._generator)
                self.assertTrue(restored.model.backbone.caches_are_setup())
                self.assertEqual(restored.model.weight.item(), 5.0)
                self.assertEqual(float(generated.audio[0]), 5.0)

                restored.load_for_training()
                self.assertTrue(restored._loaded_for_training)
                self.assert_training_graph_is_cache_free(restored.model)
                self.assertEqual(restored.model.weight.item(), 5.0)


if __name__ == "__main__":
    unittest.main()
