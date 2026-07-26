import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.models.fishtts.inference import FishTTSConfig, FishTTSForTextToSpeech

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class FishTTSTrainingRuntimeTests(unittest.TestCase):

    @staticmethod
    def _semantic_model():
        import torch

        class Attention(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.kv_cache = None

        class Layer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.attention = Attention()

        class SemanticModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = SimpleNamespace(
                    max_seq_len=128,
                    use_cache=False,
                )
                self.layers = torch.nn.ModuleList([Layer()])
                self.fast_layers = torch.nn.ModuleList([Layer()])
                self.max_batch_size = -1
                self.max_seq_len = -1
                self._cache_setup_done = False

            def setup_caches(
                self,
                *,
                max_batch_size,
                max_seq_len,
                dtype,
            ):
                del dtype
                self.max_batch_size = max_batch_size
                self.max_seq_len = max_seq_len
                for layer in (*self.layers, *self.fast_layers):
                    layer.attention.kv_cache = torch.nn.Identity()

        return SemanticModel()

    @staticmethod
    def _runtime(torch, codec):
        decoder = object()

        def prepare_model_for_inference(model, device, *, compile):
            del device, compile
            model.eval()
            return decoder

        def generate_long(**kwargs):
            del kwargs
            yield SimpleNamespace(
                action="sample",
                codes=torch.tensor([[1, 2]]),
            )
            yield SimpleNamespace(action="next", codes=None)

        return (
            SimpleNamespace(
                prepare_model_for_inference=Mock(side_effect=prepare_model_for_inference, ),
                load_codec_model=Mock(return_value=codec),
                generate_long=generate_long,
                decode_to_audio=Mock(return_value=torch.tensor([0.25, -0.25]), ),
            ),
            decoder,
        )

    def test_training_generation_training_preserves_optimizer_owned_model(self):
        import torch

        semantic_model = self._semantic_model()
        optimizer = torch.optim.SGD(semantic_model.parameters(), lr=0.1)
        optimizer_parameter = optimizer.param_groups[0]["params"][0]
        codec = SimpleNamespace(spec_transform=SimpleNamespace(sample_rate=32_000), )
        runtime, decoder = self._runtime(torch, codec)

        wrapper = FishTTSForTextToSpeech(
            FishTTSConfig(name_or_path="test/fish"),
            device="cpu",
        )
        wrapper.model = semantic_model
        wrapper._torch = torch
        wrapper._model_directory = Path("/test/fish")
        wrapper._loaded_for_training = True

        with patch(
                "voicehub.models.fishtts.inference.import_optional",
                return_value=runtime,
        ):
            output = wrapper.generate("hello")

            self.assertIs(wrapper.model, semantic_model)
            self.assertIs(optimizer_parameter, wrapper.model.weight)
            self.assertFalse(wrapper._loaded_for_training)
            self.assertFalse(wrapper.model.training)
            self.assertIs(wrapper._runtime, runtime)
            self.assertIs(wrapper._decode_one_token, decoder)
            self.assertIs(wrapper._codec, codec)
            self.assertEqual(wrapper.sample_rate, 32_000)
            self.assertTrue(wrapper.model._cache_setup_done)
            self.assertIsNotNone(wrapper.model.layers[0].attention.kv_cache, )
            torch.testing.assert_close(
                output.audio,
                torch.tensor([0.25, -0.25]),
            )

            wrapper.load_for_training()

            self.assertIs(wrapper.model, semantic_model)
            self.assertIs(optimizer_parameter, wrapper.model.weight)
            self.assertTrue(wrapper._loaded_for_training)
            self.assertTrue(wrapper.model.training)
            self.assertIsNone(wrapper._runtime)
            self.assertIsNone(wrapper._decode_one_token)
            self.assertFalse(wrapper.model._cache_setup_done)
            self.assertEqual(wrapper.model.max_batch_size, -1)
            self.assertEqual(wrapper.model.max_seq_len, -1)
            self.assertIsNone(wrapper.model.layers[0].attention.kv_cache, )
            self.assertIsNone(wrapper.model.fast_layers[0].attention.kv_cache, )

            wrapper.load()

        self.assertIs(wrapper.model, semantic_model)
        self.assertIs(optimizer_parameter, wrapper.model.weight)
        self.assertFalse(wrapper._loaded_for_training)
        self.assertEqual(
            runtime.prepare_model_for_inference.call_count,
            2,
        )
        runtime.load_codec_model.assert_called_once_with(
            Path("/test/fish/codec.pth"),
            "cpu",
            torch.float32,
        )

    def test_failed_serving_attachment_rolls_back_to_training_state(self):
        import torch

        semantic_model = self._semantic_model()
        codec_error = RuntimeError("codec unavailable")
        runtime = SimpleNamespace(
            prepare_model_for_inference=Mock(
                side_effect=lambda model, device, compile: (
                    model.eval(),
                    object(),
                )[-1],
            ),
            load_codec_model=Mock(side_effect=codec_error),
        )
        wrapper = FishTTSForTextToSpeech(
            FishTTSConfig(name_or_path="test/fish"),
            device="cpu",
        )
        wrapper.model = semantic_model
        wrapper._torch = torch
        wrapper._model_directory = Path("/test/fish")
        wrapper._loaded_for_training = True

        with (
                patch(
                    "voicehub.models.fishtts.inference.import_optional",
                    return_value=runtime,
                ),
                self.assertRaisesRegex(RuntimeError, "codec unavailable"),
        ):
            wrapper.load()

        self.assertIs(wrapper.model, semantic_model)
        self.assertTrue(wrapper._loaded_for_training)
        self.assertTrue(wrapper.model.training)
        self.assertIsNone(wrapper._runtime)
        self.assertIsNone(wrapper._decode_one_token)
        self.assertFalse(wrapper.model._cache_setup_done)
        self.assertEqual(wrapper.model.max_seq_len, -1)


if __name__ == "__main__":
    unittest.main()
