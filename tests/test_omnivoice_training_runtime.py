import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from voicehub.models.omnivoice.inference import (
    OmniVoiceConfig,
    OmniVoiceForTextToSpeech,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class OmniVoiceTrainingRuntimeTests(unittest.TestCase):

    @staticmethod
    def _fake_loader(wrapper):
        import torch

        class Runtime(torch.nn.Module):

            def __init__(self, *, training):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.text_tokenizer = None if training else object()
                self.audio_tokenizer = None if training else object()
                self.sampling_rate = 24_000

            def forward(self, input_ids, labels=None):
                logits = input_ids.float() * self.weight
                loss = (
                    (logits - labels).square().mean()
                    if labels is not None
                    else None
                )
                return {"loss": loss, "logits": logits}

            def generate(self, **kwargs):
                del kwargs
                return [self.weight.detach().reshape(1)]

        wrapper.model = Runtime(training=wrapper.is_training_load)
        wrapper._loaded_for_training = wrapper.is_training_load
        wrapper.config.sample_rate = 24_000

    def test_runtime_switches_preserve_fine_tuned_weights(self):
        wrapper = OmniVoiceForTextToSpeech(
            OmniVoiceConfig(name_or_path="test/omnivoice"),
            device="cpu",
        )
        with patch.object(
            OmniVoiceForTextToSpeech,
            "_load_pretrained_model",
            autospec=True,
            side_effect=self._fake_loader,
        ):
            wrapper.load_for_training()
            wrapper.model.weight.data.fill_(7.0)
            optimizer_owned_model = wrapper.model

            wrapper.load()
            self.assertFalse(wrapper._loaded_for_training)
            self.assertIs(wrapper.model, optimizer_owned_model)
            self.assertIsNotNone(wrapper.model.text_tokenizer)
            self.assertEqual(wrapper.model.weight.item(), 7.0)

            wrapper.load_for_training()
            self.assertTrue(wrapper._loaded_for_training)
            self.assertIs(wrapper.model, optimizer_owned_model)
            self.assertIsNone(wrapper.model.text_tokenizer)
            self.assertEqual(wrapper.model.weight.item(), 7.0)

    def test_portable_artifact_rebuilds_an_inference_capable_runtime(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = OmniVoiceConfig(name_or_path="test/omnivoice")
            config.save_pretrained(output)
            source = OmniVoiceForTextToSpeech(config, device="cpu")

            with patch.object(
                OmniVoiceForTextToSpeech,
                "_load_pretrained_model",
                autospec=True,
                side_effect=self._fake_loader,
            ):
                source.load_for_training()
                source.model.weight.data.fill_(5.0)
                torch.save(
                    source.get_training_adapter().state_dict(),
                    output / "model_state.pt",
                )

                restored = OmniVoiceForTextToSpeech.from_pretrained(
                    directory,
                    device="cpu",
                    lazy_load=False,
                )
                generated = restored.generate("hello")

            self.assertFalse(restored._loaded_for_training)
            self.assertIsNotNone(restored.model.text_tokenizer)
            self.assertEqual(restored.model.weight.item(), 5.0)
            self.assertEqual(float(generated.audio[0]), 5.0)


if __name__ == "__main__":
    unittest.main()
