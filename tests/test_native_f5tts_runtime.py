from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from voicehub.architectures.f5tts.audio import F5MelSpectrogram
from voicehub.architectures.f5tts.checkpoint import (
    convert_legacy_f5tts_checkpoint,
    export_f5tts_checkpoint,
    load_f5tts_checkpoint,
)
from voicehub.architectures.f5tts.configuration import F5TTSArchitectureConfig
from voicehub.architectures.f5tts.frontend import F5Vocabulary, NativeF5TextFrontend
from voicehub.architectures.f5tts.metadata import (
    F5TTS_CHECKPOINT_LICENSE,
    F5TTS_CHECKPOINT_REVISION,
    F5TTS_SOURCE_REVISION,
    F5TTS_V1_BASE_GRAPH_TENSOR_COUNT,
    F5TTS_V1_BASE_PARAMETER_COUNT,
    VOCOS_CHECKPOINT_REVISION,
    VOCOS_SOURCE_REVISION,
)
from voicehub.architectures.f5tts.modeling import F5ConditionalFlowMatcher
from voicehub.architectures.f5tts.modules import (
    RotaryEmbedding,
    apply_rotary_position_embedding,
)
from voicehub.architectures.f5tts.registration import (
    create_f5tts_architecture_spec,
)
from voicehub.architectures.f5tts.runtime import NativeF5TTSRuntime
from voicehub.architectures.f5tts.vocoder import NativeVocos
from voicehub.models.f5tts.inference import F5TTSConfig, F5TTSForTextToSpeech
from voicehub.training.recipes import F5TTSTrainingAdapter
from voicehub.training.specs import get_training_spec


def _tiny_config() -> F5TTSArchitectureConfig:
    return F5TTSArchitectureConfig(
        model_name="test-f5",
        mel_dim=8,
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        text_dim=16,
        text_num_embeds=12,
        conv_layers=1,
        n_fft=32,
        win_length=32,
        hop_length=8,
        sample_rate=8_000,
        dropout=0.0,
    )


def _tiny_vocabulary() -> F5Vocabulary:
    return F5Vocabulary(
        (" ", ".", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j")
    )


class NativeF5TTSRuntimeTests(unittest.TestCase):

    def test_registration_is_lazy_and_records_distinct_artifact_license(self):
        code = (
            "import json,sys;"
            "from voicehub.architectures.f5tts.registration import "
            "create_f5tts_architecture_spec;"
            "s=create_f5tts_architecture_spec();"
            "print(json.dumps({'torch': 'torch' in sys.modules,"
            "'license': s.license_id,"
            "'checkpoint': s.metadata['checkpoint_license']}))"
        )
        result = subprocess.run(
            (sys.executable, "-c", code),
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout)
        self.assertFalse(payload["torch"])
        self.assertEqual(payload["license"], "MIT")
        self.assertEqual(payload["checkpoint"], F5TTS_CHECKPOINT_LICENSE)
        spec = create_f5tts_architecture_spec()
        self.assertTrue(spec.capabilities.training)
        self.assertTrue(spec.metadata["full_finetuning_ready"])
        self.assertEqual(len(F5TTS_SOURCE_REVISION), 40)
        self.assertEqual(len(F5TTS_CHECKPOINT_REVISION), 40)
        self.assertEqual(len(VOCOS_SOURCE_REVISION), 40)
        self.assertEqual(len(VOCOS_CHECKPOINT_REVISION), 40)

    def test_released_graph_has_the_audited_tensor_inventory_on_meta(self):
        with torch.device("meta"):
            model = F5ConditionalFlowMatcher(F5TTSArchitectureConfig())
        state = model.state_dict()
        self.assertEqual(len(state), F5TTS_V1_BASE_GRAPH_TENSOR_COUNT)
        self.assertEqual(
            sum(tensor.numel() for tensor in state.values()),
            F5TTS_V1_BASE_PARAMETER_COUNT,
        )
        self.assertEqual(
            tuple(
                state[
                    "transformer.text_embed.text_embed.weight"
                ].shape
            ),
            (2_546, 512),
        )
        self.assertEqual(
            tuple(state["transformer.rotary_embed.inv_freq"].shape),
            (32,),
        )

    def test_rotary_embedding_uses_x_transformers_adjacent_pairs(self):
        rotary = RotaryEmbedding(4)
        frequencies, scale = rotary.forward_from_seq_len(2)
        hidden = torch.tensor(
            [[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]
        )
        actual = apply_rotary_position_embedding(hidden, frequencies, scale)
        angle = frequencies[1]
        expected_last = (
            hidden[0, 0, 1] * angle.cos()
            + torch.tensor([-6.0, 5.0, -8.0, 7.0]) * angle.sin()
        )
        self.assertTrue(torch.allclose(actual[0, 0, 0], hidden[0, 0, 0]))
        self.assertTrue(torch.allclose(actual[0, 0, 1], expected_last))

    def test_native_flow_objective_is_differentiable(self):
        model = F5ConditionalFlowMatcher(_tiny_config())
        mel = torch.randn(2, 20, 8)
        text = torch.tensor(
            [[2, 3, 4, 5, -1], [6, 7, 8, -1, -1]],
            dtype=torch.long,
        )
        loss, conditioning, prediction = model(
            mel,
            text,
            lens=torch.tensor((20, 17)),
        )
        loss.backward()
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(conditioning.shape, mel.shape)
        self.assertEqual(prediction.shape, mel.shape)
        self.assertIsNotNone(
            model.transformer.input_embed.proj.weight.grad
        )

    def test_native_sampler_is_seeded_and_preserves_reference_frames(self):
        model = F5ConditionalFlowMatcher(_tiny_config()).eval()
        reference = torch.randn(1, 7, 8)
        text = torch.tensor([[2, 3, 4, 5]])
        first, first_path = model.sample(
            reference,
            text,
            12,
            lengths=torch.tensor((7,)),
            steps=3,
            seed=31,
        )
        second, second_path = model.sample(
            reference,
            text,
            12,
            lengths=torch.tensor((7,)),
            steps=3,
            seed=31,
        )
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first_path, second_path))
        self.assertTrue(torch.equal(first[:, :7], reference))
        self.assertEqual(tuple(first_path.shape), (4, 1, 12, 8))

    def test_checkpoint_round_trip_is_strict_and_prefix_compatible(self):
        source = F5ConditionalFlowMatcher(_tiny_config())
        target = F5ConditionalFlowMatcher(_tiny_config())
        with torch.no_grad():
            for index, parameter in enumerate(source.parameters(), start=1):
                parameter.fill_((index % 7) / 10)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            export_f5tts_checkpoint(source, checkpoint)
            report = load_f5tts_checkpoint(target, checkpoint)
            self.assertEqual(report.prefix, "ema_model.")
            self.assertEqual(report.tensor_count, len(source.state_dict()))
            for name, tensor in source.state_dict().items():
                self.assertTrue(torch.equal(tensor, target.state_dict()[name]))

            incompatible = F5ConditionalFlowMatcher(
                F5TTSArchitectureConfig(
                    **{
                        **_tiny_config().to_dict(),
                        "text_num_embeds": 13,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                load_f5tts_checkpoint(incompatible, checkpoint)

    def test_legacy_conversion_is_weights_only(self):
        model = F5ConditionalFlowMatcher(_tiny_config())
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.pt"
            native = Path(directory) / "model.safetensors"
            torch.save(
                {
                    "state_dict": {
                        f"ema_model.{name}": value
                        for name, value in model.state_dict().items()
                    }
                },
                legacy,
            )
            with patch.object(torch, "load", wraps=torch.load) as load:
                convert_legacy_f5tts_checkpoint(legacy, native)
            self.assertTrue(load.call_args.kwargs["weights_only"])
            restored = F5ConditionalFlowMatcher(_tiny_config())
            load_f5tts_checkpoint(restored, native)

    def test_vocos_namespace_and_decoder_are_native(self):
        with torch.device("meta"):
            vocoder = NativeVocos()
        state = vocoder.state_dict()
        self.assertEqual(len(state), 83)
        self.assertEqual(
            tuple(state["feature_extractor.mel_spec.mel_scale.fb"].shape),
            (513, 100),
        )
        self.assertEqual(
            tuple(state["head.out.weight"].shape),
            (1_026, 512),
        )
        actual = NativeVocos()
        waveform = actual.decode(torch.randn(1, 100, 8))
        self.assertEqual(tuple(waveform.shape), (1, 1_792))
        self.assertTrue(torch.isfinite(waveform).all())

    def test_frontend_requires_explicit_chinese_normalization(self):
        frontend = NativeF5TextFrontend(_tiny_vocabulary())
        self.assertEqual(frontend.encode("abc.").tolist(), [2, 3, 4, 1])
        with self.assertRaisesRegex(ValueError, "pinyin-with-tone"):
            frontend.encode("你好")
        normalized = NativeF5TextFrontend(
            _tiny_vocabulary(),
            normalizer=lambda text: ("a", "b"),
        )
        self.assertEqual(normalized.encode("你好").tolist(), [2, 3])

    def test_ema_training_export_reloads_into_a_fresh_flow_graph(self):
        architecture = _tiny_config()
        flow = F5ConditionalFlowMatcher(architecture)
        runtime = NativeF5TTSRuntime(
            flow_model=flow,
            vocoder=None,
            frontend=NativeF5TextFrontend(_tiny_vocabulary()),
        )
        wrapper = F5TTSForTextToSpeech(
            F5TTSConfig(
                architecture=architecture.to_dict(),
                model_name=architecture.model_name,
            ),
            device="cpu",
        )
        wrapper.model = runtime
        adapter = F5TTSTrainingAdapter(
            wrapper,
            get_training_spec("f5tts"),
        ).setup()
        with torch.no_grad():
            next(flow.parameters()).add_(0.25)
        adapter.on_optimizer_step(optimizer_names=("model",), step=1)

        with tempfile.TemporaryDirectory() as directory:
            adapter.save_pretrained(directory)
            self.assertTrue((Path(directory) / "config.json").is_file())
            self.assertTrue((Path(directory) / "vocab.txt").is_file())
            fresh = F5ConditionalFlowMatcher(architecture)
            load_f5tts_checkpoint(
                fresh,
                Path(directory) / "model.safetensors",
            )
            shadow = adapter.recipe_state_dict()["ema"]["shadow"]
            for name, tensor in fresh.state_dict().items():
                expected = shadow.get(name, flow.state_dict()[name])
                self.assertTrue(torch.equal(tensor, expected))

    def test_mel_frontend_is_pure_torch_and_has_expected_shape(self):
        mel = F5MelSpectrogram(
            sample_rate=8_000,
            n_fft=32,
            hop_length=8,
            win_length=32,
            n_mels=8,
        )(torch.randn(2, 128))
        self.assertEqual(mel.shape[:2], (2, 8))
        self.assertTrue(torch.isfinite(mel).all())


if __name__ == "__main__":
    unittest.main()
