from __future__ import annotations

import ast
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional

from voicehub.architectures.sensevoice.checkpoint import (
    NATIVE_SENSEVOICE_FORMAT,
    SenseVoiceSafeTensorsCheckpointAdapter,
    load_native_sensevoice_model,
    native_sensevoice_tensor_shapes,
    tensor_inventory_fingerprint,
)
from voicehub.architectures.sensevoice.configuration import SenseVoiceSmallConfig
from voicehub.architectures.sensevoice.decoding import ctc_forced_align, ctc_greedy_tokens
from voicehub.architectures.sensevoice.frontend import load_sensevoice_cmvn, low_frame_rate_stack
from voicehub.architectures.sensevoice.metadata import (
    FUNASR_SOURCE_REVISION,
    SENSEVOICE_REVISION,
    SENSEVOICE_STATE_VALUES,
    SENSEVOICE_TENSOR_COUNT,
    SENSEVOICE_TENSOR_FINGERPRINT,
)
from voicehub.architectures.sensevoice.modeling import (
    MultiHeadedAttentionSANM,
    SenseVoiceSmallForCTC,
    SinusoidalPositionEncoder,
)
from voicehub.architectures.sensevoice.registration import create_sensevoice_architecture_spec
from voicehub.architectures.sensevoice.tokenization import SenseVoiceTokenizer, rich_transcription_postprocess
from voicehub.architectures.sensevoice.training import NativeSenseVoiceTrainingAdapter
from voicehub.checkpointing import save_safetensors
from voicehub.models.asr_native.configuration import FunASRConfig
from voicehub.models.asr_native.funasr import FunASRForSpeechRecognition
from voicehub.training.specs import TrainingFamily, get_training_spec


def _tiny_config() -> SenseVoiceSmallConfig:
    return SenseVoiceSmallConfig(
        variant="custom",
        num_mel_bins=4,
        lfr_window=3,
        input_dimension=12,
        vocabulary_size=25_055,
        encoder_dimension=8,
        attention_heads=2,
        linear_units=16,
        encoder_blocks=2,
        temporal_blocks=1,
        memory_kernel_size=3,
        query_embedding_size=16,
        dropout=0.0,
        attention_dropout=0.0,
        language_dropout=0.0,
    )


class NativeSenseVoiceTests(unittest.TestCase):

    def test_training_contract_is_ctc_not_sequence_to_sequence(self):
        spec = get_training_spec("asr_funasr")
        self.assertIs(spec.family, TrainingFamily.CTC)
        wrapper = FunASRForSpeechRecognition(device="cpu")
        self.assertEqual(wrapper.architecture_family, "ctc")
        self.assertIsInstance(
            wrapper.get_training_adapter(),
            NativeSenseVoiceTrainingAdapter,
        )

    def test_public_path_has_no_external_model_runtime_imports(self):
        root = Path(__file__).resolve().parents[1]
        files = [
            *sorted((root / "voicehub" / "architectures" / "sensevoice").glob("*.py")),
            root / "voicehub" / "models" / "asr_native" / "funasr.py",
        ]
        forbidden = {
            "funasr",
            "huggingface_hub",
            "modelscope",
            "numpy",
            "sentencepiece",
            "torchaudio",
            "transformers",
        }
        found = set()
        for path in files:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.update(
                        alias.name.split(".", 1)[0] for alias in node.names
                        if alias.name.split(".", 1)[0] in forbidden)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    package = node.module.split(".", 1)[0]
                    if package in forbidden:
                        found.add(package)
        self.assertEqual(found, set())

    def test_release_inventory_is_exact_and_provenance_is_immutable(self):
        shapes = native_sensevoice_tensor_shapes()
        self.assertEqual(len(shapes), SENSEVOICE_TENSOR_COUNT)
        self.assertEqual(
            sum(math.prod(shape) for shape in shapes.values()),
            SENSEVOICE_STATE_VALUES,
        )
        with torch.device("meta"):
            model = SenseVoiceSmallForCTC()
        self.assertEqual(
            tensor_inventory_fingerprint(model.state_dict()),
            SENSEVOICE_TENSOR_FINGERPRINT,
        )
        self.assertEqual(
            shapes["encoder.encoders0.0.self_attn.linear_q_k_v.weight"],
            (1_536, 560),
        )
        self.assertEqual(shapes["ctc.ctc_lo.weight"], (25_055, 512))
        self.assertEqual(shapes["embed.weight"], (16, 560))
        source = (
            Path(__file__).resolve().parents[1] / "voicehub" / "architectures" / "sensevoice" / "SOURCE.json")
        metadata = json.loads(source.read_text(encoding="utf-8"))
        self.assertEqual(metadata["source"]["revision"], FUNASR_SOURCE_REVISION)
        self.assertEqual(
            metadata["checkpoint"]["revision"],
            SENSEVOICE_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            SenseVoiceSmallConfig(encoder_blocks=49)

    def test_architecture_spec_is_native_and_scope_limited(self):
        spec = create_sensevoice_architecture_spec()
        self.assertEqual(spec.architecture_id, "sensevoice-small")
        self.assertTrue(spec.capabilities.training)
        self.assertIn(
            "safetensors",
            spec.capabilities.checkpoint_formats,
        )
        self.assertIn("Paraformer", spec.metadata["verified_scope"])

    def test_position_encoding_matches_published_equation(self):
        value = torch.zeros(1, 5, 8, dtype=torch.float64)
        actual = SinusoidalPositionEncoder()(value)
        positions = torch.arange(1, 6, dtype=torch.float64)
        inverse = torch.exp(torch.arange(4, dtype=torch.float64) * -(math.log(10_000.0) / 3))
        scaled = positions[:, None] * inverse[None, :]
        expected = torch.cat((scaled.sin(), scaled.cos()), dim=-1)
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-12)

    def test_sanm_attention_matches_direct_source_equation(self):
        torch.manual_seed(7)
        layer = MultiHeadedAttentionSANM(
            heads=2,
            input_dimension=6,
            output_dimension=4,
            dropout=0.0,
            kernel_size=3,
        ).eval()
        value = torch.randn(2, 5, 6)
        mask = torch.tensor([
            [[True, True, True, True, True]],
            [[True, True, True, False, False]],
        ])
        actual = layer(value, mask)

        qkv = functional.linear(
            value,
            layer.linear_q_k_v.weight,
            layer.linear_q_k_v.bias,
        )
        query, key, projected_value = qkv.chunk(3, dim=-1)

        def heads(item):
            return item.reshape(2, 5, 2, 2).transpose(1, 2)

        query_heads = heads(query)
        key_heads = heads(key)
        value_heads = heads(projected_value)
        visible = mask.reshape(2, 5, 1).to(value.dtype)
        masked_value = projected_value * visible
        memory = functional.conv1d(
            functional.pad(masked_value.transpose(1, 2), (1, 1)),
            layer.fsmn_block.weight,
            groups=4,
        ).transpose(1, 2)
        memory = (memory + masked_value) * visible
        scores = torch.matmul(
            query_heads * 2**-0.5,
            key_heads.transpose(-2, -1),
        )
        blocked = ~mask.unsqueeze(1)
        probabilities = torch.softmax(
            scores.masked_fill(blocked, -float("inf")),
            dim=-1,
        ).masked_fill(blocked, 0.0)
        attended = torch.matmul(probabilities, value_heads)
        attended = attended.transpose(1, 2).reshape(2, 5, 4)
        attended = functional.linear(
            attended,
            layer.linear_out.weight,
            layer.linear_out.bias,
        )
        torch.testing.assert_close(actual, attended + memory)

    def test_lfr_and_cmvn_match_published_boundaries(self):
        features = torch.arange(1, 6, dtype=torch.float32).unsqueeze(1)
        actual = low_frame_rate_stack(features, window=3, stride=2)
        expected = torch.tensor([
            [1.0, 1.0, 2.0],
            [2.0, 3.0, 4.0],
            [4.0, 5.0, 5.0],
        ])
        torch.testing.assert_close(actual, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "am.mvn"
            path.write_text(
                "<Nnet>\n"
                "<AddShift> 3 3\n"
                "<LearnRateCoef> 0 [ -1 -2 -3 ]\n"
                "<Rescale> 3 3\n"
                "<LearnRateCoef> 0 [ 0.5 0.25 0.125 ]\n"
                "</Nnet>\n",
                encoding="utf-8",
            )
            shift, scale = load_sensevoice_cmvn(
                path,
                expected_dimension=3,
            )
        torch.testing.assert_close(shift, torch.tensor([-1.0, -2.0, -3.0]))
        torch.testing.assert_close(
            scale,
            torch.tensor([0.5, 0.25, 0.125]),
        )

    def test_native_objective_backpropagates_through_queries_and_encoder(self):
        torch.manual_seed(11)
        model = SenseVoiceSmallForCTC(_tiny_config())
        features = torch.randn(2, 7, 12)
        feature_lengths = torch.tensor([7, 6])
        labels = torch.tensor([
            [24_885, 25_004, 24_993, 25_017, 8, 9],
            [24_885, 25_001, 24_997, 25_016, 7, -1],
        ])
        output = model(
            features,
            feature_lengths,
            labels=labels,
            label_lengths=torch.tensor([6, 5]),
        )
        self.assertTrue(torch.isfinite(output.loss))
        self.assertEqual(set(output.losses), {"ctc", "rich"})
        output.loss.backward()
        self.assertIsNotNone(model.embed.weight.grad)
        self.assertIsNotNone(model.encoder.encoders0[0].self_attn.linear_q_k_v.weight.grad)
        self.assertIsNotNone(model.ctc.ctc_lo.weight.grad)

    def test_safetensors_export_reloads_a_fresh_graph(self):
        torch.manual_seed(13)
        config = _tiny_config()
        model = SenseVoiceSmallForCTC(config).eval()
        features = torch.randn(1, 5, 12)
        lengths = torch.tensor([5])
        expected = model.infer(features, lengths, language="en").logits
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.safetensors"
            save_safetensors(
                model.state_dict(),
                checkpoint,
                metadata={
                    "format": NATIVE_SENSEVOICE_FORMAT,
                    "architecture": "sensevoice-small",
                },
            )
            reloaded = load_native_sensevoice_model(
                checkpoint,
                config,
                device="cpu",
            ).eval()
            actual = reloaded.infer(features, lengths, language="en").logits
            adapter = SenseVoiceSafeTensorsCheckpointAdapter()
        self.assertEqual(
            set(reloaded.state_dict()),
            set(native_sensevoice_tensor_shapes(config)),
        )
        self.assertEqual(
            adapter.qualified_id,
            "voicehub-sensevoice-small-safetensors@1",
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_greedy_and_forced_alignment_preserve_ctc_semantics(self):
        probabilities = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ]).log()
        self.assertEqual(
            ctc_greedy_tokens(probabilities, 5),
            (1, 2),
        )
        alignment = ctc_forced_align(
            probabilities,
            torch.tensor([1, 2]),
        )
        self.assertEqual(alignment.shape, (5, ))
        self.assertEqual(
            tuple(token for token in torch.unique_consecutive(alignment).tolist() if token),
            (1, 2),
        )

    def test_rich_postprocess_and_composed_model_boundary(self):
        self.assertEqual(
            rich_transcription_postprocess("<|en|><|HAPPY|><|Laughter|><|woitn|>hello"),
            "😀hello😊",
        )
        wrapper = FunASRForSpeechRecognition(
            FunASRConfig(
                name_or_path="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
            ),
            device="cpu",
        )
        with self.assertRaisesRegex(ValueError, "separate architectures"):
            wrapper._validate_composed_options()
        with self.assertRaisesRegex(ValueError, "only a VoiceHub"):
            wrapper._validate_architecture({
                "model_type": "paraformer",
                "architectures": ["Paraformer"],
            })

    @unittest.skipUnless(
        os.environ.get("VOICEHUB_TEST_SENSEVOICE_ASSETS"),
        "set VOICEHUB_TEST_SENSEVOICE_ASSETS for release tokenizer checks",
    )
    def test_release_tokenizer_matches_published_control_and_text_vectors(self):
        root = Path(os.environ["VOICEHUB_TEST_SENSEVOICE_ASSETS"])
        tokenizer = SenseVoiceTokenizer.from_model_file(root / "chn_jpn_yue_eng_ko_spectok.bpe.model")
        self.assertEqual(tokenizer.encode_text("hello world"), (5_000, 439, 234))
        labels = tokenizer.prepare_training_labels(
            "hello",
            language="en",
            emotion="neutral",
            event="speech",
        )
        self.assertEqual(
            labels,
            (24_885, 25_004, 24_993, 25_017, 5_000, 439),
        )
        semantics = tokenizer.semantics(labels)
        self.assertEqual(semantics.language, "en")
        self.assertEqual(semantics.emotion, "neutral")
        self.assertEqual(semantics.events, ("speech", ))


if __name__ == "__main__":
    unittest.main()
