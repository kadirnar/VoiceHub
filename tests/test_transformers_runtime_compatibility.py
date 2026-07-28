import importlib
import importlib.util
import unittest
from unittest.mock import patch


def _modules_are_available(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


HIGGS_RUNTIME_AVAILABLE = _modules_are_available(
    "safetensors",
    "torch",
    "transformers",
)
XTTS_RUNTIME_AVAILABLE = _modules_are_available(
    "fsspec",
    "numpy",
    "torch",
    "torchaudio",
    "transformers",
)
MELO_RUNTIME_AVAILABLE = _modules_are_available(
    "MeCab",
    "pykakasi",
    "transformers",
    "unidic_lite",
)
PARLER_RUNTIME_AVAILABLE = _modules_are_available("torch", "transformers")
OUTE_RUNTIME_AVAILABLE = _modules_are_available(
    "loguru",
    "polars",
    "torch",
    "transformers",
)


@unittest.skipUnless(
    HIGGS_RUNTIME_AVAILABLE,
    "HiggsTTS runtime dependencies are optional",
)
class HiggsTransformersCompatibilityTests(unittest.TestCase):

    def test_llama_attention_factory_uses_the_installed_transformers_api(self):
        import torch
        from transformers import LlamaConfig

        runtime = importlib.import_module(
            "voicehub.models.higgstts.source.boson_multimodal.model."
            "higgs_audio.modeling_higgs_audio"
        )
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        config._attn_implementation = "eager"

        attention = runtime._build_llama_attention(config, 3, "eager")

        self.assertIsInstance(attention, torch.nn.Module)
        self.assertEqual(attention.layer_idx, 3)


@unittest.skipUnless(
    XTTS_RUNTIME_AVAILABLE,
    "XTTS runtime dependencies are optional",
)
class XTTSTransformersCompatibilityTests(unittest.TestCase):

    def test_streaming_and_tortoise_modules_import(self):
        stream_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.xtts.stream_generator"
        )
        tortoise_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.tortoise.arch_utils"
        )

        config = stream_runtime.StreamGenerationConfig(do_stream=True)
        self.assertTrue(config.do_stream)
        self.assertTrue(callable(tortoise_runtime.TypicalLogitsWarper))

    def test_legacy_beam_types_are_loaded_only_when_available(self):
        stream_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.xtts.stream_generator"
        )

        try:
            scorer = stream_runtime._load_legacy_generation_symbol(
                "BeamSearchScorer"
            )
        except RuntimeError as error:
            self.assertIn("Use Transformers 4", str(error))
        else:
            self.assertEqual(scorer.__name__, "BeamSearchScorer")

    def test_typical_logits_warper_preserves_tensor_shape(self):
        import torch

        tortoise_runtime = importlib.import_module(
            "voicehub.models.xtts.source.TTS.tts.layers.tortoise.arch_utils"
        )
        warper = tortoise_runtime.TypicalLogitsWarper(mass=0.9)
        input_ids = torch.tensor([[1, 2]])
        scores = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        filtered_scores = warper(input_ids, scores)

        self.assertEqual(filtered_scores.shape, scores.shape)


@unittest.skipUnless(
    PARLER_RUNTIME_AVAILABLE,
    "ParlerTTS runtime dependencies are unavailable",
)
class ParlerTransformersCompatibilityTests(unittest.TestCase):

    def test_logits_processor_uses_public_torch_isin_fallback(self):
        import torch

        runtime = importlib.import_module(
            "voicehub.models.parlertts.source.parler_tts.logits_processors"
        )
        processor = runtime.ParlerTTSLogitsProcessor(
            eos_token_id=3,
            num_codebooks=2,
            batch_size=1,
        )
        scores = torch.zeros((2, 5))

        output = processor(torch.tensor([[1, 3], [1, 2]]), scores)

        self.assertEqual(output.shape, scores.shape)


@unittest.skipUnless(
    MELO_RUNTIME_AVAILABLE,
    "MeloTTS language runtime dependencies are unavailable",
)
class MeloImportCompatibilityTests(unittest.TestCase):

    def test_japanese_frontend_import_does_not_download_a_tokenizer(self):
        from transformers import AutoTokenizer

        with patch.object(
            AutoTokenizer,
            "from_pretrained",
            side_effect=AssertionError("tokenizer download during import"),
        ) as loader:
            runtime = importlib.import_module(
                "voicehub.models.melotts.source.melo.text.japanese"
            )
            importlib.reload(runtime)

        loader.assert_not_called()
        self.assertEqual(runtime.text2kata("こんにちは"), "コンニチワ")


@unittest.skipUnless(
    OUTE_RUNTIME_AVAILABLE,
    "OuteTTS runtime dependencies are unavailable",
)
class OuteBackendCompatibilityTests(unittest.TestCase):

    def test_gguf_dependency_is_checked_only_when_the_backend_is_used(self):
        runtime = importlib.import_module(
            "voicehub.models.outetts.source.outetts.models.gguf_model"
        )
        if runtime._GGUF_AVAILABLE:
            self.assertTrue(callable(runtime.Llama))
            return

        with self.assertRaisesRegex(ImportError, "llama-cpp-python"):
            runtime.GGUFModel("model.gguf")


if __name__ == "__main__":
    unittest.main()
