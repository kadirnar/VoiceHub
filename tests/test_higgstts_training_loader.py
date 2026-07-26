import gc
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import voicehub.models.higgstts.training as higgs_training
from voicehub.models.higgstts.inference import HiggsTTSConfig, HiggsTTSForTextToSpeech
from voicehub.models.higgstts.training import HiggsTrainingBackend, load_higgs_training_backend

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HiggsTrainingImportTests(unittest.TestCase):

    def test_training_module_keeps_heavy_dependencies_lazy(self):
        command = (
            "import sys; "
            "import voicehub.models.higgstts.training; "
            "print('torch' in sys.modules, 'transformers' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional Higgs extra")
class HiggsTrainingBackendTests(unittest.TestCase):

    def test_training_loader_never_imports_or_retains_serving_caches(self):
        import torch

        calls = {}

        class FakeModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(
                    use_cache=True,
                    text_config=SimpleNamespace(
                        use_cache=True,
                        num_hidden_layers=2,
                    ),
                    encode_whisper_embed=True,
                    encode_audio_in_tokens=True,
                    audio_in_token_idx=10,
                    audio_out_token_idx=11,
                    audio_stream_bos_id=12,
                    audio_stream_eos_id=13,
                    pad_token_id=0,
                    use_delay_pattern=True,
                    audio_num_codebooks=2,
                    audio_codebook_size=16,
                    audio_dual_ffn_layers=[1],
                )
                self.decode_graph_runners = {"stale": object()}
                self.current_past_key_values_bucket = 4096
                self.special_tokens_from = None
                self.to_device = None

            @property
            def device(self):
                return self.weight.device

            @property
            def dtype(self):
                return self.weight.dtype

            def to(self, device):
                self.to_device = device
                return super().to(device)

            def set_audio_special_tokens(self, tokenizer):
                self.special_tokens_from = tokenizer

            def capture_model(self, caches):
                calls["captured"] = list(caches)

        model = FakeModel()

        class FakeModelFactory:

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                calls["model"] = (name, kwargs)
                return model

        tokenizer = SimpleNamespace()
        tokenizer.save_pretrained = Mock()

        class FakeTokenizerFactory:

            @classmethod
            def from_pretrained(cls, name):
                calls["tokenizer"] = name
                return tokenizer

        whisper_processor = object()

        class FakeProcessorFactory:

            @classmethod
            def from_pretrained(cls, name, **kwargs):
                calls["processor"] = (name, kwargs)
                return whisper_processor

        class FakeAudioTokenizer(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.sampling_rate = 24_000
                self.tps = 50

        audio_tokenizer = FakeAudioTokenizer()

        class FakeCollator:

            def __init__(self, **kwargs):
                calls["collator"] = kwargs
                for name, value in kwargs.items():
                    setattr(self, name, value)

        modules = {
            "torch":
            torch,
            "transformers":
            SimpleNamespace(
                AutoTokenizer=FakeTokenizerFactory,
                AutoProcessor=FakeProcessorFactory,
            ),
            ("voicehub.models.higgstts.source.boson_multimodal.model."
             "higgs_audio"):
            SimpleNamespace(HiggsAudioModel=FakeModelFactory),
            ("voicehub.models.higgstts.source.boson_multimodal."
             "audio_processing.higgs_audio_tokenizer"):
            SimpleNamespace(load_higgs_audio_tokenizer=Mock(return_value=audio_tokenizer, ), ),
            ("voicehub.models.higgstts.source.boson_multimodal."
             "data_collator.higgs_audio_collator"):
            SimpleNamespace(HiggsAudioSampleCollator=FakeCollator),
        }
        imported = []

        def optional_dependency(name, **kwargs):
            del kwargs
            imported.append(name)
            if name not in modules:
                raise AssertionError(f"Unexpected training import: {name}")
            return modules[name]

        with patch.object(
                higgs_training,
                "import_optional",
                side_effect=optional_dependency,
        ):
            backend = load_higgs_training_backend(
                "bosonai/training-checkpoint",
                "bosonai/audio-tokenizer",
                device="cpu",
                torch_dtype="bfloat16",
            )

        self.assertIsInstance(backend, HiggsTrainingBackend)
        self.assertNotIn("serve.serve_engine", " ".join(imported))
        self.assertNotIn("transformers.cache_utils", imported)
        self.assertFalse(hasattr(backend, "kv_caches"))
        self.assertEqual(model.decode_graph_runners, {})
        self.assertIsNone(model.current_past_key_values_bucket)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)
        self.assertFalse(audio_tokenizer.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in audio_tokenizer.parameters()))
        self.assertEqual(
            calls["model"],
            (
                "bosonai/training-checkpoint",
                {
                    "torch_dtype": torch.float32,
                    "use_safetensors": True,
                },
            ),
        )
        self.assertEqual(model.to_device, "cpu")
        self.assertIs(model.special_tokens_from, tokenizer)
        self.assertEqual(
            calls["processor"],
            (
                "openai/whisper-large-v3-turbo",
                {
                    "trust_remote_code": True,
                },
            ),
        )
        self.assertEqual(calls["collator"]["round_to"], 8)
        self.assertFalse(calls["collator"]["pad_left"])
        self.assertTrue(calls["collator"]["return_audio_in_tokens"])

        class FakeStaticCache:

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def reset(self):
                return None

        class FakeServeEngine:
            pass

        inference_modules = {
            "transformers.cache_utils":
            SimpleNamespace(StaticCache=FakeStaticCache),
            ("voicehub.models.higgstts.source.boson_multimodal.serve."
             "serve_engine"):
            SimpleNamespace(HiggsAudioServeEngine=FakeServeEngine),
        }
        with patch.object(
                higgs_training,
                "import_optional",
                side_effect=lambda name, **kwargs: inference_modules[name],
        ):
            engine = backend.build_inference_runtime(
                device="cpu",
                kv_cache_lengths=(32, 16),
            )

        self.assertIs(engine.model, model)
        self.assertEqual(list(engine.kv_caches), [16, 32])
        self.assertTrue(model.config.use_cache)
        self.assertTrue(model.config.text_config.use_cache)
        self.assertEqual(engine.collator.round_to, 1)
        self.assertFalse(engine.collator.return_audio_in_tokens)
        self.assertNotIn("captured", calls)


class HiggsWrapperRuntimeTests(unittest.TestCase):

    def test_training_selection_does_not_construct_the_serve_engine(self):
        backend = SimpleNamespace(
            model=object(),
            sample_rate=24_000,
            prepare_for_training=Mock(),
        )
        model = HiggsTTSForTextToSpeech(device="cpu")

        with patch(
                "voicehub.models.higgstts.training."
                "load_higgs_training_backend",
                return_value=backend,
        ) as loader, patch(
                "voicehub.models.higgstts.inference.import_optional",
                side_effect=AssertionError("serve engine import is forbidden"),
        ):
            model._loading_for_training = True
            try:
                model._load_pretrained_model()
            finally:
                model._loading_for_training = False

        loader.assert_called_once_with(
            "bosonai/higgs-audio-v2-generation-3B-base",
            "bosonai/higgs-audio-v2-tokenizer",
            device="cpu",
            torch_dtype="bfloat16",
        )
        self.assertIs(model.model, backend)
        self.assertIs(model.training_backend, backend)
        self.assertFalse(hasattr(model.model, "kv_caches"))

    def test_switching_to_training_releases_the_cached_serve_engine(self):

        class InferenceRuntime:

            def __init__(self):
                self.kv_caches = {
                    1024: object(),
                }

        backend = SimpleNamespace(
            model=object(),
            prepare_for_training=Mock(),
        )
        model = HiggsTTSForTextToSpeech(device="cpu")
        inference_runtime = InferenceRuntime()
        inference_reference = weakref.ref(inference_runtime)
        model.model = inference_runtime
        del inference_runtime

        def load_training_runtime():
            self.assertTrue(model.is_training_load)
            model.model = backend
            model._training_backend = backend
            return model

        with patch.object(
                model,
                "load",
                side_effect=load_training_runtime,
        ):
            model._prepare_for_training()

        gc.collect()
        self.assertIsNone(inference_reference())
        self.assertIs(model.model, backend)

    def test_trainer_artifact_can_convert_to_fresh_inference_runtime(self):

        class Message:

            def __init__(self, **values):
                self.__dict__.update(values)

        class ChatMLSample:

            def __init__(self, **values):
                self.__dict__.update(values)

        trained_model = object()
        response = SimpleNamespace(
            audio=np.array([0.25, -0.25], dtype=np.float32),
            sampling_rate=24_000,
            generated_text="restored",
            usage={"total_tokens": 3},
        )
        engine = SimpleNamespace(
            model=trained_model,
            generate=Mock(return_value=response),
        )
        backend = SimpleNamespace(
            model=trained_model,
            sample_rate=24_000,
            build_inference_runtime=Mock(return_value=engine),
        )
        data_types = SimpleNamespace(
            Message=Message,
            ChatMLSample=ChatMLSample,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            HiggsTTSConfig(name_or_path="bosonai/trained-base", ).save_pretrained(output)
            (output / "model_state.pt").touch()

            restored = HiggsTTSForTextToSpeech.from_pretrained(
                directory,
                device="cpu",
            )

            def restore_state(wrapper, *, training):
                self.assertFalse(training)
                wrapper._pending_model_state_path = None

            with patch(
                    "voicehub.models.higgstts.training."
                    "load_higgs_training_backend",
                    return_value=backend,
            ) as loader, patch.object(
                    HiggsTTSForTextToSpeech,
                    "_restore_voicehub_model_state",
                    autospec=True,
                    side_effect=restore_state,
            ), patch(
                    "voicehub.models.higgstts.inference.import_optional",
                    return_value=data_types,
            ), patch(
                    "voicehub.models.higgstts.inference.seeded_inference",
                    return_value=nullcontext(23),
            ):
                generated = restored.generate("restored artifact")

        loader.assert_called_once_with(
            "bosonai/trained-base",
            "bosonai/higgs-audio-v2-tokenizer",
            device="cpu",
            torch_dtype="bfloat16",
        )
        backend.build_inference_runtime.assert_called_once_with(device="cpu")
        self.assertIs(restored.model, engine)
        self.assertIsNone(restored.training_backend)
        self.assertIs(engine.model, trained_model)
        np.testing.assert_array_equal(generated.audio, response.audio)
        self.assertEqual(generated.metadata["generated_text"], "restored")


if __name__ == "__main__":
    unittest.main()
