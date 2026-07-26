import sys
import tempfile
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voicehub.models.chatterbox.inference import ChatterboxConfig, ChatterboxForTextToSpeech
from voicehub.models.conversationtts.inference import ConversationTTSConfig, ConversationTTSForTextToSpeech
from voicehub.models.cosyvoice.inference import CosyVoiceConfig, CosyVoiceForTextToSpeech
from voicehub.models.dia.inference import DiaConfig, DiaForTextToSpeech
from voicehub.models.echo.inference import EchoTTSConfig, EchoTTSForTextToSpeech
from voicehub.models.f5tts.inference import F5TTSConfig, F5TTSForTextToSpeech
from voicehub.models.irodoritts.inference import IrodoriTTSConfig, IrodoriTTSForTextToSpeech
from voicehub.models.omnivoice.inference import OmniVoiceConfig, OmniVoiceForTextToSpeech
from voicehub.models.voxcpm.inference import VoxCPMConfig, VoxCPMForTextToSpeech


@contextmanager
def _temporary_modules(modules):
    """Override selected imports without discarding modules loaded in scope."""
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


class _ModeModule:

    def __init__(self):
        self.training = True
        self.eval_calls = 0
        self.train_calls = 0
        self.to_calls = []

    def eval(self):
        self.training = False
        self.eval_calls += 1
        return self

    def train(self):
        self.training = True
        self.train_calls += 1
        return self

    def to(self, device):
        self.to_calls.append(device)
        return self


class FlowInferencePreflightTests(unittest.TestCase):

    def test_invalid_requests_fail_before_loading_optional_backends(self):
        cases = (
            (
                DiaForTextToSpeech(DiaConfig(), device="cpu"),
                {
                    "max_tokens": 8,
                    "max_new_tokens": 8,
                },
                "either",
            ),
            (
                ConversationTTSForTextToSpeech(
                    ConversationTTSConfig(),
                    device="cpu",
                ),
                {
                    "speaker_audio_path": "reference.wav",
                },
                "provided together",
            ),
            (
                ConversationTTSForTextToSpeech(
                    ConversationTTSConfig(),
                    device="cpu",
                ),
                {
                    "max_audio_length_ms": 39,
                },
                "interval",
            ),
            (
                ConversationTTSForTextToSpeech(
                    ConversationTTSConfig(),
                    device="cpu",
                ),
                {
                    "max_audio_length_ms": 81_720,
                },
                "interval",
            ),
            (
                CosyVoiceForTextToSpeech(CosyVoiceConfig(), device="cpu"),
                {
                    "mode": "zero-shot",
                    "speaker_audio_path": "reference.wav",
                },
                "prompt_text",
            ),
            (
                F5TTSForTextToSpeech(F5TTSConfig(), device="cpu"),
                {},
                "speaker_audio_path",
            ),
            (
                VoxCPMForTextToSpeech(VoxCPMConfig(), device="cpu"),
                {
                    "reference_text": "prompt",
                },
                "provided together",
            ),
            (
                OmniVoiceForTextToSpeech(OmniVoiceConfig(), device="cpu"),
                {
                    "duration": 0,
                },
                "duration",
            ),
            (
                IrodoriTTSForTextToSpeech(IrodoriTTSConfig(), device="cpu"),
                {
                    "num_steps": 0,
                },
                "num_steps",
            ),
            (
                ChatterboxForTextToSpeech(ChatterboxConfig(), device="cpu"),
                {
                    "speaker_audio_path": "speaker.wav",
                    "audio_prompt_path": "prompt.wav",
                },
                "either",
            ),
            (
                EchoTTSForTextToSpeech(EchoTTSConfig(), device="cpu"),
                {
                    "cfg_min_t": 0.9,
                    "cfg_max_t": 0.2,
                },
                "cfg_min_t",
            ),
        )

        for model, options, message in cases:
            with (
                    self.subTest(model=model.config.model_type),
                    patch.object(
                        model,
                        "_load_pretrained_model",
                        side_effect=AssertionError("backend should not load"),
                    ) as loader,
                    self.assertRaisesRegex((TypeError, ValueError), message),
            ):
                model.generate("Test request.", **options)
            loader.assert_not_called()
            self.assertFalse(model.is_loaded)

    def test_missing_reference_files_fail_before_loading_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.wav"
            cases = (
                (
                    ChatterboxForTextToSpeech(device="cpu"),
                    {
                        "speaker_audio_path": missing,
                    },
                ),
                (
                    ConversationTTSForTextToSpeech(device="cpu"),
                    {
                        "speaker_audio_path": missing,
                        "reference_text": "Reference transcript.",
                    },
                ),
                (
                    CosyVoiceForTextToSpeech(device="cpu"),
                    {
                        "mode": "zero-shot",
                        "speaker_audio_path": missing,
                        "prompt_text": "Reference transcript.",
                    },
                ),
                (
                    DiaForTextToSpeech(device="cpu"),
                    {
                        "audio_prompt_path": missing,
                    },
                ),
                (
                    EchoTTSForTextToSpeech(device="cpu"),
                    {
                        "speaker_audio_path": missing,
                    },
                ),
                (
                    IrodoriTTSForTextToSpeech(device="cpu"),
                    {
                        "speaker_audio_path": missing,
                    },
                ),
                (
                    OmniVoiceForTextToSpeech(device="cpu"),
                    {
                        "speaker_audio_path": missing,
                    },
                ),
                (
                    VoxCPMForTextToSpeech(device="cpu"),
                    {
                        "speaker_audio_path": missing,
                    },
                ),
            )

            for model, options in cases:
                with (
                        self.subTest(model=model.config.model_type),
                        patch.object(
                            model,
                            "_load_pretrained_model",
                            side_effect=AssertionError("backend should not load"),
                        ) as loader,
                        self.assertRaisesRegex(FileNotFoundError, "not found"),
                ):
                    model.generate("Test request.", **options)
                loader.assert_not_called()
                self.assertFalse(model.is_loaded)

    def test_finite_backend_option_typos_fail_before_loading(self):
        models = (
            ChatterboxForTextToSpeech(device="cpu"),
            ConversationTTSForTextToSpeech(device="cpu"),
            IrodoriTTSForTextToSpeech(device="cpu"),
            VoxCPMForTextToSpeech(device="cpu"),
        )
        for model in models:
            with (
                    self.subTest(model=model.config.model_type),
                    patch.object(
                        model,
                        "_load_pretrained_model",
                        side_effect=AssertionError("backend should not load"),
                    ) as loader,
                    self.assertRaisesRegex(ValueError, "temprature"),
            ):
                model.generate("Test request.", temprature=0.5)
            loader.assert_not_called()


class FlowInferenceLoaderTests(unittest.TestCase):

    def test_chatterbox_loads_a_local_checkpoint_without_hub_resolution(self):
        calls = []
        runtime = SimpleNamespace(
            generate=Mock(),
            sr=24_000,
            device="cpu",
        )

        class FakeChatterbox:

            @classmethod
            def from_local(cls, directory, device):
                calls.append(("local", directory, device))
                return runtime

            @classmethod
            def from_pretrained(cls, **kwargs):
                calls.append(("hub", kwargs))
                return runtime

        module = SimpleNamespace(ChatterboxTTS=FakeChatterbox)
        patched_modules = {"voicehub.models.chatterbox.tts": module}
        with tempfile.TemporaryDirectory() as directory:
            model = ChatterboxForTextToSpeech(
                model_path=directory,
                device="cpu",
            )
            with _temporary_modules(patched_modules):
                model.load()

        self.assertEqual(calls[0][0], "local")
        self.assertEqual(calls[0][1], Path(directory).resolve())
        self.assertEqual(model.sample_rate, 24_000)

    def test_echo_does_not_publish_a_partial_runtime_when_auxiliary_load_fails(self):
        source_model = object()
        module = SimpleNamespace(
            load_model_from_hf=Mock(return_value=source_model),
            load_fish_ae_from_hf=Mock(side_effect=RuntimeError("codec download failed"), ),
            load_pca_state_from_hf=Mock(),
        )
        model = EchoTTSForTextToSpeech(device="cpu")

        with (
                _temporary_modules({"voicehub.models.echo.sampling": module}, ),
                self.assertRaisesRegex(RuntimeError, "codec download failed"),
        ):
            model.load()

        self.assertIsNone(model.model)
        self.assertIsNone(model.fish_ae)
        self.assertIsNone(model.pca_state)
        module.load_pca_state_from_hf.assert_not_called()

    def test_echo_adopts_codec_sample_rate(self):
        source_model = _ModeModule()
        codec = SimpleNamespace(
            sample_rate=48_000,
            model=_ModeModule(),
        )
        model = EchoTTSForTextToSpeech(
            EchoTTSConfig(sample_rate=16_000),
            device="cpu",
        )

        with patch.object(
                model,
                "_build_runtime_components",
                return_value=(source_model, codec, object()),
        ):
            model.load()

        self.assertEqual(model.sample_rate, 48_000)
        self.assertFalse(source_model.training)
        self.assertFalse(codec.model.training)

    def test_echo_uses_sibling_pca_for_noncanonical_model_file(self):
        module = SimpleNamespace(
            load_model_from_hf=Mock(return_value=object()),
            load_fish_ae_from_hf=Mock(return_value=SimpleNamespace(sample_rate=44_100), ),
            load_pca_state_from_hf=Mock(return_value=object()),
        )
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            model_path = model_directory / "echo-ft.safetensors"
            pca_path = model_directory / "pca_state.safetensors"
            model_path.touch()
            pca_path.touch()
            model = EchoTTSForTextToSpeech(
                model_path=model_path,
                device="cpu",
            )
            with _temporary_modules({"voicehub.models.echo.sampling": module}, ):
                model.load()

        self.assertEqual(
            module.load_model_from_hf.call_args.kwargs["repo_id"],
            str(model_path.resolve()),
        )
        self.assertEqual(
            module.load_pca_state_from_hf.call_args.kwargs["repo_id"],
            model_directory.resolve(),
        )

    def test_chatterbox_inference_transition_evaluates_nested_modules(self):
        model = ChatterboxForTextToSpeech(device="cpu")
        components = {name: _ModeModule() for name in ("t3", "s3gen", "ve")}
        runtime = SimpleNamespace(**components)
        model.model = runtime
        model._training_ready = True

        model.load()

        self.assertIs(model.model, runtime)
        self.assertTrue(model._inference_ready)
        for component in components.values():
            self.assertFalse(component.training)
            self.assertEqual(component.eval_calls, 1)

    def test_f5_inference_transition_evaluates_nested_modules(self):
        ema_model = _ModeModule()
        vocoder = _ModeModule()
        runtime = SimpleNamespace(
            ema_model=ema_model,
            vocoder=vocoder,
        )
        model = F5TTSForTextToSpeech(device="cpu")
        model.model = runtime
        model._training_ready = True

        model.load()

        self.assertIs(model.model, runtime)
        self.assertTrue(model._inference_ready)
        self.assertFalse(model._training_ready)
        for component in (ema_model, vocoder):
            self.assertFalse(component.training)
            self.assertEqual(component.eval_calls, 1)

    def test_irodori_lifecycle_switches_the_nested_trainable_model(self):
        source_model = _ModeModule()
        codec_model = _ModeModule()
        runtime = SimpleNamespace(
            model=source_model,
            codec=SimpleNamespace(model=codec_model),
        )
        model = IrodoriTTSForTextToSpeech(device="cpu")
        model.model = runtime
        model._loaded_for_training = True

        model._prepare_for_inference()
        self.assertFalse(source_model.training)
        self.assertFalse(codec_model.training)

        model._prepare_for_training()
        self.assertTrue(source_model.training)
        self.assertEqual(source_model.train_calls, 1)

    def test_cosyvoice_inference_spec_forwards_selected_device(self):
        runtime = SimpleNamespace(
            CosyVoice=object(),
            CosyVoice2=object(),
            CosyVoice3=object(),
        )
        model = CosyVoiceForTextToSpeech(device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "cosyvoice3.yaml").touch()

            _, options = model._inference_runtime_spec(
                runtime,
                model_directory,
            )

        self.assertEqual(options["device"], "cpu")

    def test_irodori_ignores_speaker_inversion_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            expected = model_directory / "model.safetensors"
            expected.touch()
            (model_directory / "voice.speaker.safetensors").touch()
            model = IrodoriTTSForTextToSpeech(
                IrodoriTTSConfig(name_or_path=directory),
                device="cpu",
            )

            resolved = model._resolve_checkpoint()

        self.assertEqual(resolved, expected.resolve())

    def test_irodori_requires_selection_when_checkpoints_are_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory)
            (model_directory / "first.safetensors").touch()
            (model_directory / "second.pt").touch()
            model = IrodoriTTSForTextToSpeech(
                IrodoriTTSConfig(name_or_path=directory),
                device="cpu",
            )

            with self.assertRaisesRegex(ValueError, "checkpoint_filename"):
                model._resolve_checkpoint()

    def test_f5_uses_the_sample_rate_returned_by_inference(self):
        runtime = SimpleNamespace(
            seed=17,
            infer=Mock(return_value=([0.25, -0.25], 22_050, "spectrogram")),
        )
        model = F5TTSForTextToSpeech(device="cpu")
        model.model = runtime

        output = model._generate(
            "Test request.",
            speaker_audio_path="reference.wav",
        )

        self.assertEqual(output.audio, [0.25, -0.25])
        self.assertEqual(output.sample_rate, 22_050)
        self.assertEqual(model.sample_rate, 22_050)
        self.assertEqual(output.metadata["seed"], 17)

    def test_f5_preserves_explicit_model_name(self):
        model = F5TTSForTextToSpeech(
            model_name="E2TTS_Base",
            device="cpu",
        )

        self.assertEqual(model.config.model_name, "E2TTS_Base")
        self.assertEqual(model.config.name_or_path, "E2TTS_Base")

    def test_f5_routes_direct_weight_files_to_checkpoint_path(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fine-tuned.safetensors"
            checkpoint.touch()

            model = F5TTSForTextToSpeech.from_pretrained(
                checkpoint,
                model_name="E2TTS_Base",
                device="cpu",
            )

            self.assertEqual(model.config.model_name, "E2TTS_Base")
            self.assertEqual(
                model.config.checkpoint_path,
                str(checkpoint.resolve()),
            )
            self.assertEqual(
                model.config.name_or_path,
                str(checkpoint.resolve()),
            )

    def test_dia_legacy_maps_public_aliases(self):
        runtime = SimpleNamespace(generate=Mock(return_value=[0.1, -0.1]), )
        model = DiaForTextToSpeech(
            DiaConfig(backend="legacy"),
            device="cpu",
        )
        model.model = runtime
        model._loaded_backend = "legacy"

        model._generate_legacy(
            "Test request.",
            {
                "max_new_tokens": 16,
                "guidance_scale": 2.5,
                "top_k": 20,
                "audio_prompt_path": "prompt.wav",
            },
        )

        runtime.generate.assert_called_once_with(
            "Test request.",
            use_torch_compile=False,
            max_tokens=16,
            cfg_scale=2.5,
            cfg_filter_top_k=20,
            audio_prompt="prompt.wav",
        )

    def test_dia_transformers_transition_restores_generation_cache(self):
        source_model = _ModeModule()
        source_model.config = SimpleNamespace(use_cache=False)
        model = DiaForTextToSpeech(device="cpu")
        model.model = source_model
        model._loaded_backend = "transformers"

        model._prepare_for_inference()

        self.assertFalse(source_model.training)
        self.assertTrue(source_model.config.use_cache)

    def test_voxcpm_reports_the_seed_that_succeeded(self):
        source_model = SimpleNamespace(last_successful_seed=29)
        runtime = SimpleNamespace(
            tts_model=source_model,
            generate=Mock(return_value=[0.1, -0.1]),
        )
        model = VoxCPMForTextToSpeech(device="cpu")
        model.model = runtime

        output = model._generate(
            "Test request.",
            seed=17,
        )

        self.assertEqual(output.metadata["requested_seed"], 17)
        self.assertEqual(output.metadata["seed"], 29)

    def test_voxcpm_training_runtime_uses_parameter_dtype_for_generation(self):

        class Parameter:
            dtype = "torch.float32"

        base_lm = SimpleNamespace(setup_cache=Mock())
        residual_lm = SimpleNamespace(setup_cache=Mock())
        source_model = _ModeModule()
        source_model.parameters = lambda: iter([Parameter()])
        source_model.config = SimpleNamespace(
            dtype="bfloat16",
            max_length=128,
        )
        source_model.device = "cpu"
        source_model._dtype = lambda: "runtime-float32"
        source_model.optimize = Mock()
        source_model.base_lm = base_lm
        source_model.residual_lm = residual_lm
        runtime = SimpleNamespace(
            tts_model=source_model,
            denoiser=None,
        )
        model = VoxCPMForTextToSpeech(device="cpu")
        model.model = runtime
        model._loaded_for_training = True

        model._prepare_for_inference()

        self.assertEqual(source_model.config.dtype, "float32")
        base_lm.setup_cache.assert_called_once_with(
            1,
            128,
            "cpu",
            "runtime-float32",
        )
        residual_lm.setup_cache.assert_called_once_with(
            1,
            128,
            "cpu",
            "runtime-float32",
        )
        self.assertFalse(source_model.training)
        source_model.optimize.assert_called_once_with()

    def test_voxcpm_restores_uncompiled_modules_before_training(self):
        source_model = _ModeModule()
        source_model.deoptimize = Mock()
        runtime = SimpleNamespace(tts_model=source_model)
        model = VoxCPMForTextToSpeech(device="cpu")
        model.model = runtime
        model._loaded_for_training = True

        model._prepare_for_training()

        source_model.deoptimize.assert_called_once_with()
        self.assertTrue(source_model.training)


class OmniVoiceInferenceContractTests(unittest.TestCase):

    def test_unsupported_common_options_fail_before_loading(self):
        for options, message in (
            ({"top_p": 0.9}, "does not support `top_p`"),
            ({"max_new_tokens": 10}, "does not support `max_new_tokens`"),
            ({"misspelled_option": 1}, "Unsupported OmniVoice"),
        ):
            with self.subTest(options=options):
                model = OmniVoiceForTextToSpeech(device="cpu")
                with (
                        patch.object(
                            model,
                            "_load_pretrained_model",
                            side_effect=AssertionError("backend should not load"),
                        ) as loader,
                        self.assertRaisesRegex(ValueError, message),
                ):
                    model.generate("Test request.", **options)
                loader.assert_not_called()

    def test_temperature_maps_to_class_temperature(self):
        runtime = SimpleNamespace(generate=Mock(return_value=([0.1, -0.1], )), )
        model = OmniVoiceForTextToSpeech(device="cpu")
        model.model = runtime

        with patch(
                "voicehub.models.omnivoice.inference.seeded_inference",
                return_value=nullcontext(17),
        ) as seeded:
            output = model.generate(
                "Test request.",
                temperature=0.4,
                num_step=8,
            )

        seeded.assert_called_once_with(
            None,
            device="cpu",
            model_type="omnivoice",
        )
        self.assertEqual(output.audio, [0.1, -0.1])
        self.assertEqual(
            runtime.generate.call_args.kwargs["class_temperature"],
            0.4,
        )
        self.assertEqual(runtime.generate.call_args.kwargs["num_step"], 8)


if __name__ == "__main__":
    unittest.main()
