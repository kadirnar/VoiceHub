from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from voicehub.architectures.registry import ArchitectureRegistry
from voicehub.architectures.vits.checkpoint import (
    FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT,
    FACEBOOK_MMS_TTS_ENG_REVISION,
    HuggingFaceVitsCheckpointAdapter,
    native_vits_tensor_shapes,
    safetensors_header_fingerprint,
)
from voicehub.architectures.vits.configuration import VitsConfig
from voicehub.architectures.vits.frontend import VitsFrontendCapabilityError, VitsFrontendConfig, VitsTokenizer
from voicehub.architectures.vits.registration import create_vits_architecture_spec, register_vits_architecture
from voicehub.tasks import SpeechTask

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch


def _tiny_config(*, stochastic: bool = False) -> VitsConfig:
    return VitsConfig(
        vocab_size=8,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        window_size=2,
        ffn_dim=16,
        ffn_kernel_size=3,
        flow_size=8,
        spectrogram_bins=5,
        layerdrop=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        use_stochastic_duration_prediction=stochastic,
        upsample_initial_channel=16,
        upsample_rates=(2, ),
        upsample_kernel_sizes=(4, ),
        resblock_kernel_sizes=(3, ),
        resblock_dilation_sizes=((1, ), ),
        depth_separable_channels=2,
        depth_separable_num_layers=1,
        duration_predictor_flow_bins=4,
        duration_predictor_tail_bound=2.0,
        duration_predictor_kernel_size=3,
        duration_predictor_dropout=0.0,
        duration_predictor_num_flows=2,
        duration_predictor_filter_channels=8,
        prior_encoder_num_flows=2,
        prior_encoder_num_wavenet_layers=2,
        posterior_encoder_num_wavenet_layers=2,
        wavenet_kernel_size=3,
        wavenet_dilation_rate=2,
        wavenet_dropout=0.0,
        pad_token_id=0,
    )


class NativeVitsDeclarationTests(unittest.TestCase):

    def test_registration_is_lazy_and_capability_boundary_is_explicit(self):
        script = """
import sys
from voicehub.architectures.vits.registration import create_vits_architecture_spec
spec = create_vits_architecture_spec()
print("voicehub.architectures.vits.modeling" in sys.modules)
print(spec.metadata["full_finetuning_ready"])
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.splitlines(), ["False", "True"])

        spec = create_vits_architecture_spec()
        self.assertEqual(spec.architecture_id, "vits")
        self.assertTrue(spec.supports_task(SpeechTask.TEXT_TO_SPEECH))
        self.assertTrue(spec.capabilities.training)
        self.assertIn(
            "fine-tuning-requires-explicit-acoustic-config",
            spec.capabilities.features,
        )
        self.assertIn(
            "full-adversarial-fine-tuning",
            spec.capabilities.features,
        )
        registry = ArchitectureRegistry()
        register_vits_architecture(registry=registry)
        self.assertIs(registry.get("mms-tts"), registry.get("vits"))

    def test_vits_modules_do_not_import_architecture_frameworks(self):
        package = Path(__file__).parents[1] / "voicehub" / "architectures" / "vits"
        allowed_roots = {
            "__future__",
            "collections",
            "copy",
            "dataclasses",
            "functools",
            "hashlib",
            "importlib",
            "json",
            "math",
            "numbers",
            "operator",
            "pathlib",
            "re",
            "torch",
            "types",
            "typing",
            "voicehub",
        }
        violations = []
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [alias.name.partition(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = [node.module.partition(".")[0]]
                else:
                    continue
                violations.extend(
                    f"{path.name}:{node.lineno}:{root}" for root in roots if root not in allowed_roots)
        self.assertEqual(violations, [])

    def test_source_provenance_pins_training_and_checkpoint_revisions(self):
        source = json.loads(
            (Path(__file__).parents[1] / "voicehub" / "architectures" / "vits" /
             "SOURCE.json").read_text(encoding="utf-8"))
        self.assertEqual(
            source["sources"][0]["revision"],
            "2e561ba58618d021b5b8323d3765880f7e0ecfdb",
        )
        self.assertEqual(
            source["reference_checkpoint"]["revision"],
            FACEBOOK_MMS_TTS_ENG_REVISION,
        )
        self.assertIn(
            "requires training_acoustic_config explicitly",
            source["implementation"]["training_boundary"],
        )


class VitsConfigurationTests(unittest.TestCase):

    def test_defaults_match_mms_tts_generator(self):
        config = VitsConfig()
        self.assertEqual(config.vocab_size, 38)
        self.assertEqual(config.hidden_size, 192)
        self.assertEqual(config.num_hidden_layers, 6)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.flow_size, 192)
        self.assertEqual(config.spectrogram_bins, 513)
        self.assertEqual(config.upsample_rates, (8, 8, 2, 2))
        self.assertEqual(config.upsample_factor, 256)
        self.assertEqual(config.fft_size, 1024)
        self.assertTrue(config.use_stochastic_duration_prediction)

    def test_huggingface_config_round_trip_preserves_unknown_metadata(self):
        config = VitsConfig.from_dict({
            "model_type": "vits",
            "vocab_size": 40,
            "torch_dtype": "float32",
            "noise_scale": 0,
        }, )
        payload = config.to_dict()
        self.assertEqual(payload["vocab_size"], 40)
        self.assertEqual(payload["torch_dtype"], "float32")
        self.assertEqual(payload["noise_scale"], 0.0)
        self.assertEqual(payload["model_type"], "vits")

    def test_configuration_rejects_invalid_architecture_geometry(self):
        with self.assertRaisesRegex(ValueError, "flow_size"):
            VitsConfig(flow_size=191)
        with self.assertRaisesRegex(ValueError, "halving"):
            VitsConfig(
                upsample_initial_channel=8,
                upsample_rates=(2, 2, 2, 2),
                upsample_kernel_sizes=(4, 4, 4, 4),
            )
        with self.assertRaisesRegex(ValueError, "spectrogram_bins"):
            VitsConfig(spectrogram_bins=1)
        with self.assertRaisesRegex(ValueError, "speaker_embedding_size"):
            VitsConfig(num_speakers=2, speaker_embedding_size=0)


class VitsFrontendTests(unittest.TestCase):

    def test_mms_character_frontend_adds_blank_token_zero(self):
        tokenizer = VitsTokenizer(
            {
                "k": 0,
                "a": 1,
                "b": 2,
                " ": 3
            },
            config=VitsFrontendConfig(
                language="eng",
                add_blank=True,
                normalize=True,
                phonemize=False,
                pad_token="k",
            ),
        )
        encoded = tokenizer.encode("A b!")
        self.assertEqual(encoded.input_ids, (0, 1, 0, 3, 0, 2, 0))
        self.assertEqual(tokenizer.decode(encoded), "a b")
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertIsNone(tokenizer.unk_token_id)
        self.assertEqual(tokenizer.encode("").input_ids, ())
        self.assertEqual(tokenizer.encode("ka").input_ids, (0, 0, 1, 0))

    def test_required_language_provider_is_never_imported_implicitly(self):
        tokenizer = VitsTokenizer(
            {
                "_": 0,
                "a": 1
            },
            config=VitsFrontendConfig(
                add_blank=True,
                normalize=False,
                phonemize=True,
                pad_token="_",
            ),
        )
        with self.assertRaisesRegex(
                VitsFrontendCapabilityError,
                "TextPhonemizer",
        ):
            tokenizer.encode("a")

    def test_declarative_frontend_assets_load_without_upstream_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / "vocab.json"
            metadata = root / "tokenizer_config.json"
            vocab.write_text(
                json.dumps({
                    "k": 0,
                    "x": 1
                }),
                encoding="utf-8",
            )
            metadata.write_text(
                json.dumps({
                    "language": "eng",
                    "add_blank": True,
                    "normalize": True,
                    "phonemize": False,
                    "pad_token": "k",
                }, ),
                encoding="utf-8",
            )
            tokenizer = VitsTokenizer.from_files(
                vocab,
                tokenizer_config_file=metadata,
            )
        self.assertEqual(tokenizer.encode("X").input_ids, (0, 1, 0))

    def test_frontend_assets_reject_duplicate_json_keys(self):
        from voicehub.architectures.vits.frontend import VitsFrontendAssetError

        with tempfile.TemporaryDirectory() as directory:
            vocab = Path(directory) / "vocab.json"
            vocab.write_text('{"k": 0, "k": 1}', encoding="utf-8")
            with self.assertRaisesRegex(
                    VitsFrontendAssetError,
                    "repeats key",
            ):
                VitsTokenizer.from_files(vocab)


class VitsCheckpointTests(unittest.TestCase):

    def test_real_mms_tts_header_inventory_is_fully_covered(self):
        shapes = native_vits_tensor_shapes(VitsConfig())
        self.assertEqual(len(shapes), 762)
        self.assertEqual(
            safetensors_header_fingerprint(shapes),
            FACEBOOK_MMS_TTS_ENG_HEADER_FINGERPRINT,
        )
        self.assertEqual(
            FACEBOOK_MMS_TTS_ENG_REVISION,
            "c71de0fe7204c83f1c10820a7d696d0b450048ba",
        )

    def test_huggingface_adapter_has_a_strict_identity_plan(self):
        config = VitsConfig().to_dict()
        adapter = HuggingFaceVitsCheckpointAdapter()
        self.assertTrue(adapter.probe((Path("model.safetensors"), ), config), )
        plan = adapter.tensor_plan(config)
        names = frozenset(native_vits_tensor_shapes(config))
        self.assertEqual(plan.source_names, names)
        self.assertEqual(plan.target_names, names)
        self.assertEqual(plan.ignored_source_patterns, ())

    @unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
    def test_declared_shapes_match_the_executable_tiny_graph(self):
        from voicehub.architectures.vits.modeling import VitsModel

        config = _tiny_config(stochastic=True)
        actual = {name: tuple(tensor.shape) for name, tensor in VitsModel(config).state_dict().items()}
        self.assertEqual(native_vits_tensor_shapes(config), actual)


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsAlignmentTests(unittest.TestCase):

    def test_duration_expansion_and_monotonic_search_agree(self):
        from voicehub.architectures.vits.alignment import generate_path, maximum_path

        durations = torch.tensor([[[2.0, 1.0, 2.0]]])
        mask = torch.ones(1, 1, 5, 3)
        expected = torch.tensor([[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]], )
        self.assertTrue(torch.equal(generate_path(durations, mask).squeeze(1), expected), )
        self.assertTrue(
            torch.equal(
                generate_path(durations.long(), mask.bool()).squeeze(1),
                expected.bool(),
            ), )
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            generate_path(torch.tensor([[[2.0, 0.0, 3.0]]]), mask)
        scores = torch.where(
            expected.bool(),
            torch.tensor(8.0),
            torch.tensor(-8.0),
        )
        self.assertTrue(torch.equal(maximum_path(scores, mask[:, 0]), expected), )

    def test_spline_duration_flow_is_numerically_reversible(self):
        from voicehub.architectures.vits.modeling import VitsConvFlow

        torch.manual_seed(4)
        flow = VitsConvFlow(_tiny_config(stochastic=True)).eval()
        value = torch.randn(2, 2, 4) * 0.5
        mask = torch.ones(2, 1, 4)
        condition = torch.randn(2, 8, 4)
        transformed, logdet = flow(value, mask, condition)
        restored, reverse_logdet = flow(
            transformed,
            mask,
            condition,
            reverse=True,
        )
        self.assertTrue(torch.allclose(restored, value, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            torch.allclose(
                logdet + reverse_logdet,
                torch.zeros_like(logdet),
                atol=1e-5,
                rtol=1e-5,
            ), )


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsRuntimeTests(unittest.TestCase):

    def test_request_seed_is_repeatable_and_preserves_global_rng(self):
        from voicehub.architectures.vits.modeling import VitsModel, VitsSamplingConfig

        torch.manual_seed(91)
        model = VitsModel(_tiny_config()).eval()
        input_ids = torch.tensor([[1, 2, 3]])
        state = torch.random.get_rng_state().clone()
        first = model.synthesize(
            input_ids,
            sampling=VitsSamplingConfig(
                seed=7,
                noise_scale=0.3,
                noise_scale_duration=0.0,
                max_output_frames=50,
            ),
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        second = model.synthesize(
            input_ids,
            sampling=VitsSamplingConfig(
                seed=7,
                noise_scale=0.3,
                noise_scale_duration=0.0,
                max_output_frames=50,
            ),
        )
        third = model.synthesize(
            input_ids,
            sampling=VitsSamplingConfig(
                seed=8,
                noise_scale=0.3,
                noise_scale_duration=0.0,
                max_output_frames=50,
            ),
        )
        self.assertTrue(torch.equal(first.waveform, second.waveform))
        self.assertFalse(torch.equal(first.waveform, third.waveform))
        self.assertEqual(first.waveform.shape[0], 1)
        self.assertEqual(
            first.waveform.shape[1],
            first.sequence_lengths.item(),
        )
        self.assertEqual(
            first.sequence_lengths.item(),
            int(first.durations.sum().item()) * 2,
        )

    def test_supervised_generator_graph_runs_mas_and_backward(self):
        from voicehub.architectures.vits.losses import vits_kl_loss
        from voicehub.architectures.vits.modeling import VitsModel

        model = VitsModel(_tiny_config()).train()
        output = model(
            torch.tensor([[1, 2, 3]]),
            spectrogram=torch.randn(1, 5, 5),
            generator=torch.Generator().manual_seed(2),
        )
        self.assertEqual(output.alignment.shape, (1, 1, 5, 3))
        self.assertEqual(output.durations.sum().item(), 5.0)
        self.assertEqual(output.waveform.shape, (1, 10))
        kl = vits_kl_loss(
            output.prior_latents,
            output.posterior_log_variances,
            output.expanded_prior_means,
            output.expanded_prior_log_variances,
            output.spectrogram_mask,
        )
        loss = output.duration_loss + kl + output.waveform.square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.text_encoder.embed_tokens.weight.grad)
        self.assertIsNotNone(model.decoder.conv_post.weight.grad)

    def test_stochastic_duration_training_graph_is_differentiable(self):
        from voicehub.architectures.vits.modeling import VitsModel

        model = VitsModel(_tiny_config(stochastic=True)).train()
        output = model(
            torch.tensor([[1, 2, 3]]),
            spectrogram=torch.randn(1, 5, 5),
            durations=torch.tensor([[[1.0, 2.0, 2.0]]]),
            generator=torch.Generator().manual_seed(3),
        )
        loss = output.duration_loss + output.waveform.square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.duration_predictor.conv_pre.weight.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsLossTests(unittest.TestCase):

    def test_generator_objective_combines_every_source_term(self):
        from voicehub.architectures.vits.losses import VitsGeneratorLoss
        from voicehub.architectures.vits.modeling import VitsTrainingOutput

        latent = torch.zeros(1, 2, 3, requires_grad=True)
        generated_output = torch.zeros(1, 2, requires_grad=True)
        generated_feature = torch.ones(1, 2, requires_grad=True)
        output = VitsTrainingOutput(
            waveform=torch.zeros(1, 6),
            sequence_lengths=torch.tensor([6]),
            alignment=torch.ones(1, 1, 3, 1),
            durations=torch.tensor([[[3.0]]]),
            duration_loss=torch.tensor(2.0),
            posterior_latents=torch.zeros_like(latent),
            prior_latents=latent,
            expanded_prior_means=torch.zeros_like(latent),
            expanded_prior_log_variances=torch.zeros_like(latent),
            posterior_means=torch.zeros_like(latent),
            posterior_log_variances=torch.zeros_like(latent),
            text_mask=torch.ones(1, 1, 1),
            spectrogram_mask=torch.ones(1, 1, 3),
        )
        losses = VitsGeneratorLoss()(
            output,
            mel_reconstruction_loss=torch.tensor(0.5),
            generated_discriminator_outputs=(generated_output, ),
            real_feature_maps=((torch.zeros(1, 2), ), ),
            generated_feature_maps=((generated_feature, ), ),
        )
        self.assertTrue(torch.allclose(losses.total, torch.tensor(26.5)))
        losses.total.backward()
        self.assertIsNotNone(latent.grad)
        self.assertIsNotNone(generated_output.grad)
        self.assertIsNotNone(generated_feature.grad)

    def test_source_loss_equations_and_gradient_boundaries(self):
        from voicehub.architectures.vits.losses import (
            discriminator_loss,
            feature_matching_loss,
            generator_adversarial_loss,
            vits_kl_loss,
        )

        real = torch.tensor([[1.0, 0.0]], requires_grad=True)
        generated = torch.tensor([[0.5, -0.5]], requires_grad=True)
        disc, real_terms, generated_terms = discriminator_loss(
            (real, ),
            (generated, ),
        )
        self.assertTrue(torch.allclose(disc, torch.tensor(0.75)))
        self.assertEqual(len(real_terms), 1)
        self.assertEqual(len(generated_terms), 1)

        adversarial, terms = generator_adversarial_loss((generated, ))
        self.assertTrue(torch.allclose(adversarial, torch.tensor(1.25)))
        self.assertEqual(len(terms), 1)
        feature = feature_matching_loss(
            ((real, ), ),
            ((generated, ), ),
        )
        self.assertTrue(torch.allclose(feature, torch.tensor(1.0)))
        feature.backward(retain_graph=True)
        self.assertIsNone(real.grad)
        self.assertIsNotNone(generated.grad)

        zeros = torch.zeros(1, 2, 3, requires_grad=True)
        kl = vits_kl_loss(
            zeros,
            torch.zeros_like(zeros),
            torch.zeros_like(zeros),
            torch.zeros_like(zeros),
            torch.ones(1, 1, 3),
        )
        self.assertEqual(kl.item(), -1.0)

    def test_small_discriminators_execute_the_reference_topology(self):
        from voicehub.architectures.vits.losses import VitsPeriodDiscriminator, VitsScaleDiscriminator

        waveform = torch.randn(2, 1, 64, requires_grad=True)
        scale = VitsScaleDiscriminator(
            channels=(2, 4, 4, 4, 4, 4),
            groups=(1, 1, 1, 1, 1, 1),
        )
        period = VitsPeriodDiscriminator(
            3,
            channels=(2, 4, 4, 4, 4),
        )
        scale_output, scale_features = scale(waveform)
        period_output, period_features = period(waveform)
        self.assertEqual(len(scale_features), 7)
        self.assertEqual(len(period_features), 6)
        (scale_output.mean() + period_output.mean()).backward()
        self.assertIsNotNone(waveform.grad)

    def test_training_support_distinguishes_recipe_from_checkpoint_metadata(self):
        from voicehub.architectures.vits.losses import VITS_TRAINING_SUPPORT

        self.assertTrue(VITS_TRAINING_SUPPORT.differentiable_generator_graph)
        self.assertTrue(VITS_TRAINING_SUPPORT.monotonic_alignment_search)
        self.assertTrue(VITS_TRAINING_SUPPORT.discriminator_architecture)
        self.assertTrue(VITS_TRAINING_SUPPORT.source_acoustic_frontend)
        self.assertTrue(VITS_TRAINING_SUPPORT.adversarial_optimizer_phases)
        self.assertTrue(VITS_TRAINING_SUPPORT.random_discriminator_initialization)
        self.assertFalse(VITS_TRAINING_SUPPORT.checkpoint_discriminator_weights)
        self.assertFalse(VITS_TRAINING_SUPPORT.checkpoint_acoustic_frontend)
        self.assertTrue(VITS_TRAINING_SUPPORT.full_finetuning_ready)
        self.assertGreaterEqual(
            len(VITS_TRAINING_SUPPORT.blocking_requirements),
            2,
        )


@unittest.skipUnless(TORCH_AVAILABLE, "native VITS requires PyTorch")
class VitsAdversarialTrainingTests(unittest.TestCase):

    @staticmethod
    def _acoustic_config(**overrides):
        values = {
            "sampling_rate": 16_000,
            "filter_length": 8,
            "hop_length": 2,
            "win_length": 8,
            "num_mel_channels": 3,
            "mel_fmin": 0.0,
            "mel_fmax": 8_000.0,
            "segment_size": 8,
        }
        values.update(overrides)
        return values

    def test_acoustic_frontend_matches_its_spectrogram_projection(self):
        from voicehub.architectures.vits.training import VitsAcousticConfig, VitsAcousticFrontend

        config = VitsAcousticConfig.from_mapping(self._acoustic_config())
        frontend = VitsAcousticFrontend(config)
        waveform = torch.linspace(-0.8, 0.8, 16).unsqueeze(0)
        spectrogram = frontend.spectrogram(waveform)
        self.assertEqual(spectrogram.shape, (1, 5, 8))
        torch.testing.assert_close(
            frontend.mel_spectrogram(waveform),
            frontend.spectrogram_to_mel(spectrogram),
        )
        torch.testing.assert_close(
            frontend.spectrogram_lengths(torch.tensor([16])),
            torch.tensor([8]),
        )

    def test_acoustic_config_rejects_unverifiable_or_misaligned_values(self):
        from voicehub.architectures.vits.modeling import VitsModel
        from voicehub.architectures.vits.training import VitsAcousticConfig

        with self.assertRaisesRegex(ValueError, "incomplete"):
            VitsAcousticConfig.from_mapping({"sampling_rate": 16_000})
        with self.assertRaisesRegex(ValueError, "divisible"):
            VitsAcousticConfig.from_mapping(self._acoustic_config(segment_size=7), )
        with self.assertRaisesRegex(ValueError, "upsample factor"):
            VitsAcousticConfig.from_mapping(self._acoustic_config(hop_length=1), ).validate_model(
                VitsModel(_tiny_config()))

    def test_both_source_phases_are_differentiable(self):
        from torch import nn

        from voicehub.architectures.vits.modeling import VitsModel
        from voicehub.architectures.vits.training import VitsAdversarialTrainingModel

        class TinyDiscriminator(nn.Module):

            def __init__(self):
                super().__init__()
                self.hidden = nn.Conv1d(1, 2, 3, padding=1)
                self.output = nn.Conv1d(2, 1, 1)

            def _one(self, waveform):
                hidden = torch.nn.functional.leaky_relu(
                    self.hidden(waveform.unsqueeze(1)),
                    0.1,
                )
                output = self.output(hidden)
                return output.flatten(1), (hidden, output)

            def forward(self, real_waveform, generated_waveform):
                real_output, real_features = self._one(real_waveform)
                generated_output, generated_features = self._one(generated_waveform)
                return (
                    (real_output, ),
                    (generated_output, ),
                    (real_features, ),
                    (generated_features, ),
                )

        training_model = VitsAdversarialTrainingModel(
            VitsModel(_tiny_config()),
            self._acoustic_config(),
            discriminator=TinyDiscriminator(),
        )
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "audio_values": torch.randn(1, 16),
            "generator": torch.Generator().manual_seed(3),
        }

        discriminator_output = training_model.discriminator_step(**inputs)
        self.assertTrue(torch.isfinite(discriminator_output["loss"]))
        discriminator_output["loss"].backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in training_model.discriminator.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in training_model.native_model.parameters()))

        training_model.zero_grad(set_to_none=True)
        generator_output = training_model.generator_step(**inputs)
        self.assertEqual(
            set(generator_output["losses"]),
            {
                "adversarial_loss",
                "duration_loss",
                "feature_matching_loss",
                "generator_loss",
                "kl_loss",
                "mel_reconstruction_loss",
            },
        )
        self.assertTrue(torch.isfinite(generator_output["loss"]))
        generator_output["loss"].backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in training_model.native_model.parameters()))

    def test_adapter_routes_and_freezes_the_two_optimizer_phases(self):
        from torch import nn

        from voicehub.architectures.vits.modeling import VitsModel
        from voicehub.architectures.vits.training import VitsAdversarialTrainingModel
        from voicehub.models.vits.configuration_vits import VitsConfig as PublicVitsConfig
        from voicehub.models.vits.training import NativeVitsGeneratorTrainingAdapter
        from voicehub.training.specs import get_training_spec

        class TinyDiscriminator(nn.Module):

            def __init__(self):
                super().__init__()
                self.hidden = nn.Conv1d(1, 2, 3, padding=1)
                self.output = nn.Conv1d(2, 1, 1)

            def _one(self, waveform):
                hidden = self.hidden(waveform.unsqueeze(1)).tanh()
                output = self.output(hidden)
                return output.flatten(1), (hidden, output)

            def forward(self, real_waveform, generated_waveform):
                real_output, real_features = self._one(real_waveform)
                generated_output, generated_features = self._one(generated_waveform)
                return (
                    (real_output, ),
                    (generated_output, ),
                    (real_features, ),
                    (generated_features, ),
                )

        class Wrapper:

            def __init__(self):
                self.config = PublicVitsConfig(
                    enable_native_adversarial_training=True,
                    training_acoustic_config=(VitsAdversarialTrainingTests._acoustic_config()),
                )
                self.model = VitsModel(_tiny_config())
                self.training_model = VitsAdversarialTrainingModel(
                    self.model,
                    self.config.training_acoustic_config,
                    discriminator=TinyDiscriminator(),
                )

            def load_for_training(self):
                return self

            @staticmethod
            def prepare_training_inputs(inputs, *, phase):
                if phase not in {"discriminator", "generator"}:
                    raise ValueError(phase)
                return dict(inputs)

        wrapper = Wrapper()
        adapter = NativeVitsGeneratorTrainingAdapter(
            wrapper,
            get_training_spec("vits"),
        )
        inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "audio_values": torch.randn(1, 16),
            "generator": torch.Generator().manual_seed(7),
        }
        discriminator = adapter(
            training_phase="discriminator",
            **inputs,
        )
        self.assertEqual(discriminator.optimizer_names, ("discriminator", ))
        discriminator.loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in wrapper.training_model.discriminator.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in wrapper.model.parameters()))

        wrapper.training_model.zero_grad(set_to_none=True)
        generator = adapter(
            training_phase="generator",
            **inputs,
        )
        self.assertEqual(generator.optimizer_names, ("generator", ))
        generator.loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in wrapper.model.parameters()))
        self.assertTrue(
            all(parameter.grad is None for parameter in wrapper.training_model.discriminator.parameters()))


if __name__ == "__main__":
    unittest.main()
