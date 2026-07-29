import importlib.util
import subprocess
import sys
import unittest

from voicehub.training.tts_objectives import (
    build_diffusion_training_pair,
    build_flow_matching_training_pair,
    masked_diffusion_regression_loss,
    multi_codebook_cross_entropy,
    vits_discriminator_loss,
    vits_feature_matching_loss,
    vits_generator_adversarial_loss,
    vits_kl_loss,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TTSObjectiveImportTests(unittest.TestCase):

    def test_import_does_not_eagerly_load_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                ("import sys; "
                 "import voicehub.training.tts_objectives; "
                 "print('torch' in sys.modules)"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training dependency")
class TokenObjectiveTests(unittest.TestCase):

    def test_multi_codebook_cross_entropy_is_shifted_masked_and_weighted(self):
        import torch

        logits = torch.tensor(
            [[
                [
                    [3.0, 0.0, -1.0],
                    [0.0, 2.0, -1.0],
                    [0.0, -1.0, 2.0],
                    [1.0, 0.0, -1.0],
                ],
                [
                    [-1.0, 0.0, 3.0],
                    [0.0, 3.0, -1.0],
                    [2.0, 0.0, -1.0],
                    [-1.0, 2.0, 0.0],
                ],
            ]],
            requires_grad=True,
        )
        labels = torch.tensor([[
            [2, 1, -100, 2],
            [0, 1, 0, 1],
        ]])
        loss_mask = torch.tensor(
            [[
                [1, 1, 1, 1],
                [1, 0, 1, 1],
            ]],
            dtype=torch.bool,
        )
        codebook_weights = torch.tensor([1.0, 3.0])

        actual = multi_codebook_cross_entropy(
            logits,
            labels,
            loss_mask=loss_mask,
            causal_shift=True,
            sequence_dim=2,
            codebook_weights=codebook_weights,
            codebook_dim=1,
        )

        shifted_labels = labels[:, :, 1:]
        per_token = torch.nn.functional.cross_entropy(
            logits[:, :, :-1].reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shifted_labels)
        valid = shifted_labels.ne(-100) & loss_mask[:, :, 1:]
        weights = codebook_weights.reshape(1, 2, 1).expand_as(per_token)
        expected = (per_token * valid * weights).sum() / (valid * weights).sum()

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_multi_codebook_cross_entropy_rejects_implicit_broadcasting(self):
        import torch

        logits = torch.randn(2, 3, 5)
        labels = torch.zeros(2, 3, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "exactly match"):
            multi_codebook_cross_entropy(
                logits,
                labels,
                loss_mask=torch.ones(2, 1, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "labels.shape"):
            multi_codebook_cross_entropy(logits[:, :2], labels)

    def test_multi_codebook_cross_entropy_rejects_an_empty_mean(self):
        import torch

        logits = torch.randn(1, 2, 4)
        labels = torch.full((1, 2), -100, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "does not select"):
            multi_codebook_cross_entropy(logits, labels)

    def test_multi_codebook_mask_excludes_nonfinite_padding_and_gradients(self):
        import torch

        logits = torch.tensor(
            [[[2.0, 0.0], [float("nan"), float("nan")]]],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1]])
        mask = torch.tensor([[True, False]])

        loss = multi_codebook_cross_entropy(
            logits,
            labels,
            loss_mask=mask,
        )

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(
            logits.grad[:, 1],
            torch.zeros_like(logits.grad[:, 1]),
        ))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training dependency")
class DiffusionObjectiveTests(unittest.TestCase):

    @staticmethod
    def _coefficients(timesteps, samples):
        import torch

        del timesteps, samples
        return torch.tensor([1.0, 0.25]), torch.tensor([0.0, 0.75])

    def test_diffusion_pair_supports_epsilon_velocity_and_sample_targets(self):
        import torch

        samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        noise = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        timesteps = torch.tensor([0, 1])
        expected_alpha = torch.tensor([[1.0], [0.25]])
        expected_sigma = torch.tensor([[0.0], [0.75]])
        expected_noisy = expected_alpha * samples + expected_sigma * noise

        epsilon = build_diffusion_training_pair(
            samples,
            coefficient_fn=self._coefficients,
            prediction_type="epsilon",
            timesteps=timesteps,
            noise=noise,
        )
        velocity = build_diffusion_training_pair(
            samples,
            coefficient_fn=self._coefficients,
            prediction_type="velocity",
            timesteps=timesteps,
            noise=noise,
        )
        sample = build_diffusion_training_pair(
            samples,
            coefficient_fn=self._coefficients,
            prediction_type="sample",
            timesteps=timesteps,
            noise=noise,
        )

        torch.testing.assert_close(epsilon.noisy_inputs, expected_noisy)
        torch.testing.assert_close(epsilon.targets, noise)
        torch.testing.assert_close(
            velocity.targets,
            expected_alpha * noise - expected_sigma * samples,
        )
        torch.testing.assert_close(sample.targets, samples)
        torch.testing.assert_close(epsilon.alpha, expected_alpha)
        torch.testing.assert_close(epsilon.sigma, expected_sigma)
        torch.testing.assert_close(epsilon.timesteps, timesteps)

    def test_flow_matching_pair_uses_hooks_and_linear_velocity(self):
        import torch

        samples = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
        generator = torch.Generator().manual_seed(7)
        seen = {}

        def timestep_sampler(batch_size, *, device, generator):
            seen["timestep"] = (batch_size, device, generator)
            return torch.tensor([0.25, 0.75], device=device)

        def noise_sampler(values, *, generator):
            seen["noise"] = (values, generator)
            return torch.full_like(values, 6.0)

        pair = build_flow_matching_training_pair(
            samples,
            generator=generator,
            timestep_sampler=timestep_sampler,
            noise_sampler=noise_sampler,
        )

        expected_timesteps = torch.tensor([0.25, 0.75])
        expected_alpha = torch.tensor([[0.75], [0.25]])
        expected_sigma = torch.tensor([[0.25], [0.75]])
        torch.testing.assert_close(pair.timesteps, expected_timesteps)
        torch.testing.assert_close(
            pair.noisy_inputs,
            expected_alpha * samples + expected_sigma * 6.0,
        )
        torch.testing.assert_close(pair.targets, torch.full_like(samples, 6.0) - samples)
        self.assertEqual(seen["timestep"][0], 2)
        self.assertIs(seen["timestep"][2], generator)
        self.assertIs(seen["noise"][0], samples)
        self.assertIs(seen["noise"][1], generator)

    def test_generator_makes_default_flow_sampling_reproducible(self):
        import torch

        samples = torch.zeros(3, 2)
        first = build_flow_matching_training_pair(
            samples,
            generator=torch.Generator().manual_seed(11),
        )
        second = build_flow_matching_training_pair(
            samples,
            generator=torch.Generator().manual_seed(11),
        )

        torch.testing.assert_close(first.timesteps, second.timesteps)
        torch.testing.assert_close(first.noise, second.noise)
        torch.testing.assert_close(first.noisy_inputs, second.noisy_inputs)

    def test_diffusion_pair_requires_exact_noise_and_valid_flow_time(self):
        import torch

        samples = torch.zeros(2, 3)

        with self.assertRaisesRegex(ValueError, "exactly match"):
            build_diffusion_training_pair(
                samples,
                coefficient_fn=self._coefficients,
                timesteps=torch.tensor([0, 1]),
                noise=torch.zeros(2, 1),
            )
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            build_flow_matching_training_pair(
                samples,
                timesteps=torch.tensor([0.0, 1.1]),
                noise=torch.zeros_like(samples),
            )

    def test_discrete_diffusion_rejects_invalid_explicit_and_sampled_timesteps(self):
        import torch

        samples = torch.zeros(2, 3)
        noise = torch.zeros_like(samples)
        invalid = (
            torch.tensor([-1, 1]),
            torch.tensor([0, 10]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, float("nan")]),
        )
        for timesteps in invalid:
            with self.subTest(timesteps=timesteps):
                with self.assertRaises((TypeError, ValueError)):
                    build_diffusion_training_pair(
                        samples,
                        coefficient_fn=self._coefficients,
                        timesteps=timesteps,
                        noise=noise,
                        num_train_timesteps=10,
                    )

        with self.assertRaisesRegex(TypeError, "integer dtype"):
            build_diffusion_training_pair(
                samples,
                coefficient_fn=self._coefficients,
                timestep_sampler=lambda *args, **kwargs: torch.tensor([0.2, 0.8]),
                noise=noise,
                num_train_timesteps=10,
            )

    def test_masked_regression_excludes_padding_and_normalizes_weights(self):
        import torch

        predictions = torch.zeros(2, 2, 2, requires_grad=True)
        targets = torch.tensor([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        mask = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        weights = torch.tensor([1.0, 2.0])

        actual = masked_diffusion_regression_loss(
            predictions,
            targets,
            mask=mask,
            weights=weights,
        )

        expanded_mask = mask.unsqueeze(-1).expand_as(targets)
        expanded_weights = weights.reshape(2, 1, 1).expand_as(targets)
        expected = (targets.square() * expanded_mask *
                    expanded_weights).sum() / (expanded_mask * expanded_weights).sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(predictions.grad)

    def test_masked_regression_rejects_target_broadcasting(self):
        import torch

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            masked_diffusion_regression_loss(
                torch.zeros(2, 3, 4),
                torch.zeros(2, 3, 1),
            )

    def test_masked_regression_excludes_nonfinite_padding_and_gradients(self):
        import torch

        predictions = torch.tensor(
            [[1.0, float("nan")]],
            requires_grad=True,
        )
        targets = torch.tensor([[0.0, float("nan")]])

        loss = masked_diffusion_regression_loss(
            predictions,
            targets,
            mask=torch.tensor([[True, False]]),
        )

        torch.testing.assert_close(loss, torch.tensor(1.0))
        loss.backward()
        self.assertTrue(torch.isfinite(predictions.grad).all())
        self.assertEqual(predictions.grad[0, 1].item(), 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training dependency")
class VITSObjectiveTests(unittest.TestCase):

    def test_discriminator_loss_keeps_both_discriminator_branches_differentiable(self):
        import torch

        real = [
            torch.tensor([[1.0, 0.5]], requires_grad=True),
            torch.tensor([[0.0]], requires_grad=True),
        ]
        fake = [
            torch.tensor([[0.0, 0.25]], requires_grad=True),
            torch.tensor([[0.5]], requires_grad=True),
        ]
        masks = [
            torch.tensor([[1, 0]], dtype=torch.bool),
            torch.tensor([[1]], dtype=torch.bool),
        ]

        result = vits_discriminator_loss(real, fake, masks=masks)

        self.assertEqual(len(result.real_losses), 2)
        self.assertEqual(len(result.fake_losses), 2)
        torch.testing.assert_close(result.loss, torch.tensor(1.25))
        result.loss.backward()
        self.assertIsNotNone(real[0].grad)
        self.assertIsNotNone(real[1].grad)
        self.assertIsNotNone(fake[0].grad)
        self.assertIsNotNone(fake[1].grad)

    def test_generator_adversarial_loss_keeps_generator_gradients(self):
        import torch

        fake = [
            torch.tensor([[0.0, 0.5]], requires_grad=True),
            torch.tensor([[0.25]], requires_grad=True),
        ]

        loss = vits_generator_adversarial_loss(fake)

        expected = torch.tensor((1.0 + 0.25) / 2 + 0.75**2)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(fake[0].grad)
        self.assertIsNotNone(fake[1].grad)

    def test_feature_matching_uses_all_layers_and_detaches_real_features(self):
        import torch

        real = [
            [
                torch.tensor([[0.0, 1.0]], requires_grad=True),
                torch.tensor([[2.0]], requires_grad=True),
            ],
            [torch.tensor([[3.0, 5.0]], requires_grad=True)],
        ]
        fake = [
            [
                torch.tensor([[2.0, 1.0]], requires_grad=True),
                torch.tensor([[5.0]], requires_grad=True),
            ],
            [torch.tensor([[1.0, 4.0]], requires_grad=True)],
        ]

        loss = vits_feature_matching_loss(real, fake)

        expected = torch.tensor(2.0 * (1.0 + 3.0 + 1.5))
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for pyramid in real:
            for feature in pyramid:
                self.assertIsNone(feature.grad)
        for pyramid in fake:
            for feature in pyramid:
                self.assertIsNotNone(feature.grad)

    def test_kl_matches_vits_channel_sum_normalization(self):
        import torch

        latents = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
        zeros = torch.zeros_like(latents)
        mask = torch.tensor([[[1, 0]]], dtype=torch.bool)

        loss = vits_kl_loss(latents, zeros, zeros, zeros, mask=mask)

        # Selected t=0 terms are 0.0 and 4.0. VITS divides their sum by
        # one selected time position, not by the two broadcast channels.
        torch.testing.assert_close(loss, torch.tensor(4.0))
        loss.backward()
        self.assertIsNotNone(latents.grad)

    def test_kl_rejects_shape_mismatch_and_an_empty_mask(self):
        import torch

        values = torch.zeros(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            vits_kl_loss(
                values,
                torch.zeros(1, 1, 3),
                values,
                values,
            )
        with self.assertRaisesRegex(ValueError, "does not select"):
            vits_kl_loss(
                values,
                values,
                values,
                values,
                mask=torch.zeros(1, 1, 3, dtype=torch.bool),
            )

    def test_kl_mask_excludes_nonfinite_padding_and_gradients(self):
        import torch

        latents = torch.tensor(
            [[[1.0, float("nan")]]],
            requires_grad=True,
        )
        zeros = torch.zeros_like(latents)
        mask = torch.tensor([[[True, False]]])

        loss = vits_kl_loss(latents, zeros, zeros, zeros, mask=mask)

        torch.testing.assert_close(loss, torch.tensor(0.0))
        loss.backward()
        self.assertTrue(torch.isfinite(latents.grad).all())
        self.assertEqual(latents.grad[0, 0, 1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
