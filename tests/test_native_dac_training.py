from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from voicehub.components.audio.codecs.dac.model.discriminator import (
    Discriminator,
    MPD,
    MRD,
)
from voicehub.components.audio.codecs.dac.nn.loss import (
    GANLoss,
    MelSpectrogramLoss,
    MultiScaleSTFTLoss,
    SISDRLoss,
)
from voicehub.policies.architecture_dependencies import inspect_native_imports


class NativeDACTrainingTests(unittest.TestCase):

    def test_period_and_resolution_discriminators_are_differentiable(self):
        waveform = torch.randn(1, 1, 256, requires_grad=True)
        period_features = MPD(3)(waveform)
        resolution_features = MRD(
            64,
            bands=((0.0, 0.5), (0.5, 1.0)),
        )(waveform)

        self.assertEqual(len(period_features), 6)
        self.assertEqual(len(resolution_features), 11)
        loss = period_features[-1].mean() + resolution_features[-1].mean()
        loss.backward()
        self.assertIsNotNone(waveform.grad)
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_composite_discriminator_preserves_published_weight_names(self):
        discriminator = Discriminator(
            periods=(2,),
            fft_sizes=(64,),
            sample_rate=16_000,
            bands=((0.0, 0.5), (0.5, 1.0)),
        )
        names = set(discriminator.state_dict())

        self.assertIn(
            "discriminators.0.convs.0.0.weight_g",
            names,
        )
        self.assertIn(
            "discriminators.1.band_convs.0.0.0.weight_v",
            names,
        )
        outputs = discriminator(torch.randn(1, 1, 256))
        self.assertEqual(len(outputs), 2)

    def test_native_spectral_losses_have_finite_input_gradients(self):
        estimate = torch.randn(2, 1, 256, requires_grad=True)
        reference = torch.randn_like(estimate)
        stft_loss = MultiScaleSTFTLoss(
            window_lengths=(64, 32),
            match_stride=True,
        )
        mel_loss = MelSpectrogramLoss(
            n_mels=(8, 4),
            window_lengths=(64, 32),
            match_stride=True,
            mel_fmin=(0.0, 0.0),
            mel_fmax=(7_500.0, 7_500.0),
            sample_rate=16_000,
        )

        loss = stft_loss(estimate, reference) + mel_loss(estimate, reference)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(estimate.grad)
        self.assertTrue(torch.isfinite(estimate.grad).all())

    def test_si_sdr_identity_is_better_than_noise(self):
        reference = torch.randn(2, 1, 128)
        objective = SISDRLoss()

        identity = objective(reference, reference + 1e-4)
        noise = objective(reference, torch.randn_like(reference))

        self.assertLess(float(identity), float(noise))

    def test_gan_loss_detaches_fake_for_discriminator_step(self):
        class TinyDiscriminator(nn.Module):

            def forward(self, waveform):
                hidden = waveform * 2.0
                return [[hidden, hidden.mean(dim=-1, keepdim=True)]]

        fake = torch.randn(1, 1, 32, requires_grad=True)
        real = torch.randn_like(fake)
        objective = GANLoss(TinyDiscriminator())

        discriminator_loss = objective.discriminator_loss(fake, real)
        self.assertFalse(discriminator_loss.requires_grad)
        adversarial, feature_matching = objective.generator_loss(fake, real)
        (adversarial + feature_matching).backward()
        self.assertIsNotNone(fake.grad)

    def test_dac_training_graph_has_no_external_model_imports(self):
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "voicehub/components/audio/codecs/_compat.py",
            root
            / "voicehub/components/audio/codecs/dac/model/discriminator.py",
            root / "voicehub/components/audio/codecs/dac/nn/loss.py",
        )
        violations = tuple(
            violation
            for path in paths
            for violation in inspect_native_imports(path)
        )
        self.assertEqual(
            violations,
            (),
            "\n".join(str(violation) for violation in violations),
        )


if __name__ == "__main__":
    unittest.main()
