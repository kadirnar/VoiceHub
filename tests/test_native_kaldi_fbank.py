from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from voicehub.processing.kaldi import (
    GlobalFeatureNormalization,
    KaldiFbank,
    KaldiFbankConfig,
    kaldi_fbank,
    kaldi_mel_filter_bank,
    load_global_cmvn,
)

TORCHAUDIO_AVAILABLE = importlib.util.find_spec("torchaudio") is not None


class KaldiFbankTests(unittest.TestCase):

    def test_config_rejects_invalid_feature_contracts(self):
        with self.assertRaisesRegex(ValueError, "greater than three"):
            KaldiFbankConfig(num_mel_bins=3)
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            KaldiFbankConfig(high_frequency=9_000)
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            KaldiFbankConfig(preemphasis_coefficient=1.1)
        with self.assertRaisesRegex(ValueError, "window_type"):
            KaldiFbankConfig(window_type="cosine")

    def test_batch_frontend_tracks_lengths_and_is_differentiable(self):
        frontend = KaldiFbank(
            KaldiFbankConfig(dither=0.0),
            waveform_scale=32_768.0,
        )
        waveforms = torch.randn(2, 4_000, requires_grad=True)
        features, lengths = frontend(
            waveforms,
            torch.tensor([4_000, 3_200]),
        )
        self.assertEqual(features.shape, (2, 23, 80))
        self.assertEqual(lengths.tolist(), [23, 18])
        self.assertTrue(torch.equal(features[1, 18:], torch.zeros_like(features[1, 18:])))
        features.square().mean().backward()
        self.assertIsNotNone(waveforms.grad)
        self.assertTrue(torch.isfinite(waveforms.grad).all())

    def test_global_normalization_broadcasts_over_leading_dimensions(self):
        normalizer = GlobalFeatureNormalization(
            torch.tensor([1.0, -1.0]),
            torch.tensor([2.0, 0.5]),
        )
        values = torch.tensor([[[2.0, 3.0], [0.0, -1.0]]])
        actual = normalizer(values)
        expected = torch.tensor([[[2.0, 2.0], [-2.0, 0.0]]])
        torch.testing.assert_close(actual, expected)

    def test_json_cmvn_loader_matches_accumulated_moment_formula(self):
        payload = {
            "mean_stat": [8.0, -4.0],
            "var_stat": [20.0, 20.0],
            "frame_num": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_cmvn"
            path.write_text(json.dumps(payload), encoding="utf-8")
            normalizer = load_global_cmvn(
                path,
                expected_dimension=2,
            )
        torch.testing.assert_close(
            normalizer.mean,
            torch.tensor([2.0, -1.0]),
        )
        torch.testing.assert_close(
            normalizer.inverse_std,
            torch.tensor([1.0, 0.5]),
        )

    def test_text_kaldi_cmvn_loader_matches_json_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_cmvn.txt"
            path.write_text(
                "[ 8 -4 4 20 20 0 ]",
                encoding="utf-8",
            )
            normalizer = load_global_cmvn(path, format="kaldi")
        torch.testing.assert_close(
            normalizer.mean,
            torch.tensor([2.0, -1.0]),
        )
        torch.testing.assert_close(
            normalizer.inverse_std,
            torch.tensor([1.0, 0.5]),
        )

    def test_cmvn_loader_rejects_malformed_or_mismatched_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "global_cmvn"
            path.write_text(
                json.dumps({
                    "mean_stat": [1.0, 2.0],
                    "var_stat": [3.0],
                    "frame_num": 1,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identical dimensions"):
                load_global_cmvn(path)
            with self.assertRaisesRegex(ValueError, "Expected 3"):
                path.write_text(
                    json.dumps({
                        "mean_stat": [1.0, 2.0],
                        "var_stat": [3.0, 4.0],
                        "frame_num": 1,
                    }),
                    encoding="utf-8",
                )
                load_global_cmvn(path, expected_dimension=3)

    def test_mel_bank_has_expected_shape_and_finite_weights(self):
        config = KaldiFbankConfig(num_mel_bins=40)
        weights = kaldi_mel_filter_bank(config)
        self.assertEqual(
            weights.shape,
            (40, config.padded_window_size // 2),
        )
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())

    @unittest.skipUnless(
        TORCHAUDIO_AVAILABLE,
        "TorchAudio is used only as an audit oracle",
    )
    def test_randomized_outputs_match_torchaudio_reference(self):
        from torchaudio.compliance import kaldi as reference

        cases = (
            KaldiFbankConfig(),
            KaldiFbankConfig(
                num_mel_bins=40,
                frame_length=20.0,
                frame_shift=5.0,
                window_type="hamming",
                use_energy=True,
                htk_compatibility=True,
                energy_floor=1.0,
            ),
            KaldiFbankConfig(
                num_mel_bins=64,
                snip_edges=False,
                window_type="blackman",
                raw_energy=False,
                use_power=False,
                use_log_fbank=False,
                subtract_mean=True,
            ),
            KaldiFbankConfig(
                num_mel_bins=32,
                vtln_warp=1.1,
                vtln_low=100.0,
                vtln_high=-500.0,
            ),
        )
        generator = torch.Generator().manual_seed(7)
        for case_index, config in enumerate(cases):
            for sample_count in (4_001, 8_123, 16_000):
                with self.subTest(
                        case=case_index,
                        sample_count=sample_count,
                ):
                    waveform = torch.randn(
                        1,
                        sample_count,
                        generator=generator,
                    )
                    expected = reference.fbank(
                        waveform,
                        blackman_coeff=config.blackman_coefficient,
                        dither=config.dither,
                        energy_floor=config.energy_floor,
                        frame_length=config.frame_length,
                        frame_shift=config.frame_shift,
                        high_freq=config.high_frequency,
                        htk_compat=config.htk_compatibility,
                        low_freq=config.low_frequency,
                        min_duration=config.minimum_duration,
                        num_mel_bins=config.num_mel_bins,
                        preemphasis_coefficient=(config.preemphasis_coefficient),
                        raw_energy=config.raw_energy,
                        remove_dc_offset=config.remove_dc_offset,
                        round_to_power_of_two=(config.round_to_power_of_two),
                        sample_frequency=config.sample_frequency,
                        snip_edges=config.snip_edges,
                        subtract_mean=config.subtract_mean,
                        use_energy=config.use_energy,
                        use_log_fbank=config.use_log_fbank,
                        use_power=config.use_power,
                        vtln_high=config.vtln_high,
                        vtln_low=config.vtln_low,
                        vtln_warp=config.vtln_warp,
                        window_type=config.window_type,
                    )
                    actual = kaldi_fbank(waveform, config)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        rtol=1e-5,
                        atol=1e-5,
                    )


if __name__ == "__main__":
    unittest.main()
