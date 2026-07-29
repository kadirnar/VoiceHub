from __future__ import annotations

import unittest

import torch

from voicehub.generation import ctc_forced_alignment, ctc_greedy_decode, ctc_prefix_beam_search


class NativeCTCDecodingTests(unittest.TestCase):

    def test_greedy_collapse_resets_repetitions_after_blank(self):
        paths = torch.tensor([[1, 1, 0, 1, 2, 2, 0]])
        logits = torch.full((1, paths.shape[1], 3), -10.0)
        logits.scatter_(2, paths.unsqueeze(-1), 10.0)

        result = ctc_greedy_decode(logits)[0]

        self.assertEqual(result.tokens, (1, 1, 2))
        self.assertEqual(
            tuple((span.start_frame, span.end_frame) for span in result.token_spans),
            ((0, 2), (3, 4), (4, 6)),
        )

    def test_prefix_beam_search_recovers_a_globally_better_sequence(self):
        probabilities = torch.tensor([[
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
            [0.40, 0.35, 0.25],
        ]])
        logits = probabilities.log()

        result = ctc_prefix_beam_search(
            logits,
            blank_id=0,
            beam_size=8,
            token_beam_size=3,
        )[0]

        self.assertTrue(result.tokens)
        self.assertEqual(
            tuple(span.token_id for span in result.token_spans),
            result.tokens,
        )

    def test_hotword_bias_changes_only_complete_phrase_ranking(self):
        probabilities = torch.tensor([[
            [0.10, 0.46, 0.44],
            [0.80, 0.10, 0.10],
        ]])
        logits = probabilities.log()
        unbiased = ctc_prefix_beam_search(
            logits,
            beam_size=4,
            token_beam_size=3,
            return_timestamps=False,
        )[0]
        biased = ctc_prefix_beam_search(
            logits,
            beam_size=4,
            token_beam_size=3,
            hotwords={(2, ): 1.0},
            return_timestamps=False,
        )[0]

        self.assertNotEqual(unbiased.tokens, biased.tokens)
        self.assertEqual(biased.tokens, (2, ))

    def test_forced_alignment_separates_repeated_tokens_with_blank(self):
        path = torch.tensor([1, 0, 1])
        logits = torch.full((3, 3), -10.0)
        logits.scatter_(1, path.unsqueeze(-1), 10.0)

        spans = ctc_forced_alignment(logits, (1, 1), blank_id=0)

        self.assertEqual(
            tuple((span.start_frame, span.end_frame) for span in spans),
            ((0, 1), (2, 3)),
        )

    def test_invalid_lengths_and_impossible_alignments_fail_early(self):
        logits = torch.zeros(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "interval"):
            ctc_greedy_decode(logits, lengths=torch.tensor([3]))
        with self.assertRaisesRegex(ValueError, "Cannot align"):
            ctc_forced_alignment(logits[0], (1, 1), blank_id=0)


if __name__ == "__main__":
    unittest.main()
