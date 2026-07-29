from __future__ import annotations

import unittest

import torch
from torch.nn import functional

from voicehub.objectives import CTCLoss, Seq2SeqCrossEntropyLoss, ctc_loss, sequence_cross_entropy


class SequenceCrossEntropyTests(unittest.TestCase):

    def test_masked_loss_matches_selected_reference_tokens(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 2.0]],
                [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]],
            ],
            requires_grad=True,
        )
        targets = torch.tensor([[0, 1, -100], [0, 1, 2]])
        mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)

        loss = sequence_cross_entropy(
            logits,
            targets,
            attention_mask=mask,
        )
        selected_logits = torch.stack((logits[0, 0], logits[0, 1], logits[1, 0], logits[1, 2]))
        selected_targets = torch.tensor([0, 1, 0, 2])
        expected = functional.cross_entropy(selected_logits, selected_targets)

        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad[1, 1].abs().sum().item(), 0.0)

    def test_all_masked_batch_returns_differentiable_zero(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        targets = torch.randint(0, 5, (2, 3))

        loss = sequence_cross_entropy(
            logits,
            targets,
            attention_mask=torch.zeros_like(targets),
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.abs().sum().item(), 0.0)

    def test_half_precision_logits_use_float32_loss_math(self):
        logits = torch.randn(2, 3, 5, dtype=torch.float16, requires_grad=True)
        targets = torch.randint(0, 5, (2, 3))

        loss = Seq2SeqCrossEntropyLoss(label_smoothing=0.1)(logits, targets)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_invalid_non_ignored_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            sequence_cross_entropy(
                torch.randn(1, 2, 3),
                torch.tensor([[0, 3]]),
            )

    def test_masked_out_of_vocabulary_target_is_not_materialized(self):
        loss = sequence_cross_entropy(
            torch.randn(1, 2, 3),
            torch.tensor([[1, 999]]),
            attention_mask=torch.tensor([[1, 0]]),
        )
        self.assertTrue(torch.isfinite(loss))


class ConnectionistTemporalClassificationTests(unittest.TestCase):

    @staticmethod
    def _fixture():
        torch.manual_seed(41)
        logits = torch.randn(2, 5, 4, requires_grad=True)
        targets = torch.tensor([[1, 2, 3], [2, 1, 0]])
        input_lengths = torch.tensor([5, 4])
        target_lengths = torch.tensor([3, 2])
        return logits, targets, input_lengths, target_lengths

    def test_native_ctc_matches_pytorch_reference(self):
        logits, targets, input_lengths, target_lengths = self._fixture()

        actual = ctc_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
        )
        expected = functional.ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
            blank=0,
            reduction="none",
            zero_infinity=True,
        )

        torch.testing.assert_close(actual, expected)
        actual.mean().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_time_major_module_matches_batch_major_function(self):
        logits, targets, input_lengths, target_lengths = self._fixture()
        expected = ctc_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
        )
        actual = CTCLoss(time_major=True)(
            logits.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        torch.testing.assert_close(actual, expected)

    def test_half_precision_and_impossible_alignment_remain_finite(self):
        logits = torch.randn(1, 1, 3, dtype=torch.float16, requires_grad=True)
        targets = torch.tensor([[1, 1]])

        loss = ctc_loss(
            logits,
            targets,
            torch.tensor([1]),
            torch.tensor([2]),
            zero_infinity=True,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_concatenated_targets_are_supported(self):
        logits, targets, input_lengths, target_lengths = self._fixture()
        concatenated = torch.cat((targets[0, :3], targets[1, :2]))

        padded_loss = ctc_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
        )
        concatenated_loss = ctc_loss(
            logits,
            concatenated,
            input_lengths,
            target_lengths,
        )
        torch.testing.assert_close(concatenated_loss, padded_loss)

    def test_blank_targets_and_inconsistent_lengths_are_rejected(self):
        logits = torch.randn(1, 3, 4)
        with self.assertRaisesRegex(ValueError, "blank"):
            ctc_loss(
                logits,
                torch.tensor([[0]]),
                torch.tensor([3]),
                torch.tensor([1]),
            )
        with self.assertRaisesRegex(ValueError, "sum"):
            ctc_loss(
                logits,
                torch.tensor([1, 2]),
                torch.tensor([3]),
                torch.tensor([1]),
            )


if __name__ == "__main__":
    unittest.main()
