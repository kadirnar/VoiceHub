from __future__ import annotations

import unittest

import torch

from voicehub.generation import (
    AutoregressiveGenerator,
    GenerationConfig,
    GenerationStepOutput,
    apply_repetition_penalty,
    evaluate_stopping_criteria,
    filter_min_p,
    filter_top_k,
    filter_top_p,
)


class GenerationConfigurationTests(unittest.TestCase):

    def test_configuration_normalizes_terminal_tokens_and_copies_safely(self):
        config = GenerationConfig(
            max_new_tokens=8,
            eos_token_id=[7, 9],
            top_k=0,
            top_p=0.95,
            min_p=0.05,
        )

        self.assertEqual(config.eos_token_ids, (7, 9))
        self.assertEqual(config.effective_pad_token_id, 7)
        updated = config.with_updates(max_new_tokens=4, pad_token_id=0)
        self.assertEqual(config.max_new_tokens, 8)
        self.assertEqual(updated.max_new_tokens, 4)
        self.assertEqual(updated.effective_pad_token_id, 0)

    def test_configuration_rejects_ambiguous_or_unsafe_values(self):
        invalid_cases = (
            ({
                "max_new_tokens": 0
            }, "max_new_tokens"),
            ({
                "temperature": 0.0
            }, "temperature"),
            ({
                "top_p": 1.1
            }, "top_p"),
            ({
                "min_p": float("nan")
            }, "min_p"),
            ({
                "repetition_penalty": 0.0
            }, "repetition_penalty"),
            ({
                "eos_token_id": ()
            }, "eos_token_id"),
            ({
                "eos_token_id": (2, 2)
            }, "duplicates"),
            ({
                "seed": 2**64
            }, "seed"),
        )
        for values, message in invalid_cases:
            with self.subTest(values=values):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    GenerationConfig(**values)


class LogitTransformTests(unittest.TestCase):

    def test_repetition_penalty_is_sign_aware_and_non_mutating(self):
        logits = torch.tensor([[-2.0, 3.0, 1.0, 0.0]])
        original = logits.clone()
        result = apply_repetition_penalty(
            logits,
            torch.tensor([[0, 1, 1]]),
            penalty=2.0,
        )

        torch.testing.assert_close(logits, original)
        torch.testing.assert_close(
            result,
            torch.tensor([[-4.0, 1.5, 1.0, 0.0]]),
        )

    def test_sampling_filters_retain_their_required_candidates(self):
        probabilities = torch.tensor([[0.60, 0.25, 0.10, 0.05]])
        logits = probabilities.log()

        top_k = filter_top_k(logits, 2)
        self.assertEqual(torch.isfinite(top_k).sum().item(), 2)
        self.assertTrue(torch.isfinite(top_k[0, :2]).all())

        top_p = filter_top_p(logits, 0.70)
        self.assertEqual(torch.isfinite(top_p).sum().item(), 2)
        self.assertTrue(torch.isfinite(top_p[0, :2]).all())

        min_p = filter_min_p(logits, 0.50)
        self.assertEqual(torch.isfinite(min_p).sum().item(), 1)
        self.assertTrue(torch.isfinite(min_p[0, 0]))

    def test_filters_reject_rows_without_a_finite_candidate(self):
        logits = torch.full((1, 3), float("-inf"))
        with self.assertRaisesRegex(ValueError, "finite candidate"):
            filter_top_k(logits, 1)


class AutoregressiveGeneratorTests(unittest.TestCase):

    def test_cache_is_reused_and_eos_stops_each_row_independently(self):
        requests = []

        def decoder_step(request):
            requests.append(request)
            logits = torch.full((2, 4), -20.0)
            if request.step_index == 0:
                logits[0, 2] = 10.0
                logits[1, 1] = 10.0
            else:
                logits[0, 0] = 10.0
                logits[1, 2] = 10.0
            return GenerationStepOutput(
                logits=logits,
                cache=f"cache-{request.step_index}",
            )

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0, 1], [1, 0]]),
            GenerationConfig(
                max_new_tokens=5,
                eos_token_id=2,
                pad_token_id=3,
                use_cache=True,
            ),
        )

        self.assertEqual([tuple(request.token_ids.shape) for request in requests], [(2, 2), (2, 1)])
        self.assertIsNone(requests[0].cache)
        self.assertEqual(requests[1].cache, "cache-0")
        torch.testing.assert_close(
            output.sequences,
            torch.tensor([[0, 1, 2, 3], [1, 0, 1, 2]]),
        )
        torch.testing.assert_close(output.generated_lengths, torch.tensor([1, 2]))
        self.assertTrue(output.finished.all())
        self.assertEqual(output.cache, "cache-1")

    def test_disabled_cache_receives_the_growing_sequence(self):
        sequence_widths = []

        def decoder_step(request):
            sequence_widths.append(request.token_ids.shape[1])
            self.assertIsNone(request.cache)
            self.assertFalse(request.use_cache)
            logits = torch.tensor([[0.0, 1.0, -1.0]])
            return GenerationStepOutput(logits=logits, cache=object())

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0, 2]]),
            GenerationConfig(max_new_tokens=3, use_cache=False),
        )

        self.assertEqual(sequence_widths, [2, 3, 4])
        torch.testing.assert_close(output.sequences, torch.tensor([[0, 2, 1, 1, 1]]))
        self.assertIsNone(output.cache)
        self.assertFalse(output.finished.any())

    def test_seeded_sampling_is_request_local_and_repeatable(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        def decoder_step(request):
            return GenerationStepOutput(logits=logits, cache=request.step_index)

        config = GenerationConfig(
            max_new_tokens=12,
            do_sample=True,
            temperature=0.8,
            top_k=3,
            seed=1234,
        )
        global_state = torch.random.get_rng_state().clone()
        first = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            config,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), global_state)

        torch.rand(17)
        state_after_unrelated_work = torch.random.get_rng_state().clone()
        second = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            config,
        )
        torch.testing.assert_close(torch.random.get_rng_state(), state_after_unrelated_work)
        torch.testing.assert_close(first.sequences, second.sequences)

    def test_three_dimensional_decoder_logits_use_the_last_time_step(self):

        def decoder_step(request):
            logits = torch.tensor([[[9.0, 0.0], [0.0, 9.0]]])
            return GenerationStepOutput(logits=logits)

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            GenerationConfig(max_new_tokens=1),
        )
        self.assertEqual(output.sequences.tolist(), [[0, 1]])

    def test_logits_processors_receive_complete_history_without_mutating_model_logits(self):
        model_logits = torch.tensor([[0.0, 10.0, 9.0]])
        histories = []

        def decoder_step(request):
            del request
            return GenerationStepOutput(logits=model_logits)

        def force_token_two(input_ids, logits):
            histories.append(input_ids.clone())
            logits[:, 1] = float("-inf")
            return logits

        output = AutoregressiveGenerator().generate(
            decoder_step,
            torch.tensor([[0]]),
            GenerationConfig(max_new_tokens=2),
            logits_processors=(force_token_two, ),
        )

        self.assertEqual(output.sequences.tolist(), [[0, 2, 2]])
        self.assertEqual(
            [history.tolist() for history in histories],
            [[[0]], [[0, 2]]],
        )
        torch.testing.assert_close(
            model_logits,
            torch.tensor([[0.0, 10.0, 9.0]]),
        )

    def test_logits_processors_must_preserve_tensor_contract(self):

        def decoder_step(request):
            del request
            return GenerationStepOutput(logits=torch.ones(1, 3))

        invalid_processors = (
            (lambda input_ids, logits: None, "PyTorch tensor"),
            (lambda input_ids, logits: logits[:, :2], "shape"),
            (lambda input_ids, logits: logits.long(), "floating-point"),
        )
        for processor, message in invalid_processors:
            with self.subTest(message=message):
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    AutoregressiveGenerator().generate(
                        decoder_step,
                        torch.tensor([[0]]),
                        GenerationConfig(max_new_tokens=1),
                        logits_processors=(processor, ),
                    )

    def test_stopping_criteria_must_return_boolean_row_decisions(self):

        def invalid_criterion(sequences, next_tokens, step_index):
            del sequences, step_index
            return torch.ones_like(next_tokens)

        with self.assertRaisesRegex(TypeError, "boolean"):
            evaluate_stopping_criteria(
                [invalid_criterion],
                torch.tensor([[0, 1]]),
                torch.tensor([1]),
                0,
            )


if __name__ == "__main__":
    unittest.main()
