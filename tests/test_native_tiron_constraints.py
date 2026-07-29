from __future__ import annotations

import math
import unittest

import torch

from voicehub.models.asr_tiron.constraints import (
    EOS_TOKEN_ID,
    MAX_INITIAL_TIMESTAMP_INDEX,
    NO_SPEECH_TOKEN_ID,
    NO_TIMESTAMPS_TOKEN_ID,
    SPEAKER_TOKEN_IDS,
    TIMESTAMP_BEGIN_ID,
    TIMESTAMP_END_ID,
    TironConstraintLogitsProcessor,
)

VOCABULARY_SIZE = 51_904
PROMPT = [50_258, 50_259, 50_360]


def _processor(**overrides):
    return TironConstraintLogitsProcessor(**overrides)


def _process(
    generated,
    *,
    processor=None,
    scores=None,
):
    processor = processor or _processor()
    input_ids = torch.tensor(
        [[*PROMPT, *generated]],
        dtype=torch.long,
    )
    if scores is None:
        scores = torch.zeros((1, VOCABULARY_SIZE))
    return processor(input_ids, scores)[0]


def _finite_ids(row):
    return {index for index in range(row.shape[0]) if math.isfinite(row[index].item())}


class NativeTironConstraintTests(unittest.TestCase):

    def test_first_token_is_speaker_one_or_native_no_speech(self):
        allowed = _finite_ids(_process([]))

        self.assertEqual(
            allowed,
            {SPEAKER_TOKEN_IDS[0], NO_SPEECH_TOKEN_ID},
        )
        without_silence = _finite_ids(_process(
            [],
            processor=_processor(allow_initial_no_speech=False),
        ))
        self.assertEqual(without_silence, {SPEAKER_TOKEN_IDS[0]})

    def test_speaker_forces_opening_timestamp_and_caps_first_position(self):
        allowed = _finite_ids(_process([SPEAKER_TOKEN_IDS[0]]))

        self.assertEqual(
            allowed,
            set(range(
                TIMESTAMP_BEGIN_ID,
                TIMESTAMP_BEGIN_ID + MAX_INITIAL_TIMESTAMP_INDEX + 1,
            )),
        )
        self.assertNotIn(NO_TIMESTAMPS_TOKEN_ID, allowed)
        self.assertNotIn(EOS_TOKEN_ID, allowed)

    def test_opening_timestamp_forces_ordinary_text(self):
        allowed = _finite_ids(_process([
            SPEAKER_TOKEN_IDS[0],
            TIMESTAMP_BEGIN_ID + 5,
        ]))

        self.assertEqual(allowed, set(range(EOS_TOKEN_ID)))

    def test_closing_timestamp_allows_continuation_eos_or_next_speaker(self):
        allowed = _finite_ids(
            _process([
                SPEAKER_TOKEN_IDS[0],
                TIMESTAMP_BEGIN_ID + 5,
                500,
                TIMESTAMP_BEGIN_ID + 20,
            ]))

        self.assertEqual(
            allowed,
            {
                EOS_TOKEN_ID,
                SPEAKER_TOKEN_IDS[1],
                *range(TIMESTAMP_BEGIN_ID, TIMESTAMP_END_ID + 1),
            },
        )
        self.assertNotIn(SPEAKER_TOKEN_IDS[0], allowed)
        self.assertNotIn(SPEAKER_TOKEN_IDS[2], allowed)

    def test_speaker_cap_removes_new_slots(self):
        allowed = _finite_ids(
            _process(
                [
                    SPEAKER_TOKEN_IDS[0],
                    TIMESTAMP_BEGIN_ID + 5,
                    500,
                    TIMESTAMP_BEGIN_ID + 20,
                ],
                processor=_processor(max_speakers=1),
            ))

        self.assertEqual(
            allowed,
            {
                EOS_TOKEN_ID,
                *range(TIMESTAMP_BEGIN_ID, TIMESTAMP_END_ID + 1),
            },
        )

    def test_no_speech_forces_eos_and_padded_rows_are_never_emitted(self):
        self.assertEqual(
            _finite_ids(_process([NO_SPEECH_TOKEN_ID])),
            {EOS_TOKEN_ID},
        )
        row = _process([
            SPEAKER_TOKEN_IDS[0],
            TIMESTAMP_BEGIN_ID,
            500,
        ])
        self.assertTrue(torch.isneginf(row[SPEAKER_TOKEN_IDS[-1] + 1:]).all())

    def test_ngram_repetition_is_blocked_without_affecting_other_text(self):
        first, second = 400, 401
        generated = [
            SPEAKER_TOKEN_IDS[0],
            TIMESTAMP_BEGIN_ID + 5,
            first,
            second,
            first,
            second,
        ]
        scores = torch.zeros((1, VOCABULARY_SIZE))
        scores[
            0,
            TIMESTAMP_BEGIN_ID:TIMESTAMP_END_ID + 1,
        ] = -100.0
        row = _process(
            generated,
            processor=_processor(no_repeat_ngram_size=3),
            scores=scores,
        )

        self.assertTrue(torch.isneginf(row[first]))
        self.assertTrue(torch.isfinite(row[402]))

    def test_unigram_repetition_blocks_every_previously_generated_token(self):
        scores = torch.zeros((1, VOCABULARY_SIZE))
        scores[
            0,
            TIMESTAMP_BEGIN_ID:TIMESTAMP_END_ID + 1,
        ] = -100.0
        row = _process(
            [
                SPEAKER_TOKEN_IDS[0],
                TIMESTAMP_BEGIN_ID + 5,
                400,
                401,
            ],
            processor=_processor(no_repeat_ngram_size=1),
            scores=scores,
        )

        self.assertTrue(torch.isneginf(row[400]))
        self.assertTrue(torch.isneginf(row[401]))
        self.assertTrue(torch.isfinite(row[402]))

    def test_rows_derive_grammar_state_independently(self):
        input_ids = torch.tensor([
            [
                *PROMPT,
                SPEAKER_TOKEN_IDS[0],
                TIMESTAMP_BEGIN_ID + 5,
            ],
            [
                *PROMPT,
                500,
                TIMESTAMP_BEGIN_ID + 20,
            ],
        ])
        output = _processor()(
            input_ids,
            torch.zeros((2, VOCABULARY_SIZE)),
        )

        self.assertEqual(
            _finite_ids(output[0]),
            set(range(EOS_TOKEN_ID)),
        )
        self.assertIn(EOS_TOKEN_ID, _finite_ids(output[1]))
        self.assertIn(SPEAKER_TOKEN_IDS[0], _finite_ids(output[1]))


if __name__ == "__main__":
    unittest.main()
