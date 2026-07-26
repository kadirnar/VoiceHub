import importlib.util
import unittest
from dataclasses import dataclass

from voicehub.training.collators import DataCollatorForTTSTraining, TTSFieldSchema

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class TrainingCollatorSchemaTests(unittest.TestCase):

    def test_nested_schema_pads_arbitrary_dimension_and_builds_metadata(self):
        import torch

        collator = DataCollatorForTTSTraining(
            field_schemas={
                "model_inputs.mel":
                TTSFieldSchema(
                    sequence_dim=-1,
                    padding_side="left",
                    length_field="mel_lengths",
                    mask_field="mel_mask",
                    pad_to_multiple_of=4,
                ),
            }, )
        batch = collator([
            {
                "model_inputs": {
                    "mel": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                },
                "training_phase": "flow",
            },
            {
                "model_inputs": {
                    "mel": torch.tensor([[5.0], [6.0]]),
                },
                "training_phase": "flow",
            },
        ])

        self.assertEqual(batch["training_phase"], "flow")
        self.assertEqual(tuple(batch["model_inputs"]["mel"].shape), (2, 2, 4))
        self.assertEqual(
            batch["model_inputs"]["mel"][1].tolist(),
            [[0.0, 0.0, 0.0, 5.0], [0.0, 0.0, 0.0, 6.0]],
        )
        self.assertEqual(
            batch["model_inputs"]["mel_lengths"].tolist(),
            [2, 1],
        )
        self.assertEqual(
            batch["model_inputs"]["mel_mask"].tolist(),
            [
                [False, False, True, True],
                [False, False, False, True],
            ],
        )

    def test_dataclass_samples_and_default_token_metadata(self):
        import torch

        @dataclass
        class Sample:
            input_ids: object
            labels: object
            text: str

        batch = DataCollatorForTTSTraining(return_input_lengths=True)([
            Sample(
                input_ids=torch.tensor([1, 2, 3]),
                labels=torch.tensor([4, 5, 6]),
                text="first",
            ),
            Sample(
                input_ids=torch.tensor([7]),
                labels=torch.tensor([8]),
                text="second",
            ),
        ])

        self.assertEqual(batch["input_lengths"].tolist(), [3, 1])
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[True, True, True], [True, False, False]],
        )
        self.assertEqual(
            batch["labels"].tolist(),
            [[4, 5, 6], [8, -100, -100]],
        )
        self.assertEqual(batch["text"], ["first", "second"])

    def test_default_output_preserves_legacy_token_keys(self):
        import torch

        batch = DataCollatorForTTSTraining()([
            {
                "input_ids": torch.tensor([1, 2])
            },
            {
                "input_ids": torch.tensor([3])
            },
        ])

        self.assertNotIn("input_lengths", batch)
        self.assertIn("attention_mask", batch)

    def test_explicit_input_schema_keeps_enabled_default_metadata(self):
        import torch

        collator = DataCollatorForTTSTraining(
            return_input_lengths=True,
            field_schemas={
                "input_ids": TTSFieldSchema(padding_side="left"),
            },
        )
        batch = collator([
            {
                "input_ids": torch.tensor([1, 2])
            },
            {
                "input_ids": torch.tensor([3])
            },
        ])

        self.assertEqual(batch["input_lengths"].tolist(), [2, 1])
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[True, True], [False, True]],
        )

    def test_optional_nested_schema_uses_zero_length_placeholder(self):
        import torch

        collator = DataCollatorForTTSTraining(
            field_schemas={
                "model_inputs.mel":
                TTSFieldSchema(
                    sequence_dim=-1,
                    allow_missing=True,
                    length_field="mel_lengths",
                    mask_field="mel_mask",
                ),
            }, )
        batch = collator([
            {
                "model_inputs": {
                    "mel": torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                }
            },
            {},
        ])

        self.assertEqual(tuple(batch["model_inputs"]["mel"].shape), (2, 2, 2))
        self.assertEqual(batch["model_inputs"]["mel_lengths"].tolist(), [2, 0])
        self.assertEqual(
            batch["model_inputs"]["mel_mask"].tolist(),
            [[True, True], [False, False]],
        )

    def test_missing_configured_field_is_strict_by_default(self):
        import torch

        collator = DataCollatorForTTSTraining(
            field_schemas={
                "model_inputs.mel": TTSFieldSchema(sequence_dim=-1),
            }, )
        with self.assertRaisesRegex(ValueError, "allow_missing=True"):
            collator([
                {
                    "model_inputs": {
                        "mel": torch.ones(2, 2)
                    }
                },
                {},
            ])

    def test_declared_non_sequence_shape_mismatch_is_actionable(self):
        import torch

        collator = DataCollatorForTTSTraining(
            field_schemas={
                "codec": TTSFieldSchema(sequence_dim=-1),
            }, )
        with self.assertRaisesRegex(ValueError, "outside"):
            collator([
                {
                    "codec": torch.ones(2, 3)
                },
                {
                    "codec": torch.ones(3, 2)
                },
            ])

    def test_mixed_training_phases_are_rejected(self):
        collator = DataCollatorForTTSTraining()
        with self.assertRaisesRegex(ValueError, "same training_phase"):
            collator([
                {
                    "training_phase": "generator"
                },
                {
                    "training_phase": "discriminator"
                },
            ])

    def test_training_phase_objects_remain_scalar_controls(self):
        from voicehub.training.contracts import TrainingPhaseSpec

        phase = TrainingPhaseSpec(name="generator")
        batch = DataCollatorForTTSTraining()([
            {
                "training_phase": phase
            },
            {
                "training_phase": phase
            },
        ])

        self.assertIs(batch["training_phase"], phase)

    def test_missing_training_phase_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must provide"):
            DataCollatorForTTSTraining()([
                {
                    "training_phase": "generator"
                },
                {},
            ])

    def test_unsigned_labels_are_promoted_for_negative_ignore_index(self):
        import torch

        batch = DataCollatorForTTSTraining()([
            {
                "labels": torch.tensor([1, 2], dtype=torch.uint8)
            },
            {
                "labels": torch.tensor([3], dtype=torch.uint8)
            },
        ])

        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["labels"].tolist(), [[1, 2], [3, -100]])

    def test_continuous_labels_use_feature_padding(self):
        import torch

        batch = DataCollatorForTTSTraining()([
            {
                "labels": torch.tensor([1.0, 2.0])
            },
            {
                "labels": torch.tensor([3.0])
            },
        ])

        self.assertEqual(batch["labels"].tolist(), [[1.0, 2.0], [3.0, 0.0]])

    def test_omnivoice_codebook_schema_pads_time_axis(self):
        import torch

        from voicehub.training.specs import get_training_spec

        collator = DataCollatorForTTSTraining(field_schemas=get_training_spec("omnivoice").field_schemas, )
        batch = collator([
            {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "labels": torch.tensor([[7, 8, 9], [10, 11, 12]]),
            },
            {
                "input_ids": torch.tensor([[13], [14]]),
                "labels": torch.tensor([[15], [16]]),
            },
        ])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 2, 3))
        self.assertEqual(batch["input_ids"][1].tolist(), [[13, 0, 0], [14, 0, 0]])
        self.assertEqual(batch["labels"][1].tolist(), [[15, -100, -100], [16, -100, -100]])
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[True, True, True], [True, False, False]],
        )


if __name__ == "__main__":
    unittest.main()
