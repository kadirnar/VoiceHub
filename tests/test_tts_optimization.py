import importlib.util
import unittest

from voicehub import (
    DiffusionTTSOptimizationConfig,
    EpochLengthBatchSampler,
    LLMTTSOptimizationConfig,
    Trainer,
    TrainingArguments,
    TTSBatchingConfig,
    TTSBatchingStrategy,
    TTSDataset,
    VITSOptimizationConfig,
    get_tts_training_optimization_profile,
)
from voicehub.trainer_utils import get_scheduler_lambda

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class TTSOptimizationProfileTests(unittest.TestCase):

    def test_model_and_architecture_resolution_stays_separate(self):
        self.assertIsInstance(
            get_tts_training_optimization_profile("vits"),
            VITSOptimizationConfig,
        )
        conversation = get_tts_training_optimization_profile("conversationtts")
        qwen = get_tts_training_optimization_profile("qwen3tts")
        diffusion = get_tts_training_optimization_profile("diffusion")

        self.assertIsInstance(conversation, LLMTTSOptimizationConfig)
        self.assertEqual(conversation.recipe, "conversationtts")
        self.assertEqual(qwen.recipe, "qwen3tts")
        self.assertIsInstance(diffusion, DiffusionTTSOptimizationConfig)
        with self.assertRaisesRegex(ValueError, "No source-verified"):
            get_tts_training_optimization_profile("dia")
        with self.assertRaisesRegex(ValueError, "not make another model"):
            get_tts_training_optimization_profile("irodoritts")

    def test_source_profiles_build_distinct_training_arguments(self):
        vits = VITSOptimizationConfig().training_arguments("vits-output")
        conversation = LLMTTSOptimizationConfig().training_arguments("lm-output")
        qwen = LLMTTSOptimizationConfig.qwen3tts().training_arguments("qwen-output")
        diffusion = DiffusionTTSOptimizationConfig().training_arguments("flow-output")

        self.assertEqual(vits.learning_rate, 2e-4)
        self.assertEqual((vits.adam_beta1, vits.adam_beta2), (0.8, 0.99))
        self.assertEqual(vits.lr_scheduler_type.value, "exponential")
        self.assertEqual(vits.lr_scheduler_gamma, 0.999875)
        self.assertEqual(vits.gradient_accumulation_steps, 1)
        self.assertEqual(conversation.warmup_ratio, 0.03)
        self.assertEqual(conversation.lr_scheduler_type.value, "cosine")
        self.assertTrue(conversation.adamw_fused)
        self.assertEqual(qwen.gradient_accumulation_steps, 4)
        self.assertEqual(qwen.learning_rate, 2e-6)
        self.assertEqual(diffusion.warmup_steps, 20_000)
        self.assertTrue(diffusion.gradient_checkpointing)

    def test_profiles_attach_only_to_compatible_datasets(self):
        vits = TTSDataset(
            [{
                "text": "one",
                "audio": "one.wav",
                "num_frames": 100,
            }],
            model_type="vits",
        )
        optimized = VITSOptimizationConfig().prepare_dataset(vits)

        self.assertIsNone(vits.batching)
        self.assertIsNotNone(optimized.batching)
        self.assertEqual(
            optimized.batching.strategy,
            TTSBatchingStrategy.LENGTH_BUCKET,
        )
        with self.assertRaisesRegex(ValueError, "codec-LM"):
            LLMTTSOptimizationConfig().prepare_dataset(vits)

    def test_duration_metadata_can_be_converted_to_frames(self):
        dataset = TTSDataset(
            [{
                "text": "one",
                "audio": "one.wav",
                "duration": 1.5,
            }],
            model_type="vits",
        )
        optimized = VITSOptimizationConfig().prepare_dataset(
            dataset,
            length_field="duration",
            length_multiplier=22_050 / 256,
        )
        sampler = optimized.create_batch_sampler(
            batch_size=1,
            seed=3,
            shuffle=False,
            drop_last=False,
        )

        self.assertEqual(sampler.lengths, (130, ))

    def test_profile_metadata_is_inspectable(self):
        profile = DiffusionTTSOptimizationConfig()

        self.assertIn("activation checkpointing", profile.techniques)
        self.assertIn("github.com/SWivid/F5-TTS", profile.source_url)
        self.assertEqual(
            profile.model_config_overrides()["architecture"],
            {"checkpoint_activations": True},
        )


class TTSLengthBatchSamplerTests(unittest.TestCase):

    @staticmethod
    def _records(*lengths):
        return tuple({"length": length} for length in lengths)

    def test_bucket_batches_cover_overflow_and_resume_deterministically(self):
        config = TTSBatchingConfig(
            strategy="bucket",
            length_field="length",
            bucket_boundaries=(10, 20),
        )
        sampler = EpochLengthBatchSampler(
            self._records(5, 8, 11, 19, 21, 40),
            config,
            batch_size=2,
            seed=17,
            shuffle=True,
            drop_last=False,
        )
        epoch_zero = list(sampler)
        sampler.set_epoch(1)
        epoch_one = list(sampler)
        state = sampler.state_dict()
        restored = EpochLengthBatchSampler(
            self._records(5, 8, 11, 19, 21, 40),
            config,
            batch_size=2,
            seed=17,
            shuffle=True,
            drop_last=False,
        )
        restored.load_state_dict(state)

        self.assertEqual(
            sorted(index for batch in epoch_zero for index in batch),
            list(range(6)),
        )
        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(list(restored), epoch_one)

    def test_budget_batches_respect_cost_and_keep_oversized_singletons(self):
        config = TTSBatchingConfig(
            strategy="token-budget",
            length_field="length",
            max_batch_units=30,
            max_samples=4,
            budget_mode="sum",
        )
        sampler = EpochLengthBatchSampler(
            self._records(10, 15, 20, 100),
            config,
            batch_size=4,
            seed=5,
            shuffle=False,
            drop_last=False,
        )
        batches = list(sampler)

        self.assertEqual(
            sorted(index for batch in batches for index in batch),
            list(range(4)),
        )
        for batch in batches:
            cost = sum(sampler.lengths[index] for index in batch)
            self.assertTrue(cost <= 30 or len(batch) == 1)
        oversized = next(batch for batch in batches if 3 in batch)
        self.assertEqual(oversized, [3])

    def test_padded_budget_accounts_for_padding(self):
        sampler = EpochLengthBatchSampler(
            self._records(10, 20, 21),
            TTSBatchingConfig(
                strategy="max-units",
                length_field="length",
                max_batch_units=40,
                budget_mode="padded",
            ),
            batch_size=3,
            seed=0,
            shuffle=False,
            drop_last=False,
        )

        self.assertEqual(list(sampler), [[0, 1], [2]])

    def test_invalid_or_changed_length_state_fails_closed(self):
        config = TTSBatchingConfig(
            strategy="max-units",
            length_field="length",
            max_batch_units=30,
            max_sequence_length=20,
        )
        with self.assertRaisesRegex(ValueError, "exceeding"):
            EpochLengthBatchSampler(
                self._records(21),
                config,
                batch_size=2,
                seed=0,
                shuffle=False,
                drop_last=False,
            )

        base = TTSBatchingConfig(
            strategy="bucket",
            length_field="length",
            bucket_boundaries=(10, ),
        )
        original = EpochLengthBatchSampler(
            self._records(5, 8),
            base,
            batch_size=2,
            seed=0,
            shuffle=True,
            drop_last=False,
        )
        changed = EpochLengthBatchSampler(
            self._records(5, 9),
            base,
            batch_size=2,
            seed=0,
            shuffle=True,
            drop_last=False,
        )
        with self.assertRaisesRegex(ValueError, "length_sha256"):
            changed.load_state_dict(original.state_dict())

    def test_dataset_resume_identity_includes_batching(self):
        dataset = TTSDataset(
            [{
                "text": "one",
                "audio": "one.wav",
                "num_frames": 100,
            }],
            model_type="vits",
        )
        optimized = dataset.with_batching(VITSOptimizationConfig().batching_config())

        self.assertNotEqual(
            dataset.resume_fingerprint()["content_sha256"],
            optimized.resume_fingerprint()["content_sha256"],
        )
        self.assertIsNotNone(optimized.resume_fingerprint()["batching"])


class SchedulerOptimizationTests(unittest.TestCase):

    def test_new_optimizer_arguments_validate_and_round_trip(self):
        arguments = TrainingArguments(
            adamw_fused=True,
            lr_scheduler_type="exponential",
            lr_scheduler_gamma=0.95,
        )

        self.assertTrue(arguments.to_dict()["adamw_fused"])
        self.assertEqual(arguments.to_dict()["lr_scheduler_gamma"], 0.95)
        with self.assertRaisesRegex(TypeError, "adamw_fused"):
            TrainingArguments(adamw_fused="yes")
        with self.assertRaisesRegex(ValueError, "lr_scheduler_gamma"):
            TrainingArguments(lr_scheduler_gamma=0.0)

    def test_exponential_gamma_is_normalized_per_epoch(self):
        schedule = get_scheduler_lambda(
            "exponential",
            num_warmup_steps=0,
            num_training_steps=100,
            exponential_gamma=0.9,
            num_train_epochs=2,
        )

        self.assertAlmostEqual(schedule(0), 1.0)
        self.assertAlmostEqual(schedule(50), 0.9)
        self.assertAlmostEqual(schedule(100), 0.81)


@unittest.skipUnless(TORCH_AVAILABLE, "optimizer integration requires PyTorch")
class FusedAdamWTests(unittest.TestCase):

    def test_fused_adamw_request_falls_back_portably_on_cpu(self):
        import torch

        model = torch.nn.Linear(2, 1)
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                use_cpu=True,
                adamw_fused=True,
            ),
        )
        optimizer = trainer.create_optimizer()

        self.assertFalse(bool(optimizer.defaults.get("fused", False)))


if __name__ == "__main__":
    unittest.main()
