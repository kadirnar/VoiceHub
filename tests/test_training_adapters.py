import importlib.util
import tempfile
import unittest

from voicehub import (
    AutoInferenceModel,
    AutoTrainingAdapter,
    DataCollatorForTTSTraining,
    PreTrainedTTSModel,
    Trainer,
    TrainingArguments,
    TrainingFamily,
    TTSOutput,
    VoiceHubConfig,
    get_training_spec,
    list_training_specs,
)
from voicehub.training.adapters import (
    AcousticTrainingAdapter,
    CausalLMTrainingAdapter,
    CompositeTrainingAdapter,
    FlowMatchingTrainingAdapter,
    Seq2SeqTrainingAdapter,
)
from voicehub.training.optimization import OptimizerBundle

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

EXPECTED_ADAPTERS = {
    TrainingFamily.CAUSAL_LM: CausalLMTrainingAdapter,
    TrainingFamily.SEQ2SEQ: Seq2SeqTrainingAdapter,
    TrainingFamily.FLOW_MATCHING: FlowMatchingTrainingAdapter,
    TrainingFamily.ACOUSTIC: AcousticTrainingAdapter,
    TrainingFamily.COMPOSITE: CompositeTrainingAdapter,
}


class TrainingProfileTests(unittest.TestCase):

    def test_every_registered_model_has_one_training_profile(self):
        registered = {spec.model_type for spec in AutoInferenceModel.available_models()}
        profiled = {spec.model_type for spec in list_training_specs()}
        self.assertEqual(profiled, registered)
        self.assertEqual(len(profiled), 31)

    def test_registry_connects_models_to_training_profiles(self):
        for model_spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=model_spec.model_type):
                training_spec = model_spec.training
                self.assertEqual(training_spec.model_type, model_spec.model_type)
                self.assertEqual(
                    training_spec.install_extra,
                    model_spec.install_extra,
                )

    def test_all_five_training_families_are_used(self):
        families = {spec.family for spec in list_training_specs()}
        self.assertEqual(families, set(TrainingFamily))

    def test_all_lazy_models_resolve_an_adapter_without_loading(self):
        for model_spec in AutoInferenceModel.available_models():
            with self.subTest(model_type=model_spec.model_type):
                model = AutoInferenceModel.from_pretrained(
                    model_spec.model_type,
                    device="cpu",
                )
                adapter = model.get_training_adapter()
                expected_class = EXPECTED_ADAPTERS[model_spec.training.family]
                self.assertIsInstance(adapter, expected_class)
                self.assertFalse(adapter.is_ready)
                self.assertFalse(model.is_loaded)

    def test_alias_resolves_training_profile(self):
        self.assertEqual(
            get_training_spec("f5-tts").model_type,
            "f5tts",
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class TrainingAdapterLoopTests(unittest.TestCase):

    class DummyForTextToSpeech(PreTrainedTTSModel):

        def __init__(self, config, module_factory):
            self.module_factory = module_factory
            super().__init__(config, device="cpu")

        def _load_pretrained_model(self) -> None:
            self.model = self.module_factory()

        def _generate(self, text: str, **kwargs) -> TTSOutput:
            return TTSOutput(audio=[0.0], sample_rate=24000)

    @staticmethod
    def _config(model_type):
        config = VoiceHubConfig(name_or_path="dummy")
        config.model_type = model_type
        return config

    @staticmethod
    def _token_model():
        import torch

        class TokenModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 8)
                self.projection = torch.nn.Linear(8, 8)

            def forward(self, input_ids):
                return {"logits": self.projection(self.embedding(input_ids))}

        return TokenModel()

    @staticmethod
    def _regression_model():
        import torch

        class RegressionModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values):
                return {"predictions": input_values * self.scale}

        return RegressionModel()

    @staticmethod
    def _composite_model():
        import torch

        class CompositeModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, input_values):
                predictions = input_values * self.scale
                return {
                    "logits": predictions,
                    "loss_dict": {
                        "mel_loss": predictions.square().mean(),
                        "generator_loss": (predictions - 1).square().mean(),
                        "discriminator_loss": (predictions + 1).square().mean(),
                    },
                }

        return CompositeModel()

    @classmethod
    def _composite_runtime(cls):
        import torch

        class Runtime:

            def __init__(self):
                self.generator = cls._composite_model()
                self.model = self.generator
                self.discriminator = torch.nn.Linear(2, 1)

        return Runtime()

    def _train_once(self, model_type, module_factory, dataset):
        model = self.DummyForTextToSpeech(
            self._config(model_type),
            module_factory,
        )
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
            )
            output = trainer.train()
        self.assertEqual(output.global_step, 1)
        self.assertIsNotNone(trainer.training_adapter)
        self.assertTrue(trainer.training_adapter.is_ready)
        return trainer

    def test_token_and_sequence_families_train(self):
        import torch

        dataset = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "labels": torch.tensor([1, 2, 3, 4]),
            },
            {
                "input_ids": torch.tensor([2, 3, 4]),
                "labels": torch.tensor([2, 3, 4]),
            },
        ]
        causal = self._train_once(
            "orpheustts",
            self._token_model,
            dataset,
        )
        sequence = self._train_once(
            "dia",
            self._token_model,
            dataset,
        )
        self.assertIsInstance(
            causal.training_adapter,
            CausalLMTrainingAdapter,
        )
        self.assertIsInstance(
            sequence.training_adapter,
            Seq2SeqTrainingAdapter,
        )

    def test_flow_and_acoustic_families_train(self):
        import torch

        dataset = [
            {
                "input_values": torch.tensor([0.2, 0.4, 0.6]),
                "labels": torch.tensor([0.4, 0.8, 1.2]),
            },
            {
                "input_values": torch.tensor([0.1, 0.3]),
                "labels": torch.tensor([0.2, 0.6]),
            },
        ]
        flow = self._train_once(
            "f5tts",
            self._regression_model,
            dataset,
        )
        acoustic = self._train_once(
            "melotts",
            self._regression_model,
            dataset,
        )
        self.assertIsInstance(
            flow.training_adapter,
            FlowMatchingTrainingAdapter,
        )
        self.assertIsInstance(
            acoustic.training_adapter,
            AcousticTrainingAdapter,
        )

    def test_composite_family_uses_named_source_losses(self):
        import torch

        dataset = [
            {
                "input_values": torch.tensor([0.2, 0.4]),
                "labels": torch.tensor([0.4, 0.8]),
            },
            {
                "input_values": torch.tensor([0.1, 0.3]),
                "labels": torch.tensor([0.2, 0.6]),
            },
        ]
        trainer = self._train_once(
            "styletts2",
            self._composite_runtime,
            dataset,
        )
        self.assertIsInstance(
            trainer.training_adapter,
            CompositeTrainingAdapter,
        )
        self.assertIsInstance(trainer.optimizer, OptimizerBundle)
        self.assertEqual(len(trainer.optimizer.optimizers), 2)

    def test_variable_length_collator_pads_tokens_labels_and_audio(self):
        import torch

        collator = DataCollatorForTTSTraining()
        batch = collator([
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([4, 5, 6]),
                "input_values": torch.tensor([0.1, 0.2, 0.3]),
            },
            {
                "input_ids": torch.tensor([7, 8]),
                "labels": torch.tensor([9, 10]),
                "input_values": torch.tensor([0.4, 0.5]),
            },
        ])
        self.assertEqual(batch["input_ids"].tolist(), [[1, 2, 3], [7, 8, 0]])
        self.assertEqual(
            batch["labels"].tolist(),
            [[4, 5, 6], [9, 10, -100]],
        )
        self.assertEqual(
            batch["attention_mask"].tolist(),
            [[True, True, True], [True, True, False]],
        )
        self.assertEqual(tuple(batch["input_values"].shape), (2, 3))

        spectrograms = collator([
            {
                "input_values": torch.ones(80, 3)
            },
            {
                "input_values": torch.ones(80, 2)
            },
        ])
        self.assertEqual(
            tuple(spectrograms["input_values"].shape),
            (2, 80, 3),
        )

    def test_composite_optimizer_and_scheduler_resume_together(self):
        import torch

        dataset = [
            {
                "input_values": torch.tensor([0.2, 0.4]),
                "labels": torch.tensor([0.4, 0.8]),
            },
            {
                "input_values": torch.tensor([0.1, 0.3]),
                "labels": torch.tensor([0.2, 0.6]),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            first = Trainer(
                model=self.DummyForTextToSpeech(
                    self._config("styletts2"),
                    self._composite_runtime,
                ),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=dataset,
            )
            first.train()

            resumed = Trainer(
                model=self.DummyForTextToSpeech(
                    self._config("styletts2"),
                    self._composite_runtime,
                ),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    use_cpu=True,
                ),
                train_dataset=dataset,
            )
            output = resumed.train(resume_from_checkpoint=True)

        self.assertEqual(output.global_step, 2)
        self.assertIsInstance(resumed.optimizer, OptimizerBundle)

    def test_recursive_discovery_finds_nested_trainable_module(self):

        class Runtime:

            def __init__(self, nested):
                self.nested = nested

        model = self.DummyForTextToSpeech(
            self._config("inflecttts"),
            lambda: Runtime(self._regression_model()),
        )
        adapter = AutoTrainingAdapter.from_model(model)
        adapter.setup()
        self.assertGreater(
            sum(parameter.numel() for parameter in adapter.parameters()),
            0,
        )

    def test_model_init_preserves_explicit_data_collator(self):
        import torch

        collator_calls = []

        def collator(features):
            collator_calls.append(len(features))
            return {
                "input_values": torch.stack([feature["input_values"] for feature in features]),
                "labels": torch.stack([feature["labels"] for feature in features]),
            }

        trainer = Trainer(
            model_init=lambda: self.DummyForTextToSpeech(
                self._config("melotts"),
                self._regression_model,
            ),
            args=TrainingArguments(
                max_steps=1,
                per_device_train_batch_size=2,
                logging_strategy="no",
                save_strategy="no",
                use_cpu=True,
            ),
            data_collator=collator,
            train_dataset=[
                {
                    "input_values": torch.tensor([0.1, 0.2]),
                    "labels": torch.tensor([0.2, 0.4]),
                },
                {
                    "input_values": torch.tensor([0.3, 0.4]),
                    "labels": torch.tensor([0.6, 0.8]),
                },
            ],
        )
        trainer.train()
        self.assertIs(trainer.data_collator, collator)
        self.assertEqual(collator_calls, [2])


if __name__ == "__main__":
    unittest.main()
