import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from voicehub import (
    DefaultDataCollator,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    TTSTrainingOutput,
)

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class EventRecorder(TrainerCallback):

    def __init__(self):
        self.events = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.events.append(("log", state.global_step))
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.events.append(("evaluate", state.global_step))
        return control

    def on_save(self, args, state, control, **kwargs):
        self.events.append(("save", state.global_step))
        return control


class TrainerApiTests(unittest.TestCase):

    def test_training_arguments_round_trip_and_alias(self):
        arguments = TrainingArguments(
            output_dir="trainer-output",
            evaluation_strategy="steps",
            eval_steps=2,
            save_strategy="steps",
            save_steps=2,
            load_best_model_at_end=True,
        )
        self.assertIs(arguments.eval_strategy, IntervalStrategy.STEPS)
        self.assertEqual(arguments.metric_for_best_model, "loss")
        self.assertFalse(arguments.greater_is_better)

        with tempfile.TemporaryDirectory() as directory:
            path = arguments.save_json(Path(directory) / "training_args.json")
            restored = TrainingArguments.from_json_file(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(restored.to_dict(), arguments.to_dict())
        self.assertNotIn("evaluation_strategy", payload)

    def test_training_arguments_validate_incompatible_values(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            TrainingArguments(per_device_train_batch_size=0)
        with self.assertRaisesRegex(ValueError, "At most one"):
            TrainingArguments(fp16=True, bf16=True)
        with self.assertRaisesRegex(ValueError, "must match"):
            TrainingArguments(
                eval_strategy="steps",
                save_strategy="epoch",
                load_best_model_at_end=True,
            )

    def test_trainer_state_round_trip(self):
        state = TrainerState(
            epoch=1.5,
            global_step=12,
            max_steps=20,
            log_history=[{
                "loss": 0.4,
                "step": 12
            }],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = state.save_to_json(Path(directory) / "trainer_state.json")
            restored = TrainerState.load_from_json(path)
        self.assertEqual(restored, state)

    def test_training_output_uses_loss_first_mapping_contract(self):
        output = TTSTrainingOutput(
            loss=0.25,
            logits=[1.0],
            metadata={"architecture": "dummy"},
        )
        self.assertEqual(output[0], 0.25)
        self.assertEqual(output["logits"], [1.0])
        self.assertEqual(
            output.keys(),
            ("loss", "logits", "metadata"),
        )

    def test_control_resets_recurring_actions(self):
        control = TrainerControl(
            should_save=True,
            should_evaluate=True,
            should_log=True,
        )
        control._new_step()
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)
        self.assertFalse(control.should_log)

    def test_early_stopping_uses_best_metric_and_patience(self):
        arguments = TrainingArguments(
            eval_strategy="steps",
            eval_steps=1,
            save_steps=1,
            load_best_model_at_end=True,
        )
        state = TrainerState(best_metric=0.5)
        control = TrainerControl()
        callback = EarlyStoppingCallback(early_stopping_patience=2)
        callback.on_train_begin(arguments, state, control)
        callback.on_evaluate(
            arguments,
            state,
            control,
            metrics={"eval_loss": 0.6},
        )
        self.assertFalse(control.should_training_stop)
        callback.on_evaluate(
            arguments,
            state,
            control,
            metrics={"eval_loss": 0.7},
        )
        self.assertTrue(control.should_training_stop)

    def test_public_trainer_import_remains_framework_lazy(self):
        script = (
            "import sys;"
            "from voicehub import Trainer, TrainingArguments;"
            "print('torch' in sys.modules, 'numpy' in sys.modules)")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "False False")

    def test_checkpoint_rotation_keeps_best_and_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                save_total_limit=1,
            )
            trainer = Trainer(model=object(), args=arguments)
            best = Path(directory) / "checkpoint-1"
            latest = Path(directory) / "checkpoint-2"
            best.mkdir()
            latest.mkdir()
            trainer.state.best_model_checkpoint = str(best)
            trainer._rotate_checkpoints()
            remaining = sorted(path.name for path in Path(directory).glob("checkpoint-*"))
        self.assertEqual(remaining, ["checkpoint-1", "checkpoint-2"])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
    def test_default_collator_stacks_tensors_and_preserves_text(self):
        import numpy
        import torch

        collator = DefaultDataCollator()
        batch = collator([
            {
                "input_values": torch.tensor([1.0]),
                "audio_values": numpy.asarray([0.1, 0.2]),
                "text": "one",
                "label": 1.0,
            },
            {
                "input_values": torch.tensor([2.0]),
                "audio_values": numpy.asarray([0.3, 0.4]),
                "text": "two",
                "label": 2.0,
            },
        ])
        self.assertEqual(tuple(batch["input_values"].shape), (2, 1))
        self.assertEqual(tuple(batch["audio_values"].shape), (2, 2))
        self.assertEqual(batch["text"], ["one", "two"])
        self.assertEqual(batch["labels"].tolist(), [1.0, 2.0])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is an optional training extra")
class TrainerLoopTests(unittest.TestCase):

    @staticmethod
    def _model():
        import torch

        class TinyTTSModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values, labels=None):
                logits = self.projection(input_values)
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.mse_loss(logits, labels)
                return TTSTrainingOutput(loss=loss, logits=logits)

        return TinyTTSModel()

    @staticmethod
    def _dataset():
        import torch

        return [{
            "input_values": torch.tensor([float(index)]),
            "labels": torch.tensor([float(index * 2)]),
            "metadata": f"sample-{index}",
        } for index in range(8)]

    def test_train_evaluate_predict_callbacks_and_checkpoints(self):
        recorder = EventRecorder()
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                max_steps=4,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                learning_rate=0.01,
                logging_steps=1,
                eval_strategy="steps",
                eval_steps=2,
                save_strategy="steps",
                save_steps=2,
                save_total_limit=2,
                load_best_model_at_end=True,
                use_cpu=True,
            )
            trainer = Trainer(
                model=self._model(),
                args=arguments,
                train_dataset=self._dataset(),
                eval_dataset=self._dataset(),
                compute_metrics=lambda prediction:
                {"mae": abs(prediction.predictions - prediction.label_ids).mean()},
                callbacks=[recorder],
            )
            train_output = trainer.train()
            prediction_output = trainer.predict(self._dataset())

            checkpoints = sorted(path.name for path in Path(directory).glob("checkpoint-*"))
            best_checkpoint = trainer.state.best_model_checkpoint

        self.assertEqual(train_output.global_step, 4)
        self.assertGreaterEqual(train_output.training_loss, 0.0)
        self.assertEqual(checkpoints, ["checkpoint-2", "checkpoint-4"])
        self.assertIsNotNone(best_checkpoint)
        self.assertEqual(prediction_output.predictions.shape, (8, 1))
        self.assertIn("test_mae", prediction_output.metrics)
        self.assertIn(("evaluate", 2), recorder.events)
        self.assertIn(("save", 4), recorder.events)

    def test_resume_from_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Trainer(
                model=self._model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=2,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    save_total_limit=4,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
            )
            first.train()

            resumed = Trainer(
                model=self._model(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=4,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_steps=1,
                    save_total_limit=4,
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
            )
            output = resumed.train(resume_from_checkpoint=True)

        self.assertEqual(output.global_step, 4)

    def test_custom_compute_loss_function(self):
        import torch

        class LogitsOnly(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(1, 1)

            def forward(self, input_values):
                return {"logits": self.projection(input_values)}

        received_items = []

        def compute_loss(outputs, labels, num_items_in_batch):
            received_items.append(num_items_in_batch)
            return torch.nn.functional.mse_loss(outputs["logits"], labels)

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=LogitsOnly(),
                args=TrainingArguments(
                    output_dir=directory,
                    max_steps=1,
                    per_device_train_batch_size=2,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=self._dataset(),
                compute_loss_func=compute_loss,
            )
            output = trainer.train()

        self.assertEqual(output.global_step, 1)
        self.assertEqual(received_items, [2])

    def test_incomplete_accumulation_group_keeps_full_loss_scale(self):
        import torch

        class ScalarModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.0]]))

            def forward(self, input_values, labels):
                logits = input_values @ self.weight
                loss = torch.nn.functional.mse_loss(logits, labels)
                return TTSTrainingOutput(loss=loss, logits=logits)

        model = ScalarModel()
        dataset = [{
            "input_values": torch.tensor([1.0]),
            "labels": torch.tensor([1.0]),
        }]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=directory,
                    num_train_epochs=1,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    max_grad_norm=0,
                    logging_strategy="no",
                    save_strategy="no",
                    use_cpu=True,
                ),
                train_dataset=dataset,
                optimizer_cls_and_kwargs=(torch.optim.SGD, {
                    "lr": 1.0
                }),
            )
            trainer.train()

        self.assertAlmostEqual(model.weight.item(), 2.0)

    def test_predict_keeps_first_tuple_value_without_labels(self):
        import torch

        class TupleModel(torch.nn.Module):

            def forward(self, input_values):
                return (input_values * 3, )

        dataset = [{"input_values": torch.tensor([1.0])}, {"input_values": torch.tensor([2.0])}]
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                model=TupleModel(),
                args=TrainingArguments(
                    output_dir=directory,
                    per_device_eval_batch_size=2,
                    use_cpu=True,
                ),
            )
            output = trainer.predict(dataset)

        self.assertEqual(output.predictions.tolist(), [[3.0], [6.0]])


if __name__ == "__main__":
    unittest.main()
