# Trainer

VoiceHub includes a PyTorch training API with the same core vocabulary as
Transformers:

```text
TrainingArguments
Trainer
TrainerState / TrainerControl / TrainerCallback
EvalPrediction / TrainOutput / PredictionOutput
DefaultDataCollator
TTSTrainingOutput
```

PyTorch remains optional. Install the training extra only for training:

```bash
python -m pip install "voicehub[training]"
```

## Basic training

A trainable model returns `TTSTrainingOutput` with `loss` as its first field.
It may be a source-integrated VoiceHub architecture or any `torch.nn.Module`
that follows the same contract.

```python
from voicehub import Trainer, TrainingArguments

arguments = TrainingArguments(
    output_dir="runs/my-tts-model",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=250,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=processor,
    compute_metrics=compute_metrics,
)
trainer.train(resume_from_checkpoint=True)
```

Dataset samples are mappings. Their keys are passed to `model.forward`;
`labels` is the default target key. `DefaultDataCollator` stacks fixed-size
tensors and numeric values while preserving text and other metadata as lists.
Variable-length audio and token sequences should use an
architecture-specific padding collator passed as `data_collator`.

## Loss contract

The default path accepts any of these outputs:

```python
TTSTrainingOutput(loss=loss, logits=logits)
{"loss": loss, "logits": logits}
(loss, logits)
```

When an upstream TTS training objective has a different interface, pass
`compute_loss_func`. It receives raw model outputs, labels, and the number of
items represented by the accumulated batch:

```python
def compute_loss(outputs, labels, num_items_in_batch):
    return acoustic_objective(outputs.audio_values, labels)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=train_dataset,
    compute_loss_func=compute_loss,
)
```

Inference adapters remain lazy and keep their existing `generate()` contract.
An architecture needs a differentiable source `forward` path to be fine-tuned;
inference-only upstream implementations cannot become trainable merely by
wrapping waveform generation in a loss function.

## Extension points

Subclass `Trainer` to override `get_train_dataloader`, `compute_loss`,
`training_step`, `prediction_step`, `create_optimizer`, or
`create_scheduler`. Callbacks observe the loop without replacing it:

```python
from voicehub import EarlyStoppingCallback, TrainerCallback

class StopOnTargetLoss(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics["eval_loss"] < 0.05:
            control.should_training_stop = True
        return control

trainer.add_callback(
    EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
    )
)
```

Step and epoch strategies control logging, evaluation, and checkpoint saves.
Each `checkpoint-<global_step>` directory contains model weights, optimizer
and scheduler state, random-number state, `training_args.json`, and
`trainer_state.json`. Passing `resume_from_checkpoint=True` selects the newest
numeric checkpoint automatically.

The initial implementation deliberately targets one process. Distributed
training, FSDP, DeepSpeed, and TPU execution need backend-specific orchestration
and are not silently simulated.
