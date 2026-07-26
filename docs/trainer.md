# Trainer

VoiceHub includes a PyTorch training API with the same core vocabulary as
Transformers:

```text
TrainingArguments
Trainer
TrainerState / TrainerControl / TrainerCallback
EvalPrediction / TrainOutput / PredictionOutput
DefaultDataCollator
DataCollatorForTTSTraining
TTSTrainingOutput
AutoTrainingAdapter / ModelTrainingSpec
```

PyTorch remains optional. Install the training extra only for training:

```bash
python -m pip install "voicehub[training]"
```

## Basic training

A trainable model returns `TTSTrainingOutput` with `loss` as its first field.
It may be a source-integrated VoiceHub architecture or any `torch.nn.Module`
that follows the same contract. Source-integrated models automatically receive
their registered training adapter.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    "OpenMOSS-Team/MOSS-TTS-v1.5",
    model_type="mosstts",
    device="cuda",
)

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
`labels` is the portable target key. A model profile can also recognize
upstream names such as `targets`, `mel_labels`, and `audio_labels`.
`DataCollatorForTTSTraining` pads variable token, acoustic, and waveform
sequences; integer labels use `-100`, while continuous features use zero.

## All-model adapter layer

Every registered model resolves one of five adapters:

| Family                | Default fallback objective              |
| --------------------- | --------------------------------------- |
| Causal LM             | Shifted codec-token cross entropy       |
| Sequence-to-sequence  | Teacher-forced token cross entropy      |
| Flow matching         | Native flow loss, otherwise MSE         |
| Acoustic regression   | Mel/codec/waveform L1 or MSE             |
| Composite             | Weighted native component losses        |

The adapter searches known model-specific paths first and then performs a
bounded search for the largest trainable source module. This handles runtime
wrappers without binding Trainer to private class names:

```python
adapter = model.get_training_adapter()
print(adapter.spec.family, adapter.spec.source_entrypoints)

trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset)
```

Composite profiles can resolve several modules. Trainer creates separate named
optimizer and scheduler states while deduplicating shared parameters. The
entire bundle is included in checkpoints.

The model matrix in
[`training_models.md`](https://github.com/kadirnar/VoiceHub/blob/main/docs/training_models.md)
distinguishes included native upstream recipes from models trained through the
generic family objective.

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
Training calls the resolved source module instead of waveform generation.
When a published snapshot only exposes an inference graph, pass a specialized
`BaseTrainingAdapter` backed by the corresponding differentiable source model;
VoiceHub raises a precise error instead of treating ONNX output as a gradient.

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
