---
description: Understand VoiceHub Trainer orchestration and choose the next speech-model training workflow.
---

# Trainer

[`Trainer`](../reference/api.md#trainer) provides a complete training and
evaluation loop for supported VoiceHub speech models. You need a compatible
model, a dataset, and the model-owned objective declared by its training
profile to get started.

Underneath, `Trainer` handles batching, gradient accumulation, evaluation,
checkpointing, exact resume state, and portable model export. Configure the
run with [`TrainingArguments`](../reference/api.md#training-arguments) to choose
batch size, training duration, device and precision strategy, logging,
evaluation, and checkpoint cadence.

Speech architectures do not share one safe fallback loss. VoiceHub therefore
validates the selected model, checkpoint, dataset fields, trainable components,
and recipe before it loads a training runtime. An inference-only or unsupported
path fails closed instead of reporting a plausible but invalid training run.

## Next steps

- Start with the [fine-tuning tutorial](training.md) for the complete data,
  one-step smoke, evaluation, save, and resume workflow.
- Read the [Trainer architecture](../concepts/trainer.md) to understand
  adapters, strategies, callbacks, collation, and exact checkpoint state.
- Check the [training support matrix](../models/training-support.md) before
  selecting a model or checkpoint.
- Use the [data preparation guide](data-preparation.md) to create licensed,
  group-disjoint speech datasets with stable manifests.
