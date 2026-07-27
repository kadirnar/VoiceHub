---
description: Fine-tune supported TTS families with native objectives, exact resume, and portable exports.
---

# Training

VoiceHub provides shared orchestration only after a model integration exposes
a valid differentiable graph, native objective, and batch contract. It does
not pretend that every inference backend can be fine-tuned.

At the current 31-model registry revision:

- 18 integrations have some fine-tuning path;
- 6 accept ordinary raw-data records; and
- 13 have no verified VoiceHub training path.

Read the [model-by-model matrix](../models/training-support.md) before choosing
a checkpoint or dataset format.

This page's counts and examples describe TTS integrations. ASR and VAD use the
same trainer orchestration with additional CTC, speech-seq2seq, RNN-T, TDT,
audio-classification, frame-classification, and upstream-native adapters. See
the [ASR/VAD matrix](../models/asr-vad-support.md) and
[speech data guide](speech-data.md).

## Understand the support levels

| Level            | Guarantee                                                                                                                |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `native`         | The integrated runtime exposes a differentiable backend-native loss that VoiceHub can execute.                           |
| `preprocessed`   | The objective is integrated, but callers must provide backend-shaped tensors. Raw source-record preparation is not included. |
| `custom`         | A model-specific adapter and orchestration path are required; a family label alone is insufficient.                      |
| `inference-only` | The current runtime has no verified gradient path, often because it is fused, compiled, quantized, or inference-pruned.  |

Support is checkpoint-aware. A model family may be trainable upstream while
the selected VoiceHub backend or artifact remains inference-only.

## Install the training runtime

Install the model extra and training extra together:

```bash
python -m pip install "voicehub[dia,training]"
```

The base installation does not pull PyTorch or every model stack.

## Select a differentiable checkpoint

Build a fresh lazy wrapper for training. Do not reuse an object already loaded
through an optimized serving path:

```python
from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    backend="transformers",
    compute_dtype="bfloat16",
    device="cuda",
    lazy_load=True,
)

training_spec = training_model.validate_training_support()
print(training_spec.support.value)
print(training_spec.family_name)
print([phase.name for phase in training_spec.phases])
```

Validation runs before weights are allocated. For Dia:

- `nari-labs/Dia-1.6B-0626` selects the trainable Transformers graph;
- the original `nari-labs/Dia-1.6B` Nari runtime is inference-only; and
- `backend="legacy"` is rejected for fine-tuning.

GGUF, ONNX, TensorRT, JIT, vLLM, fused, quantized, or inference-pruned
artifacts are not automatically trainable. Safetensors are suitable only when
they reconstruct the differentiable graph required by the adapter.

## Build the model-specific dataset

For an integrated raw-data recipe:

```python
train_dataset = training_model.create_training_dataset(train_records)
validation_dataset = training_model.create_training_dataset(
    validation_records
)
```

VoiceHub automatically uses a callable `train_dataset.collate_fn` unless an
explicit `data_collator` is supplied.

Models on a preprocessed route expect tensors, masks, codes, or source records
described in the [data preparation guide](data-preparation.md) and training
matrix. The generic trainer does not silently synthesize:

- codec delays or codebook offsets;
- flow noise, sampled time, or velocity targets;
- VITS alignments, spectrograms, or adversarial pairs; or
- phase-specific detach boundaries.

## Configure a one-step smoke run

```python
from voicehub import Trainer, TrainingArguments

arguments = TrainingArguments(
    output_dir="runs/dia-finetune",
    max_steps=1,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    bf16=True,
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=1,
    save_strategy="steps",
    save_steps=1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    dataloader_num_workers=0,
    seed=42,
    data_seed=42,
)

trainer = Trainer(
    model=training_model,
    args=arguments,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
```

## Track the run with Weights & Biases

The `training` extra includes the W&B SDK. Enable it without constructing a
callback manually:

```python
arguments = TrainingArguments(
    output_dir="runs/dia-finetune",
    report_to="wandb",
    run_name="dia-speaker-adaptation",
    wandb_project="voicehub-finetuning",
    wandb_entity="your-team",
    wandb_group="dia-ablation",
    wandb_tags=["tts", "dia", "speaker-adaptation"],
    wandb_notes="Baseline learning-rate and frozen-codec run.",
    wandb_mode="online",
    wandb_log_model="checkpoint",
)
```

`logging_steps` controls metric cadence. Training, evaluation, and test
metrics are grouped under `train/`, `eval/`, and `test/`. The integration:

- imports W&B only when a reported run begins;
- logs only from `TrainerState.is_world_process_zero`;
- stores the W&B run ID in exact-resume checkpoint state;
- reuses a run created explicitly with `wandb.init()` without finishing it;
- supports `wandb_mode="offline"` for later synchronization; and
- uploads only complete VoiceHub checkpoints when
  `wandb_log_model="checkpoint"`, or one portable final model when set to
  `"end"`.

Authentication stays outside serializable training arguments. Use
`wandb login` or the `WANDB_API_KEY` environment variable rather than putting
an API key in source code or a checkpoint.

!!! note "Choose precision from hardware"

    `bf16=True` is appropriate only on BF16-capable hardware. Use `fp16=True`
    where supported, or disable both flags for float32. Keep the model
    `compute_dtype` and trainer precision consistent.

Before increasing `max_steps`, verify:

1. the dataset and collator produce a complete batch;
2. the native loss is finite and scalar;
3. the loss requires gradients;
4. intended parameters receive gradients;
5. frozen codecs, vocoders, and encoders do not receive gradients; and
6. the run writes a reloadable artifact.

Dia 1.6B plus AdamW optimizer state can exceed a free Colab runtime. Measure
one-step memory use before choosing batch size, accumulation, precision, and
checkpoint cadence.

## Start a new run

Use a new or checkpoint-free `output_dir`:

```python
train_output = trainer.train()
print(train_output.global_step)
print(train_output.training_loss)
```

If the output directory already contains a numeric checkpoint, VoiceHub
requires an explicit resume or `overwrite_output_dir=True`.

## Evaluate

```python
metrics = trainer.evaluate()
print(metrics)
```

Evaluation uses the same adapter, recipe, precision strategy, and model-owned
loss contract. For generative listening tests, save a separate fixed prompt
set and use identical decoding settings before and after training.

## Resume exactly

Resume the newest complete checkpoint:

```python
train_output = trainer.train(resume_from_checkpoint=True)
```

Or choose one explicitly:

```python
train_output = trainer.train(
    resume_from_checkpoint="runs/dia-finetune/checkpoint-1000"
)
```

Exact resume validates:

- model type, adapter, and recipe identity;
- optimizer names and strategy;
- precision and scaler mode;
- batch size, accumulation, and schedule;
- dataset class and length;
- declared dataset and collator fingerprints;
- callback, sampler, and random state; and
- multi-phase optimizer topology.

Dia's dataset fingerprint describes its collator but does not hash record
contents. Keep immutable manifests and record their content hash with every
run.

Generic exact mid-epoch resume requires:

- a stable, sized dataset or dataloader; and
- `dataloader_num_workers=0`.

Worker prefetch queues and arbitrary iterable cursors cannot be reconstructed
portably.

## Save the right artifact

Save the final portable VoiceHub artifact:

```python
artifact_directory = trainer.save_model(
    "runs/dia-finetune/final"
)
```

Typical layout:

```text
runs/dia-finetune/final/
  config.json
  generation_config.json
  processor_config.json
  model_state.pt
  training_args.json
  training_recipe.json
  native_export/
```

Periodic `checkpoint-N/` directories additionally contain optimizer,
scheduler, optional precision-scaler, trainer, RNG, sampler, callback, and
strategy state.

| Artifact                  | Use                                                                 |
| ------------------------- | ------------------------------------------------------------------- |
| `checkpoint-N/`           | Exact continuation of the same training plan                        |
| Portable VoiceHub folder  | Reload, inference, or weight-only warm start                         |
| `native_export/`          | Source-native use when its recipe declares a complete export        |
| Standalone safetensors    | Weight container, never complete optimizer-bearing resume state      |

Do not serve `optimizer.pt`, rename one safetensors file as a full checkpoint,
or pass an arbitrary periodic checkpoint to an upstream inference loader.

## Compare post-training inference

Reuse the same prompt and decoding configuration used for the baseline:

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

BASELINE_TEXT = (
    "[S1] VoiceHub keeps inference, data preparation, and training "
    "on one explicit lifecycle."
)

fine_tuned_model = AutoModelForTextToSpeech.from_pretrained(
    artifact_directory,
    device="cuda",
    lazy_load=True,
)

fine_tuned_output = fine_tuned_model.generate(
    BASELINE_TEXT,
    generation_config=TTSGenerationConfig(
        seed=42,
        temperature=1.0,
        max_new_tokens=2048,
        output_file="artifacts/dia-finetuned.wav",
    ),
)
```

Human evaluation should be blinded and cover:

- intelligibility;
- speaker similarity where applicable;
- prosody and style;
- clicks, noise, collapse, or other artifacts;
- memorization and leakage; and
- safety and authorization requirements.

Automated metrics complement listening tests; they do not replace them.

## Adapt the run to another family

| Family                  | Training boundary                                                                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Causal/codec LM         | Completion masks, codebook order, frozen codecs, and token-normalized cross-entropy must match the source implementation.                 |
| Encoder-decoder LM      | Processor-owned decoder shifts, delay patterns, attention masks, and teacher-forced labels remain authoritative.                         |
| Flow matching/diffusion | Noise schedule, sampled time, conditioning dropout, target parameterization, EMA, and solver/export state are recipe-owned.              |
| VITS/GAN                | Generator and discriminator losses, feature matching, KL/duration/mel terms, detach boundaries, and update cadence remain distinct.      |
| Hybrid/composite        | Each named component owns its optimizer, loss, frequency, checkpoint state, and native export semantics.                                 |

VoiceHub's `ModelTrainingSpec` declares phase ownership, optimizer routes,
frequency, detach boundaries, and native losses. Do not pass one generic
optimizer over every reachable parameter.

For exact qualifications—such as CosyVoice component training, XTTS GPT-only
fine-tuning, or experimental Higgs Audio reconstruction—read the
[training support matrix](../models/training-support.md).

## Training strategies

The built-in strategy is single-process PyTorch. Distributed, FSDP,
DeepSpeed, Accelerate, TPU, or quantization-aware execution requires a
registered `TrainingStrategy`. The model recipe and dataset contract remain
separate from execution:

```text
model integration  -> differentiable components
training recipe    -> phases, losses, ownership
Trainer            -> dataloaders, scheduling, checkpoints
strategy           -> device, precision, backward, optimizer execution
```

This boundary lets future execution engines optimize training without
rewriting every model adapter. See the [trainer architecture](../concepts/trainer.md).

## Troubleshooting

### Training validation fails

Read the complete error. It normally identifies an incompatible checkpoint,
legacy backend, GGUF/ONNX/compiled runtime, quantization option, or missing
specialized adapter.

### Loss is missing, detached, or non-finite

Stop the run. Confirm that the training runtime exposes the native objective,
labels reach it, and intended parameters require gradients. An inference
waveform is not a training loss.

### CUDA out of memory

Reduce per-device batch size, increase gradient accumulation, shorten or
bucket sequences, use supported mixed precision, and ensure frozen codecs are
excluded from optimizer state. Do not silently enable quantization unless the
adapter explicitly supports quantized training.

### Exact resume signature mismatch

Restore the original batch, schedule, precision, strategy,
dataset/collator, callback, and optimizer topology. If the plan intentionally
changed, start a new run from a portable artifact as a weight warm start.

## Reproducibility checklist

Record:

- VoiceHub version and Git revision;
- checkpoint ID and revision;
- model license and dataset permissions;
- immutable train/evaluation manifests and hashes;
- preprocessing code and version;
- tokenizer, processor, and codec revisions;
- complete `TrainingArguments`;
- model-specific recipe configuration;
- random and data seeds;
- training strategy and world size;
- checkpoint manifest; and
- the inference configuration used for comparison.

The [end-to-end notebook](notebook.md) implements this lifecycle with Dia.
