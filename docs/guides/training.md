---
description: Fine-tune supported TTS and ASR families with native objectives, exact resume, and portable exports.
---

# Training

VoiceHub provides shared orchestration only after a model integration exposes
a valid differentiable graph, native objective, and batch contract. It does
not pretend that every inference backend can be fine-tuned.

Training coverage evolves with each model adapter. Query the registry instead
of relying on a copied count:

```python
from collections import Counter

from voicehub.training import list_training_specs

coverage = Counter(spec.support.value for spec in list_training_specs())
print(coverage)
```

`native` and `preprocessed` are turnkey trainer routes, `custom` records a
specialized upstream or multi-phase boundary, and `inference-only` fails
closed. Read the [model-by-model matrix](../models/training-support.md) before
choosing a checkpoint or dataset format.

This page's counts and examples describe TTS integrations. ASR and VAD use the
same trainer orchestration with additional CTC, speech-seq2seq, RNN-T, TDT,
audio-classification, frame-classification, native-ASR-dispatch, and
upstream-native adapters. ASR fine-tuning records use `ASRDataset`, which
loads mappings or JSON/JSONL/CSV/TSV manifests, imports WAV/transcript
sidecars or portable Kaldi directories, validates each model's raw/prepared
contract, and supplies safe homogeneous batches for Cohere and Seamless.
Inspect a profile before loading weights through
`get_training_spec(model_type).dataset_spec` or
`get_asr_dataset_spec(model_type)`. See
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

The default package already supplies every built-in inference runtime. Add the
single training feature extra for fine-tuning, evaluation, and reporting:

```bash
python -m pip install "voicehub[training]"
```

No model-specific or task-specific inference extra is required.

## Select a differentiable checkpoint

Build a fresh lazy wrapper for training. Do not reuse an object already loaded
through an optimized serving path:

```python
from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "nari-labs/Dia-1.6B-0626",
    model_type="dia",
    backend="native",
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

- `nari-labs/Dia-1.6B-0626` selects VoiceHub's complete native graph;
- every public tensor name and shape is validated before assignment;
- the native DAC is frozen while it constructs codec targets; and
- the original `nari-labs/Dia-1.6B` pickle/JAX layout and non-native backend
  selections are rejected.

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

### Fine-tune native Fish Speech S2

Fish S2-Pro trains the complete semantic model with the source-aligned
base-token and residual-codebook losses. Its ModifiedDAC remains frozen.
Point the wrapper at a previously converted safe codec directory so the final
artifact can include a fresh-inference runtime without reopening the official
pickle:

```python
from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "fishaudio/s2-pro",
    model_type="fishtts",
    codec_name_or_path="/models/fish-s2-codec-safetensors",
    training_max_length=4096,
    device="cuda",
)
train_dataset = training_model.create_training_dataset(
    [
        {
            "tokens": prepared_inputs,  # [11, time], integer IDs
            "labels": prepared_labels,  # [11, time], same-position targets
        }
    ]
)
```

Do not shift the labels again. Channel zero supervises text and the first
semantic code; channels 1 through 10 supervise the fast residual-codebook
decoder only at semantic positions. Raw audio and legacy Fish protobuf paths
are deliberately outside this adapter—encode and validate them offline.
`save_pretrained()` writes semantic weights, tokenizer assets, and the frozen
codec as Safetensors, with the Fish license and required notice. Fine-tuned
artifacts remain non-commercial derivatives unless Fish Audio grants a
separate written license.

### Fine-tune native OuteTTS from V3 profiles

OuteTTS exposes the author-verified completion-only causal-LM objective without
importing Transformers or a provider runtime. Its native adapter trains
`model.language_model` and keeps the 24 kHz DAC frozen. Records contain either
prepared `input_ids` plus `labels`, or a validated V3 profile:

```python
from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "OuteAI/Llama-OuteTTS-1.0-1B",
    model_type="outetts",
    backend="native",
    interface_version="v3",
    device="cuda",
)

train_dataset = training_model.create_training_dataset(
    [
        {
            "speaker_profile": {
                "interface_version": 3,
                "text": "Hello.",
                "words": [
                    {
                        "word": "Hello.",
                        "duration": 0.32,
                        "c1": [101, 231],
                        "c2": [77, 912],
                        "features": {
                            "energy": 28,
                            "spectral_centroid": 42,
                            "pitch": 51,
                        },
                    }
                ],
                "global_features": {
                    "energy": 28,
                    "spectral_centroid": 42,
                    "pitch": 51,
                },
            }
        }
    ],
    completion_only=True,
    max_length=4096,
)
```

The two codebooks must have equal length for every word. Feature values are
integers from 0 through 100, code values are integers from 0 through 1024, and
text must exactly match the aligned profile. Use `prompt_word_count` to mask an
explicit profile prefix, or provide labels with `-100` at every non-trainable
position. The adapter rejects raw audio because the published recipe depends
on author-equivalent word alignment and feature extraction.

`save_pretrained()` writes a fresh-inference bundle containing strict
Safetensors LM and DAC weights, tokenizer files, the default speaker, and an
integrity manifest. Quantized/GGUF and external serving backends are rejected
for training. The default Llama 1B checkpoint is CC-BY-NC-SA-4.0; select the
Apache-2.0 Qwen 0.6B checkpoint when its license and capacity fit the project.

### Fine-tune native F5-TTS

F5-TTS exposes the complete released conditional-flow objective. A batch may
provide `inp` as raw mono waveforms or model-ready mel frames, `text` as
vocabulary IDs padded with `-1`, and optional `lens` in mel frames:

```python
import torch

from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "F5TTS_v1_Base",
    model_type="f5tts",
    device="cuda",
)
training_model.load_for_training()

batch = {
    "inp": waveforms,  # [batch, samples], or [batch, frames, 100]
    "text": token_ids,  # [batch, tokens], -1 is padding
    "lens": mel_lengths,
}
```

VoiceHub samples the masked span, Gaussian endpoint, flow time, audio/text CFG
drops, and velocity target inside the differentiable model. The trainer owns
optimizer-coupled EMA; `save_pretrained()` exports EMA flow weights,
`vocab.txt`, and configuration as a fresh-inference Safetensors directory.
The separately pinned Vocos decoder is frozen. The upstream source is MIT, but
the released `SWivid/F5-TTS` model weights are CC-BY-NC-4.0.

### Fine-tune native VITS with the complete adversarial recipe

VoiceHub owns the differentiable VITS acoustic frontend, generator, and
scale-plus-five-period discriminator. Full fine-tuning accepts aligned raw
waveforms and runs separate discriminator and generator optimizers:

```python
import torch

from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "facebook/mms-tts-eng",
    model_type="vits",
    device="cuda",
    enable_native_adversarial_training=True,
    training_acoustic_config={
        # Copy these values from the checkpoint's source training recipe.
        # VoiceHub deliberately does not infer unpublished settings.
        "sampling_rate": 16_000,
        "filter_length": 1_024,
        "hop_length": 256,
        "win_length": 1_024,
        "num_mel_channels": 80,
        "mel_fmin": 0.0,
        "mel_fmax": 8_000.0,
        "segment_size": 8_192,
    },
)

train_dataset = training_model.create_training_dataset(
    [
        {
            # The tokenizer may also derive IDs from a `text` field.
            "input_ids": torch.tensor([0, 26, 0, 19, 0]),
            # [samples], aligned with the text.
            "audio_values": waveform,
            "sampling_rate": 16_000,
        }
    ]
)
```

The discriminator phase sees detached generated audio. The generator phase
freezes the discriminator and combines mel reconstruction, duration, KL,
feature-matching, and least-squares adversarial losses. Optional `durations`
bypass monotonic alignment search, and multi-speaker checkpoints additionally
require `speaker_id`. Exact trainer resumes retain the discriminator and
optimizer states; `save_pretrained()` exports the generator needed by fresh
inference.

The acoustic configuration is a deliberate provenance boundary. MMS-TTS
checkpoint metadata contains the generator topology but omits the original
FFT, hop, window, mel, and segment settings. VoiceHub validates supplied
values against the checkpoint and fails closed when they are missing or
incompatible.

For controlled compatibility work, set
`enable_native_generator_training=True` instead and provide an exact
checkpoint-compatible linear `spectrogram` with each `audio_values` tensor.
That legacy route trains the posterior, alignment, duration, flow, and decoder
objectives but has no discriminator phase; its artifact metadata records
`full_vits_fine_tuning=False`.

### Fine-tune native MeloTTS from exact linguistic features

MeloTTS exposes the complete published VITS2 recipe, including independent
generator, waveform-discriminator, and duration-discriminator optimizer
phases. Start from a converted native Safetensors directory and explicitly
acknowledge the preprocessed feature boundary:

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "/models/melotts-native",
    model_type="melotts",
    device="cuda",
    enable_native_finetuning=True,
)
dataset = model.create_training_dataset(
    [
        {
            "input_ids": phone_ids,                    # [text]
            "tone_ids": tone_ids,                      # [text]
            "language_ids": language_ids,              # [text]
            "bert_features": bert_features,            # [1024, text]
            "ja_bert_features": ja_bert_features,      # [768, text]
            "spectrogram": magnitude_spectrogram,      # [n_fft // 2 + 1, frames]
            "audio_values": waveform,                  # [samples]
            "speaker_id": speaker_id,
        }
    ]
)
```

The MeloTTS collator right-pads every sequence and supplies exact text,
spectrogram, and waveform lengths. For each item,
`floor(audio_length / hop_length)` must equal the spectrogram length, and
there must be at least one acoustic frame per valid text token. The generator
phase applies the published duration, KL, mel, LSGAN, feature-matching, and
duration-adversarial terms; the other two phases update only their respective
fresh discriminators. The alignment-noise schedule follows the trainer's
global step. Native export saves the seven deployable generator components
for fresh inference; discriminator, optimizer, scheduler, scaler, RNG, and
sampler state belong to an exact VoiceHub training checkpoint instead.

### Fine-tune Kokoro from prepared supervision

Kokoro exposes two VoiceHub-native phases over the exact released inference
graph. It is opt-in because the author repository does not publish the
raw-audio alignment, style-encoder/diffusion, discriminator, and optimizer
recipe:

```python
import torch

from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "hexgrad/Kokoro-82M",
    model_type="kokoro",
    device="cuda",
    enable_preprocessed_training=True,
)

train_dataset = training_model.create_training_dataset(
    [
        {
            # Use caller-prepared author-compatible phonemes, or provide
            # input_ids directly. Training never applies fallback G2P.
            "phonemes": "həlˈoʊ",
            # [text_tokens + 2] integer frame counts, including boundaries.
            "durations": torch.tensor([1, 18, 2, 2, 2, 3, 3, 13]),
            # [256]: decoder style followed by predictor style.
            "ref_s": style_vector,
            # Required by the acoustic phase: [samples] at 24 kHz.
            "audio_values": aligned_waveform,
            # Recommended for padded batches so waveform/spectral losses
            # ignore padding.
            "audio_lengths": torch.tensor(aligned_waveform.shape[-1]),
            # Optional source-compatible auxiliary targets.
            "f0_targets": aligned_f0,
            "energy_targets": aligned_energy,
        }
    ]
)
```

The `duration` phase updates PL-BERT, the projection, and duration predictor.
The `acoustic` phase also runs the convolutional text encoder, F0/energy
predictor, and iSTFTNet waveform decoder. A caller-supplied dense
`alignment` may replace the alignment constructed from integer `durations`.
Every exported runtime uses strict Safetensors. The official `.pth` checkpoint
and `.pt` voice packs are accepted only by the restricted one-time
`torch.load(weights_only=True)` converter; pickle is never part of the
steady-state runtime.

### Fine-tune the OpenVoice V2 converter

OpenVoice's public release contains the complete converter graph but no
training loop, discriminator, dataset, or loss. VoiceHub therefore requires an
explicit opt-in to its reconstructed paired-waveform objective:

```python
from voicehub import AutoModelForTextToSpeech

training_model = AutoModelForTextToSpeech.from_pretrained(
    "/models/openvoice-v2-native",
    model_type="openvoice",
    enable_reconstructed_finetuning=True,
    device="cuda",
)
train_dataset = training_model.create_training_dataset(
    [
        {
            "source_audio": "speaker-a/content-001.wav",
            "target_audio": "speaker-b/content-001.wav",
            "source_reference_audio": "speaker-a/reference.wav",
            "target_reference_audio": "speaker-b/reference.wav",
            "sampling_rate": 22_050,
            "tau": 0.3,
        }
    ]
)
```

`source_audio` and `target_audio` must contain the same linguistic content and
be aligned closely enough for sample-domain reconstruction. The model-owned
collator preserves variable waveform lengths until native resampling and STFT
processing. References may instead be supplied as `[256, 1]` speaker
embeddings. When raw references are used, the reference encoder remains
inside autograd, so the posterior encoder, flow, decoder, and reference
encoder all receive gradients.

The loss is a length-masked smooth-L1 reconstruction over the common generated
and target samples. It is a transparent full-graph adaptation path, but it is
not the unpublished author recipe and does not establish improved speaker
similarity or audio quality. Keep an unmodified validation set and compare
both reconstruction loss and listening/speaker metrics appropriate to your
use case.

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

Dia 1.6B plus gradients and AdamW optimizer state can exceed a free Colab
runtime. Measure one-step memory use before choosing batch size, accumulation,
precision, sharding strategy, and checkpoint cadence.

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

`TTSDataset.resume_fingerprint()` hashes normalized record content and order.
If the dataset uses a lazy `transform`, supply a stable
`transform_fingerprint` when constructing it; exact-resume fingerprinting
rejects an unversioned transform.
Model-owned datasets may expose a narrower fingerprint; keep immutable source
manifests and persist their content hash with every run.

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

## Build specialized TTS objectives explicitly

VoiceHub exposes strict, model-agnostic objective primitives for source
adapters. They import PyTorch lazily and reject ambiguous shapes.

For codec/LLM TTS, use exact multi-codebook alignment rather than flattening
unknown axes:

```python
from voicehub import multi_codebook_cross_entropy

loss = multi_codebook_cross_entropy(
    logits,                 # labels.shape + (vocabulary_size,)
    labels,
    loss_mask=codec_mask,
    causal_shift=True,
    sequence_dim=2,
    codebook_dim=1,
    codebook_weights=weights,
)
```

For diffusion or flow matching, sample randomness during training instead of
storing it in the manifest:

```python
from voicehub import (
    build_flow_matching_training_pair,
    masked_diffusion_regression_loss,
)

pair = build_flow_matching_training_pair(
    clean_latents,
    generator=generator,
    prediction_type="velocity",
)
prediction = model(
    pair.noisy_inputs,
    timesteps=pair.timesteps,
    conditioning=conditioning,
)
loss = masked_diffusion_regression_loss(
    prediction,
    pair.targets,
    mask=latent_mask,
)
```

`build_diffusion_training_pair()` accepts an explicit coefficient function for
the recipe's alpha/sigma schedule and supports epsilon, velocity, or clean
sample prediction. The caller still owns the model's scheduler, conditioning,
codec/latent extractor, SNR weighting policy, and EMA configuration.

For a VITS-family source adapter, the shared primitives cover least-squares
discriminator and generator loss, discriminator feature matching, and masked
diagonal-Gaussian KL:

```python
from voicehub import (
    vits_discriminator_loss,
    vits_feature_matching_loss,
    vits_generator_adversarial_loss,
    vits_kl_loss,
)
```

These functions do not turn an inference synthesizer into a full VITS graph.
The model integration must still expose the posterior encoder, monotonic
alignment/duration path, generator, waveform/duration discriminators, mel and
duration losses, and compatible checkpoint state.
For discriminator updates, detach generated audio before its discriminator
forward; do not detach the resulting fake score, because that would remove the
discriminator's fake-branch gradient.

An adversarial `TrainingPhaseSpec` may set
`optimizer_step_after_phase=True`. With separate named optimizers, the trainer
then steps that phase immediately before running the next phase. This supports
the source-faithful sequence “discriminator forward/backward/step, then
recomputed generator forward/backward/step.” Every scheduled phase in that
scheduled recipe must use the policy consistently. Exact phase boundaries
currently require `gradient_accumulation_steps=1`.

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

For smaller, composable transformations, pass an explicit optimization plan
to `Trainer`:

```python
from voicehub import Trainer
from voicehub.optimization import (
    OPTIMIZATION_PASSES,
    OptimizationContext,
)

OPTIMIZATION_PASSES.register("vendor-training-pass", VendorTrainingPass)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimization_plan=("vendor-training-pass", configured_pass),
    optimization_context=OptimizationContext(
        mode="training",
        device=training_args.device,
        dtype="bfloat16",
        distributed=False,
        persist_result=True,
    ),
)
```

The strategy first places the unwrapped differentiable runtime through
`prepare_device()`. The plan then runs before `prepare_model()` or
`prepare_training_adapter()` may create a strategy proxy and before Trainer
creates optimizers. Named factories stay lazy, and no plan runs unless the
caller supplies `optimization_plan`.

Trainer requires `persist_result=True` and rejects nonpersistent passes before
application. Checkpoints store the exact mode, canonical architecture, and an
immutable snapshot of each pass's ID, compatibility kind, version,
capabilities, configuration, and result metadata. Pass configuration must be
a strict JSON string-key tree and must include effective defaults. An
architecture's compatible pass kinds do not register executable factories.
Resume requires the same explicit plan and configuration and rejects a
mismatch before loading model or optimizer state.

For a recipe with separate optimizers, a topology/name-changing pass must
implement complete post-transform parameter routing for every named optimizer.
Trainer rejects missing, stale, duplicate, or incomplete routes. Exact
checkpoints may retain persistent transformed state. Public and final saves
require such a pass to export canonical state that a fresh unoptimized runtime
can load; otherwise portable save fails closed.

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
