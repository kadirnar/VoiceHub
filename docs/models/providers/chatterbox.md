---
description: Inference, data preparation, and training guide for the chatterbox integration.
---

# `chatterbox` model guide

`chatterbox` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `chatterbox` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/chatterbox.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Default checkpoint | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) |
| Architecture | `chatterbox` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.chatterbox.modeling_chatterbox.ChatterboxForTextToSpeech` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning` |
| Reusable components | `conformer` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

## Install

```bash
python -m pip install voicehub
```

## Inference

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Set the input text and generation options for your use case.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'ResembleAI/chatterbox',
    model_type='chatterbox',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. Pin a checkpoint revision in production.

## Data preparation

The `chatterbox` contract is **integrated-raw**. Its
data architecture is **hybrid** and its declared sample rate is
**24,000 Hz**.

Multi-component language-model, diffusion, acoustic, or GAN data.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `t3-raw` | `text` | audio / audio_path | Source | — |
| `flow-raw` | — | audio / audio_path | Source | — |
| `t3-precomputed` | `text_tokens`, `speech_tokens`, `speaker_emb` | — | Prepared | — |
| `flow-precomputed` | `speech_token`, `speech_feat`, `embedding` | — | Prepared | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec('chatterbox')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = TTSDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='chatterbox',
        validate_files=True,
    )
    train_records, validation_records = records.train_test_split(
        validation_fraction=0.1,
        seed=42,
        group_by="session_id",
    )
```

See the [complete data guide](../../guides/data-preparation.md) for manifest aliases, audio validation,
leakage-safe splits, and model-owned preprocessing.

## Training

| Property | Value |
| --- | --- |
| Support | `custom` |
| Family | `composite` |
| Recipe | `multi-phase` |
| Default phase | `language_model` |
| Training checkpoint | `ResembleAI/chatterbox` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `language_model` | objective | `model.t3` | — | `loss`, `text_loss`, `speech_token_loss` |
| `flow` | objective | `model.s3gen.flow` | — | `loss`, `flow_loss`, `diffusion_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    'ResembleAI/chatterbox',
    model_type='chatterbox',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/chatterbox-smoke",
    max_steps=1,
    per_device_train_batch_size=1,
    learning_rate=5e-5,
    logging_steps=1,
    save_steps=1,
    report_to="none",
    seed=42,
)
trainer = Trainer(model=model, args=arguments, train_dataset=train_dataset)
result = trainer.train(resume_from_checkpoint=False)
print(result.training_loss, result.metrics)
trainer.save_model("runs/chatterbox-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
