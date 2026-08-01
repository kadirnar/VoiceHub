---
description: Inference, data preparation, and training guide for the inflecttts integration.
---

# `inflecttts` model guide

`inflecttts` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `inflecttts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/inflecttts.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Default checkpoint | [`owensong/Inflect-Micro-v2`](https://huggingface.co/owensong/Inflect-Micro-v2) |
| Architecture | `inflecttts` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.inflecttts.modeling_inflecttts.InflectTTSForTextToSpeech` |
| Capabilities | `text-to-speech`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `vits-warm-start`, `explicit-phonemes` |
| Reusable components | — |
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
    'owensong/Inflect-Micro-v2',
    model_type='inflecttts',
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

The `inflecttts` contract is **preprocessed**. Its
data architecture is **vits** and its declared sample rate is
**24,000 Hz**.

VITS/GAN text, waveform, spectrogram, and adversarial data.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `explicit-features` | `input_ids`, `spectrogram`, `audio_values` | — | Prepared | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec('inflecttts')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = TTSDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='inflecttts',
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
| Support | `preprocessed` |
| Family | `vits` |
| Recipe | `adversarial` |
| Default phase | `generator` |
| Training checkpoint | `owensong/Inflect-Micro-v2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `training_model.generator` | `input_ids`, `input_lengths`, `spectrogram`, `spectrogram_lengths`, `audio_values` | `loss`, `mel_loss`, `kl_loss`, `duration_loss`, `adversarial_loss`, `feature_matching_loss`, `waveform_loss` |
| `discriminator` | discriminator | `training_model.discriminator` | `input_ids`, `input_lengths`, `spectrogram`, `spectrogram_lengths`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    'owensong/Inflect-Micro-v2',
    model_type='inflecttts',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/inflecttts-smoke",
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
trainer.save_model("runs/inflecttts-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
