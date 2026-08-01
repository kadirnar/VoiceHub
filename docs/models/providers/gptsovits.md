---
description: Inference, data preparation, and training guide for the gptsovits integration.
---

# `gptsovits` model guide

`gptsovits` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `gptsovits` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/gptsovits.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Default checkpoint | [`lj1995/GPT-SoVITS`](https://huggingface.co/lj1995/GPT-SoVITS) |
| Architecture | `gptsovits` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.gptsovits.modeling_gptsovits.GPTSoVITSForTextToSpeech` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `gpt-sovits-v1`, `gpt-sovits-v2`, `gpt-sovits-v2-pro`, `gpt-sovits-v2-pro-plus`, `prepared-pro-speaker-conditioning`, `variant-aware-safetensors-export` |
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
3. Provide an authorized `reference.wav` and an exact reference transcript when the example requests them.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
    "prompt_text": REFERENCE_TEXT,
    "text_language": "en",
    "prompt_language": "en",
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

The `gptsovits` contract is **preprocessed**. Its
data architecture is **hybrid** and its declared sample rate is
**32,000 Hz**.

Multi-component language-model, diffusion, acoustic, or GAN data.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `s1-preprocessed` | `phoneme_ids`, `semantic_ids`, `bert_features` | — | Prepared | — |
| `s2-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids` | — | Prepared | — |
| `s2-pro-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids`, `speaker_embedding` | — | Prepared | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec('gptsovits')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = TTSDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='gptsovits',
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
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `s1` |
| Training checkpoint | `lj1995/GPT-SoVITS` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `s1` | objective | `training_model.s1` | `phoneme_ids`, `phoneme_lengths`, `semantic_ids`, `semantic_lengths`, `bert_features` | `loss` |
| `s2_generator` | generator | `training_model.s2.generator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |
| `s2_discriminator` | discriminator | `training_model.s2.discriminator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/gptsovits-smoke",
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
trainer.save_model("runs/gptsovits-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
