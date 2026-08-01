---
description: Inference, data preparation, and training guide for the conversationtts integration.
---

# `conversationtts` model guide

`conversationtts` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `conversationtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/conversationtts.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Default checkpoint | [`AudioFoundation/SpeechFoundation`](https://huggingface.co/AudioFoundation/SpeechFoundation) |
| Architecture | `conversationtts` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.conversationtts.modeling_conversationtts.ConversationTTSForTextToSpeech` |
| Capabilities | `text-to-speech`, `voice-cloning`, `conversation`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning`, `noncommercial` |
| Reusable components | — |
| License | [CC-BY-NC-4.0](https://github.com/Audio-Foundation-Models/ConversationTTS) |

Source, checkpoints, datasets, and evaluation tools are non-commercial. Commercial use: **not allowed**.

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
    'AudioFoundation/SpeechFoundation',
    model_type='conversationtts',
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

The `conversationtts` contract is **integrated-raw**. Its
data architecture is **codec-lm** and its declared sample rate is
**24,000 Hz**.

Autoregressive text/audio-token or codec-language-model data.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-text-audio` | — | text / texts; audio / audio_values | Source | at most one: text / texts; audio / audio_values; forbidden: text_token_ids, text_ids, audio_codes, codes |
| `raw-text-code` | — | text / texts; audio_codes / codes | Prepared | at most one: text / texts; audio_codes / codes; forbidden: text_token_ids, text_ids, audio, audio_values |
| `tokenized-text-audio` | — | text_token_ids / text_ids; audio / audio_values | Prepared | at most one: text_token_ids / text_ids; audio / audio_values; forbidden: text, texts, audio_codes, codes |
| `tokenized-text-code` | — | text_token_ids / text_ids; audio_codes / codes | Prepared | at most one: text_token_ids / text_ids; audio_codes / codes; forbidden: text, texts, audio, audio_values |
| `multi-codebook-batch` | `tokens`, `labels`, `tokens_mask` | — | Prepared | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec('conversationtts')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = TTSDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='conversationtts',
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
| Support | `native` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `codec_language_model` |
| Training checkpoint | `AudioFoundation/SpeechFoundation` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `tokens`, `labels`, `tokens_mask` | `loss`, `codebook0_loss`, `residual_loss` |

The integration accepts its declared source or prepared contract directly. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    'AudioFoundation/SpeechFoundation',
    model_type='conversationtts',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/conversationtts-smoke",
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
trainer.save_model("runs/conversationtts-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
