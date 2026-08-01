---
description: Inference, data preparation, and training guide for the xtts integration.
---

# `xtts` model guide

`xtts` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `xtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/xtts.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Default checkpoint | [`coqui/XTTS-v2`](https://huggingface.co/coqui/XTTS-v2) |
| Architecture | `xtts2` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.xtts.modeling_xtts.XTTSForTextToSpeech` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preencoded-code-fine-tuning`, `gpt-fine-tuning`, `restricted-pickle-conversion` |
| Reusable components | — |
| License | [CPML](https://huggingface.co/coqui/XTTS-v2) |

XTTS checkpoint terms are separate from the MPL-2.0 runtime source. Commercial use: **review required**.

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
    'coqui/XTTS-v2',
    model_type='xtts',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
    "language": "en",
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

The `xtts` contract is **preprocessed**. Its
data architecture is **hybrid** and its declared sample rate is
**22,050 Hz**.

Multi-component language-model, diffusion, acoustic, or GAN data.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `native-gpt-tokens` | `text_inputs`, `text_lengths`, `audio_codes`, `wav_lengths` | cond_mels / cond_latents | Prepared | — |
| `native-gpt-waveform` | `text_inputs`, `text_lengths` | wav / audio_values; cond_mels / cond_latents | Prepared | at most one: wav / audio_values; forbidden: audio_codes |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec('xtts')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = TTSDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='xtts',
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
| Recipe | `single-phase` |
| Default phase | `language_model` |
| Training checkpoint | `coqui/XTTS-v2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `language_model` | objective | `model.gpt` | `text_inputs`, `text_lengths`, `audio_codes`, `wav_lengths` | `loss`, `loss_text_ce`, `loss_mel_ce` |

Prepare the exact tensors listed in the data contract before this step. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    'coqui/XTTS-v2',
    model_type='xtts',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/xtts-smoke",
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
trainer.save_model("runs/xtts-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
