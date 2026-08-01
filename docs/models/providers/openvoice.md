---
description: Inference, data preparation, and training guide for the openvoice integration.
---

# `openvoice` model guide

`openvoice` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `openvoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/openvoice.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Default checkpoint | [`myshell-ai/OpenVoiceV2`](https://huggingface.co/myshell-ai/OpenVoiceV2) |
| Architecture | `openvoice-v2-converter` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.openvoice.modeling_openvoice.OpenVoiceForTextToSpeech` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `paired-waveform-training`, `explicit-base-waveform` |
| Reusable components | `wavmark` |
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
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
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

The `openvoice` contract is **integrated-raw**. Its
data architecture is **vits** and its declared sample rate is
**22,050 Hz**.

VITS/GAN text, waveform, spectrogram, and adversarial data.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `paired-waveforms` | `source_audio`, `target_audio` | — | Source | — |
| `paired-waveform-aliases` | `audio`, `target_waveform` | — | Source | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import TTSDataset, get_tts_dataset_spec

contract = get_tts_dataset_spec('openvoice')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = TTSDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='openvoice',
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
| Family | `vits` |
| Recipe | `single-phase` |
| Default phase | `generator` |
| Training checkpoint | `myshell-ai/OpenVoiceV2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `generator` | generator | `model.enc_q`, `model.flow`, `model.dec`, `model.ref_enc` | `source_spectrogram`, `source_lengths`, `target_waveform`, `target_lengths` | `loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    'myshell-ai/OpenVoiceV2',
    model_type='openvoice',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/openvoice-smoke",
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
trainer.save_model("runs/openvoice-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
