---
description: Inference, data preparation, and training guide for the asr_parakeet_tdt integration.
---

# `asr_parakeet_tdt` model guide

`asr_parakeet_tdt` is a VoiceHub **automatic speech recognition**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `asr_parakeet_tdt` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_parakeet_tdt.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Default checkpoint | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Architecture | `parakeet-tdt` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt.ParakeetTDTForSpeechRecognition` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `timestamps`, `long-form`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| License | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

The pinned Parakeet TDT checkpoint and derivatives require CC-BY-4.0 attribution. The VoiceHub-owned architecture port is audited against Apache-2.0 Transformers and NeMo source. Commercial use: **allowed by the registered terms**.

## Install

```bash
python -m pip install voicehub
```

## Inference

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Place a supported recording at `speech.wav`.
4. Transcribe it and inspect both the full text and timed segments.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe("speech.wav")
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. Pin a checkpoint revision in production.

## Data preparation

The `asr_parakeet_tdt` contract is **integrated-raw**. Its
data architecture is **tdt** and its declared sample rate is
**16,000 Hz**.

Parakeet token-duration transducer audio and transcript records.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `parakeet-tdt-model-ready` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | — | Prepared | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import ASRDataset, get_asr_dataset_spec

contract = get_asr_dataset_spec('asr_parakeet_tdt')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = ASRDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='asr_parakeet_tdt',
        validate_files=True,
    )
    train_records, validation_records = records.train_test_split(
        validation_fraction=0.1,
        seed=42,
        group_by="session_id",
    )
```

See the [complete data guide](../../guides/speech-data.md) for manifest aliases, audio validation,
leakage-safe splits, and model-owned preprocessing.

## Training

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `tdt` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `nvidia/parakeet-tdt-0.6b-v3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.encoder_projector`, `model.decoder`, `model.joint` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | `loss` |

The integration accepts its declared source or prepared contract directly. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForSpeechRecognition, Trainer, TrainingArguments

model = AutoModelForSpeechRecognition.from_pretrained(
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/asr_parakeet_tdt-smoke",
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
trainer.save_model("runs/asr_parakeet_tdt-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
