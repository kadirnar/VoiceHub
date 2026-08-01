---
description: Inference, data preparation, and training guide for the asr_seamless_m4t_v2 integration.
---

# `asr_seamless_m4t_v2` model guide

`asr_seamless_m4t_v2` is a VoiceHub **automatic speech recognition**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `asr_seamless_m4t_v2` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_seamless_m4t_v2.ipynb).

## Model information

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Default checkpoint | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) |
| Architecture | `seamless-m4t-v2-s2t` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2.SeamlessM4Tv2ForSpeechRecognition` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `greedy-decoding`, `full-model-training` |
| Reusable components | — |
| License | [CC-BY-NC-4.0](https://huggingface.co/facebook/seamless-m4t-v2-large) |

The pinned SeamlessM4T-v2 Large checkpoint and fine-tuned derivatives are non-commercial under CC-BY-NC-4.0. The VoiceHub-native S2T architecture port is audited against Apache-2.0 Transformers source. Commercial use: **not allowed**.

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
    'facebook/seamless-m4t-v2-large',
    model_type='asr_seamless_m4t_v2',
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

The `asr_seamless_m4t_v2` contract is **integrated-raw**. Its
data architecture is **speech-sequence-to-sequence** and its declared sample rate is
**16,000 Hz**.

SeamlessM4T-v2 multilingual speech-to-text records.

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `seamless-model-ready` | `input_features`, `attention_mask`, `labels` | — | Prepared | — |

Follow this process:

1. Keep immutable source audio, exact transcripts or labels, stable IDs, consent,
   license, speaker, and session metadata.
2. Split by speaker or recording session before model preprocessing.
3. Match one of the exact variants above. Source variants are processed by the
   integration; prepared variants must already contain the listed model inputs.
4. Validate one collated batch, then persist the preprocessing version and hashes.

```python
from voicehub import ASRDataset, get_asr_dataset_spec

contract = get_asr_dataset_spec('asr_seamless_m4t_v2')
print(contract.architecture, contract.readiness, contract.sample_rate)
for variant in contract.variants:
    print(variant.name, variant.required_fields, variant.one_of)

# Source-record integrations can validate a JSONL manifest directly.
if contract.accepts_raw_records:
    records = ASRDataset.from_manifest(
        "data/manifest.jsonl",
        model_type='asr_seamless_m4t_v2',
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
| Family | `speech-sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `facebook/seamless-m4t-v2-large` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.speech_encoder`, `model.text_decoder`, `model.shared`, `model.lm_head` | `input_features`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForSpeechRecognition, Trainer, TrainingArguments

model = AutoModelForSpeechRecognition.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    model_type='asr_seamless_m4t_v2',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
train_dataset = model.create_training_dataset(
    "data/train.jsonl",
    validate_audio_files=True,
)

arguments = TrainingArguments(
    output_dir="runs/asr_seamless_m4t_v2-smoke",
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
trainer.save_model("runs/asr_seamless_m4t_v2-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
