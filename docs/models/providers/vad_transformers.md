---
description: Inference, data preparation, and training guide for the vad_transformers integration.
---

# `vad_transformers` model guide

`vad_transformers` is a VoiceHub **voice activity detection**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.

## Model information

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Default checkpoint | No default; pass a compatible Hub ID or local directory. |
| Architecture | `wav2vec2` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.vad_transformers.modeling_vad_transformers.TransformersVADForVoiceActivityDetection` |
| Capabilities | `voice-activity-detection`, `frame-scores`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
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
3. Place a supported recording at `speech.wav`.
4. Run detection and tune the threshold against labeled validation audio.

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'owner/model-or-local-directory',
    model_type='vad_transformers',
    device="cpu",
    lazy_load=True,
)
output = model.detect("speech.wav", threshold=0.5)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. Pin a checkpoint revision in production.

## Data preparation

VAD source data should pair authorized audio with clip-, frame-, or
segment-level speech labels. Training phases consume the inputs declared by the selected backend.

Follow this process:

1. Preserve source audio, annotation provenance, consent, and license metadata.
2. Split complete speakers and sessions before windowing the recordings.
3. Convert annotations to the frame or clip boundary required by the phase below.
4. Measure class balance and tune the inference threshold only on validation data.

```python
import json
from pathlib import Path

from voicehub import SpeechDataset

manifest = Path("data/vad-train.jsonl")
source_records = [
    json.loads(line)
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
records = SpeechDataset(
    source_records,
    required_fields=('audio', 'labels'),
)
print(len(records), records.column_names)
```

See the [ASR and VAD data guide](../../guides/speech-data.md) for audio input
forms, timestamp labels, frame targets, and leakage-safe evaluation.

## Training

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `audio-classification` |
| Recipe | `single-phase` |
| Default phase | `voice_activity_detection` |
| Training checkpoint | `owner/model-or-local-directory` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `voice_activity_detection` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Start with one optimizer step and verify finite loss, intended
gradients, frozen components, save, and reload before scaling the run.

```python
from voicehub import AutoModelForVoiceActivityDetection, Trainer, TrainingArguments

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'owner/model-or-local-directory',
    model_type='vad_transformers',
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()
import json
from pathlib import Path

from voicehub import SpeechDataset

manifest = Path("data/vad-train.jsonl")
train_records = [
    json.loads(line)
    for line in manifest.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
train_dataset = SpeechDataset(train_records)

arguments = TrainingArguments(
    output_dir="runs/vad_transformers-smoke",
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
trainer.save_model("runs/vad_transformers-smoke/final")
```

See the [training guide](../../guides/training.md) for validation datasets,
checkpoint resume, mixed precision, optimizations, and portable exports.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
