---
description: Inference, data preparation, and training guide for the vad_webrtc integration.
---

# `vad_webrtc` model guide

`vad_webrtc` is a VoiceHub **voice activity detection**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.

## Model information

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Default checkpoint | `webrtc-vad` |
| Architecture | `webrtc-vad` |
| Runtime | `VoiceHub-native` |
| Implementation | `voicehub.models.vad_webrtc.modeling_vad_webrtc.WebRTCVADForVoiceActivityDetection` |
| Capabilities | `voice-activity-detection`, `fixed-point`, `voicehub-native`, `native-runtime`, `streaming` |
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
    'webrtc-vad',
    model_type='vad_webrtc',
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
segment-level speech labels. VoiceHub does not expose a verified training dataset contract for this inference-only provider.

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
| Support | `inference-only` |
| Family | `upstream-native` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Training checkpoint | `webrtc-vad` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is intentionally **inference-only**. VoiceHub has no verified
gradient-bearing graph, loss, and reloadable training artifact for it. Do not
attach a generic loss to inference output. Choose a trainable model from the
[training matrix](../training-support.md), or contribute a tested training
adapter and data contract.

## Next steps

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
