---
description: Public API, checkpoint, training, and optimization guide for the asr_moonshine integration.
---

# `asr_moonshine` model guide

## Overview

`asr_moonshine` is a VoiceHub **automatic speech recognition**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `asr_moonshine` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_moonshine.ipynb).

## Quickstart

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Place a supported recording at `speech.wav`.
4. Transcribe it and inspect both the full text and timed segments.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'UsefulSensors/moonshine-tiny',
    model_type='asr_moonshine',
    device="cuda",
    lazy_load=True,
)
output = model.transcribe("speech.wav")
print(output.text)
for segment in output.segments:
    print(segment.start, segment.end, segment.text)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. The example selects a concrete device; verify checkpoint-specific
hardware needs and pin an immutable revision before production use.

## Supported tasks and capabilities

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `moonshine` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `safetensors`, `fine-tuning`, `compact`, `voicehub-native` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `speech-sequence-to-sequence` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_moonshine')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `moonshine-model-ready` | `input_values`, `labels` | — | Prepared | — |

Moonshine waveform-to-sequence fine-tuning records. Follow the [shared data workflow](../../guides/speech-data.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`UsefulSensors/moonshine-tiny`](https://huggingface.co/UsefulSensors/moonshine-tiny) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.asr_moonshine.modeling_asr_moonshine.MoonshineForSpeechRecognition` |
| Configuration | `voicehub.models.asr_moonshine.configuration_asr_moonshine.MoonshineASRConfig` |
| Source provenance | `voicehub/architectures/moonshine/SOURCE.json` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

## Optimization and training support

All public optimizations enter this model through the shared
`BaseSpeechModel` lifecycle. Use `available_optimization_passes()` to discover
the public pass registry, then apply, inspect, serialize, or restore a plan
through the common model API. Application remains fail-closed when the active
runtime or hardware cannot satisfy a pass.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `speech-sequence-to-sequence` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `UsefulSensors/moonshine-tiny` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | — | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_moonshine')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `MoonshineASRConfig` |
| Model implementation | `MoonshineForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_moonshine')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
