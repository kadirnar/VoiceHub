---
description: Public API, checkpoint, training, and optimization guide for the asr_wav2vec2 integration.
---

# `asr_wav2vec2` model guide

## Overview

`asr_wav2vec2` is a VoiceHub **automatic speech recognition**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `asr_wav2vec2` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_wav2vec2.ipynb).

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
    'facebook/wav2vec2-base-960h',
    model_type='asr_wav2vec2',
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
| Architecture | `wav2vec2` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `ctc` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_wav2vec2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `waveform-ctc` | `input_values`, `labels` | — | Prepared | — |

Wav2Vec2 waveform and CTC transcript records. Follow the [shared data workflow](../../guides/speech-data.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2.Wav2Vec2ForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2.Wav2Vec2ASRConfig` |
| Source provenance | No integration-specific bundled `SOURCE.json` is declared for this registry entry. |
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
| Family | `ctc` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `facebook/wav2vec2-base-960h` |
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
| Discover | `get_model_spec('asr_wav2vec2')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `Wav2Vec2ASRConfig` |
| Model implementation | `Wav2Vec2ForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wav2vec2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
