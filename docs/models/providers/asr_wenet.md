---
description: Public API, checkpoint, training, and optimization guide for the asr_wenet integration.
---

# `asr_wenet` model guide

## Overview

`asr_wenet` is a VoiceHub **automatic speech recognition**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.

## Quickstart

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Place a supported recording at `speech.wav`.
4. Transcribe it and inspect both the full text and timed segments.

Checkpoint note: The registry identifier is not a Hugging Face repository and the published upstream archive is currently unavailable. Replace the path below with a VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    'path/to/converted-wenet-u2pp',
    model_type='asr_wenet',
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
| Architecture | `wenet-asr` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `english`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native`, `ctc`, `attention-rescoring` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid-ctc-attention` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_wenet')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `wenet-waveform-model-ready` | `input_signal`, `input_signal_length`, `labels`, `label_lengths` | — | Prepared | — |
| `wenet-feature-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

WeNet U2++ joint CTC/attention fine-tuning records. Follow the [shared data workflow](../../guides/speech-data.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`wenet/gigaspeech-u2pp-conformer`](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |
| Checkpoint status | Upstream archive unavailable (HTTP 404 verified 2026-08-02); use a previously downloaded, fingerprint-verified local artifact |
| Implementation | `voicehub.models.asr_wenet.WeNetASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wenet.WeNetASRConfig` |
| Source provenance | `voicehub/architectures/wenet_u2pp/SOURCE.json` |
| License | [NOT DECLARED](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |

The published GigaSpeech checkpoint archive does not declare a checkpoint license. The VoiceHub-owned architecture port is Apache-2.0, but that source license is not assumed for the weights. Commercial use: **review required**.

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
| Training checkpoint | `wenet/gigaspeech-u2pp-conformer` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `labels`, `label_lengths` | `loss`, `attention_loss`, `ctc_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wenet')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `WeNetASRConfig` |
| Model implementation | `WeNetASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wenet')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
