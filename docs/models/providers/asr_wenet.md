---
description: Public API, checkpoint, training, and optimization guide for the asr_wenet integration.
---

# WeNetASR

## Usage

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Place a supported recording at `speech.wav`.
4. Transcribe it and inspect both the full text and timed segments.

Checkpoint note: The registry identifier is not a Hugging Face repository and the original upstream archive endpoints are unavailable. VoiceHub verifies an immutable mirror against the published 503,845,602-byte archive's SHA-256. Convert that trust-gated pickle archive first, then replace the path below with the resulting VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.

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

## Overview

WeNetASR uses the canonical model type `asr_wenet` and is a
VoiceHub **automatic speech recognition** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code.

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `wenet-asr` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `english`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native`, `ctc`, `attention-rescoring` |
| Reusable components | — |
| Normalized output | `ASROutput` |

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_wenet')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_wenet` |
| Configuration class | `WeNetASRConfig` |
| Architecture class | `WeNetASRForSpeechRecognition` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'path/to/converted-wenet-u2pp',
    model_type='asr_wenet',
)
print(type(processor).__name__)
```

Processor behavior remains model-owned when text normalization, audio loading,
feature extraction, or reference speech requires provider-specific semantics.

## Inference

The Usage example returns `ASROutput` through `AutoModelForSpeechRecognition`. Inputs are validated
against the task and data contracts below before model-specific execution.

### Input and output contract

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

## Training and optimization

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

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`wenet/gigaspeech-u2pp-conformer`](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |
| Checkpoint status | Original upstream archive unavailable (HTTP 404 and TLS failures verified 2026-08-04); exact bytes are available from the immutable openspeech/wenet-models mirror at 90acd57d17169a15d5ceab462c6e7db3bd003921 |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_wenet.WeNetASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wenet.WeNetASRConfig` |
| Source provenance | `voicehub/architectures/wenet_u2pp/SOURCE.json` |
| License | [NOT DECLARED](https://github.com/wenet-e2e/wenet/blob/a50d4208f13bbf3a0746e606ac29176cd2e87e6b/examples/gigaspeech/s0/README.md#conformer-u2-result) |

The published GigaSpeech checkpoint archive does not declare a checkpoint license. The VoiceHub-owned architecture port is Apache-2.0, but that source license is not assumed for the weights. Commercial use: **review required**.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

### Limitations

- The registry identifier is not a Hugging Face repository and the original upstream archive endpoints are unavailable. VoiceHub verifies an immutable mirror against the published 503,845,602-byte archive's SHA-256. Convert that trust-gated pickle archive first, then replace the path below with the resulting VoiceHub-native directory containing model.safetensors, config.json, tokenizer.model, and units.txt.
- The Usage example selects `cuda`; validate memory, precision,
  and optional dependency requirements on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not substitute for released-checkpoint evidence. Consult the
  linked release record before treating a checkpoint path as verified.

## Public API

The stable configuration and model facades keep source inspection local while
the task auto class owns pretrained loading and normalized output behavior.

### `WeNetASRConfig`

[View `WeNetASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py)

```text
WeNetASRConfig(**config_kwargs)
```

### `WeNetASRForSpeechRecognition`

[View `WeNetASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wenet/__init__.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_wenet',
    config=None,
    **model_kwargs,
)
```

The loader returns `WeNetASRForSpeechRecognition` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_wenet')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wenet')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `WeNetASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WeNetASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wenet')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
