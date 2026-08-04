---
description: Public API, checkpoint, training, and optimization guide for the asr_wav2vec2 integration.
---

# Wav2Vec2 {.vh-model-title}

## Usage

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

## Overview

Wav2Vec2 uses the canonical model type `asr_wav2vec2` and is a
VoiceHub **automatic speech recognition** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `asr_wav2vec2` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_wav2vec2.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `wav2vec2` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `automatic-speech-recognition`, `timestamps`, `safetensors`, `fine-tuning`, `voicehub-native` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_wav2vec2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_wav2vec2` |
| Configuration class | `Wav2Vec2ASRConfig` |
| Architecture class | `Wav2Vec2ForSpeechRecognition` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'facebook/wav2vec2-base-960h',
    model_type='asr_wav2vec2',
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

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_wav2vec2.modeling_asr_wav2vec2.Wav2Vec2ForSpeechRecognition` |
| Configuration | `voicehub.models.asr_wav2vec2.configuration_asr_wav2vec2.Wav2Vec2ASRConfig` |
| Source provenance | No integration-specific bundled `SOURCE.json` is declared for this registry entry. |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- The Usage example selects `cuda`; validate memory, precision,
  and optional dependency requirements on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not substitute for released-checkpoint evidence. Consult the
  linked release record before treating a checkpoint path as verified.

## Public API

The stable configuration and model facades keep source inspection local while
the task auto class owns pretrained loading and normalized output behavior.

### `Wav2Vec2ASRConfig`

[View `Wav2Vec2ASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wav2vec2/configuration_asr_wav2vec2.py)

```text
Wav2Vec2ASRConfig(**config_kwargs)
```

### `Wav2Vec2ForSpeechRecognition`

[View `Wav2Vec2ForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_wav2vec2/modeling_asr_wav2vec2.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_wav2vec2',
    config=None,
    **model_kwargs,
)
```

The loader returns `Wav2Vec2ForSpeechRecognition` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_wav2vec2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_wav2vec2')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `Wav2Vec2ASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `Wav2Vec2ForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_wav2vec2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
