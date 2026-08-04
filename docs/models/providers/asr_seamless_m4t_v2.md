---
description: Public API, checkpoint, training, and optimization guide for the asr_seamless_m4t_v2 integration.
---

# SeamlessM4Tv2

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
or evaluation. The example selects a concrete device; verify checkpoint-specific
hardware needs and pin an immutable revision before production use.

## Overview

SeamlessM4Tv2 uses the canonical model type `asr_seamless_m4t_v2` and is a
VoiceHub **automatic speech recognition** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `asr_seamless_m4t_v2` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_seamless_m4t_v2.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `seamless-m4t-v2-s2t` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `greedy-decoding`, `full-model-training` |
| Reusable components | — |
| Normalized output | `ASROutput` |

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_seamless_m4t_v2')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_seamless_m4t_v2` |
| Configuration class | `SeamlessM4Tv2ASRConfig` |
| Architecture class | `SeamlessM4Tv2ForSpeechRecognition` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'facebook/seamless-m4t-v2-large',
    model_type='asr_seamless_m4t_v2',
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
| Data architecture | `speech-sequence-to-sequence` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_seamless_m4t_v2')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `seamless-model-ready` | `input_features`, `attention_mask`, `labels` | — | Prepared | — |

SeamlessM4T-v2 multilingual speech-to-text records. Follow the [shared data workflow](../../guides/speech-data.md) for
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
| Training checkpoint | `facebook/seamless-m4t-v2-large` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.speech_encoder`, `model.text_decoder`, `model.shared`, `model.lm_head` | `input_features`, `attention_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/seamless-m4t-v2-large`](https://huggingface.co/facebook/seamless-m4t-v2-large) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_seamless_m4t_v2.modeling_asr_seamless_m4t_v2.SeamlessM4Tv2ForSpeechRecognition` |
| Configuration | `voicehub.models.asr_seamless_m4t_v2.configuration_asr_seamless_m4t_v2.SeamlessM4Tv2ASRConfig` |
| Source provenance | `voicehub/architectures/seamless_m4t_v2/SOURCE.json` |
| License | [CC-BY-NC-4.0](https://huggingface.co/facebook/seamless-m4t-v2-large) |

The pinned SeamlessM4T-v2 Large checkpoint and fine-tuned derivatives are non-commercial under CC-BY-NC-4.0. The VoiceHub-native S2T architecture port is audited against Apache-2.0 Transformers source. Commercial use: **not allowed**.

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

### `SeamlessM4Tv2ASRConfig`

[View `SeamlessM4Tv2ASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/configuration_asr_seamless_m4t_v2.py)

```text
SeamlessM4Tv2ASRConfig(**config_kwargs)
```

### `SeamlessM4Tv2ForSpeechRecognition`

[View `SeamlessM4Tv2ForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_seamless_m4t_v2/modeling_asr_seamless_m4t_v2.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_seamless_m4t_v2',
    config=None,
    **model_kwargs,
)
```

The loader returns `SeamlessM4Tv2ForSpeechRecognition` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_seamless_m4t_v2')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_seamless_m4t_v2')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `SeamlessM4Tv2ASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SeamlessM4Tv2ForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_seamless_m4t_v2')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
