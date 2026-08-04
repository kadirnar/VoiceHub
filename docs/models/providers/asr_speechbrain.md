---
description: Public API, checkpoint, training, and optimization guide for the asr_speechbrain integration.
---

# SpeechBrainASR {.vh-model-title}

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
    'speechbrain/asr-crdnn-rnnlm-librispeech',
    model_type='asr_speechbrain',
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

SpeechBrainASR uses the canonical model type `asr_speechbrain` and is a
VoiceHub **automatic speech recognition** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `asr_speechbrain` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_speechbrain.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `speechbrain-crdnn-asr` |
| Runtime | `VoiceHub-native` |
| Languages | `en` |
| Capabilities | `automatic-speech-recognition`, `english`, `beam-search`, `safetensors`, `fine-tuning`, `voicehub-native`, `crdnn`, `ctc-seq2seq`, `rnnlm-shallow-fusion` |
| Reusable components | — |
| Normalized output | `ASROutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>1 documented language</summary>

`en`

</details>

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_speechbrain')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_speechbrain` |
| Configuration class | `SpeechBrainASRConfig` |
| Architecture class | `SpeechBrainASRForSpeechRecognition` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'speechbrain/asr-crdnn-rnnlm-librispeech',
    model_type='asr_speechbrain',
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
| Contract getter | `get_asr_dataset_spec('asr_speechbrain')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | — | audio / audio_path; text / transcription / transcript | Source | at most one: audio / audio_path; text / transcription / transcript |
| `speechbrain-model-ready` | `waveforms`, `waveform_lengths`, `tokens_bos`, `tokens_eos`, `token_lengths`, `ctc_tokens`, `ctc_token_lengths` | — | Prepared | — |

SpeechBrain CRDNN joint CTC/attention fine-tuning records. Follow the [shared data workflow](../../guides/speech-data.md) for
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
| Training checkpoint | `speechbrain/asr-crdnn-rnnlm-librispeech` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `waveforms`, `waveform_lengths`, `tokens_bos`, `tokens_eos`, `token_lengths`, `ctc_tokens`, `ctc_token_lengths` | `loss`, `seq2seq_loss`, `ctc_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`speechbrain/asr-crdnn-rnnlm-librispeech`](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.speechbrain.SpeechBrainASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.SpeechBrainASRConfig` |
| Source provenance | `voicehub/architectures/speechbrain_asr/SOURCE.json` |
| License | [Apache-2.0](https://huggingface.co/speechbrain/asr-crdnn-rnnlm-librispeech) |

The pinned CRDNN, RNNLM, tokenizer, and source implementation are Apache-2.0. The original pickle files cross a strict one-time conversion boundary; steady-state artifacts are Safetensors. Commercial use: **allowed by the registered terms**.

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

### `SpeechBrainASRConfig`

[View `SpeechBrainASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py)

```text
SpeechBrainASRConfig(**config_kwargs)
```

### `SpeechBrainASRForSpeechRecognition`

[View `SpeechBrainASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/speechbrain.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_speechbrain',
    config=None,
    **model_kwargs,
)
```

The loader returns `SpeechBrainASRForSpeechRecognition` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_speechbrain')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_speechbrain')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `SpeechBrainASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `SpeechBrainASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_speechbrain')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
