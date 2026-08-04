---
description: Public API, checkpoint, training, and optimization guide for the asr_espnet integration.
---

# ESPnetASR

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
    'espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best',
    model_type='asr_espnet',
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

ESPnetASR uses the canonical model type `asr_espnet` and is a
VoiceHub **automatic speech recognition** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `asr_espnet` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_espnet.ipynb).

| Property | Value |
| --- | --- |
| Task | Automatic speech recognition |
| Architecture | `espnet-librispeech-transformer-e18` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `english`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `hybrid-ctc-attention` |
| Reusable components | — |
| Normalized output | `ASROutput` |

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('asr_espnet')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `asr_espnet` |
| Configuration class | `ESPnetASRConfig` |
| Architecture class | `ESPnetASRForSpeechRecognition` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best',
    model_type='asr_espnet',
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
| Contract getter | `get_asr_dataset_spec('asr_espnet')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | — | audio / audio_path; text / transcription / transcript | Source | at most one: audio / audio_path; text / transcription / transcript |
| `espnet-feature-transcript` | `features` | text / transcription / transcript | Prepared | at most one: text / transcription / transcript |
| `espnet-waveform-model-ready` | `waveforms`, `waveform_lengths`, `labels`, `label_lengths` | — | Prepared | — |
| `espnet-feature-model-ready` | `features`, `feature_lengths`, `labels`, `label_lengths` | — | Prepared | — |

ESPnet Transformer joint CTC/attention raw and cached records. Follow the [shared data workflow](../../guides/speech-data.md) for
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
| Training checkpoint | `espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model` | `labels`, `label_lengths` | `loss`, `ctc_loss`, `attention_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best`](https://huggingface.co/espnet/shinji-watanabe-librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.asr_native.espnet.ESPnetASRForSpeechRecognition` |
| Configuration | `voicehub.models.asr_native.configuration.ESPnetASRConfig` |
| Source provenance | `voicehub/architectures/espnet_transformer/SOURCE.json` |
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

### `ESPnetASRConfig`

[View `ESPnetASRConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/configuration.py)

```text
ESPnetASRConfig(**config_kwargs)
```

### `ESPnetASRForSpeechRecognition`

[View `ESPnetASRForSpeechRecognition` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/asr_native/espnet.py)

```text
AutoModelForSpeechRecognition.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='asr_espnet',
    config=None,
    **model_kwargs,
)
```

The loader returns `ESPnetASRForSpeechRecognition` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('asr_espnet')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_espnet')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `ESPnetASRConfig` |
| Process | `AutoProcessor` |
| Model implementation | `ESPnetASRForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_espnet')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
