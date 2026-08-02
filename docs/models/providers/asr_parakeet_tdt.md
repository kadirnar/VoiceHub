---
description: Public API, checkpoint, training, and optimization guide for the asr_parakeet_tdt integration.
---

# `asr_parakeet_tdt` model guide

## Overview

`asr_parakeet_tdt` is a VoiceHub **automatic speech recognition**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `asr_parakeet_tdt` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/asr_parakeet_tdt.ipynb).

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
    'nvidia/parakeet-tdt-0.6b-v3',
    model_type='asr_parakeet_tdt',
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
| Architecture | `parakeet-tdt` |
| Runtime | `VoiceHub-native` |
| Capabilities | `automatic-speech-recognition`, `multilingual`, `timestamps`, `long-form`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `tdt` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_asr_dataset_spec('asr_parakeet_tdt')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `audio` | text / transcription / transcript | Source | at most one: text / transcription / transcript |
| `parakeet-tdt-model-ready` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | — | Prepared | — |

Parakeet token-duration transducer audio and transcript records. Follow the [shared data workflow](../../guides/speech-data.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.asr_parakeet_tdt.modeling_asr_parakeet_tdt.ParakeetTDTForSpeechRecognition` |
| Configuration | `voicehub.models.asr_parakeet_tdt.configuration_asr_parakeet_tdt.ParakeetTDTASRConfig` |
| Source provenance | `voicehub/architectures/parakeet_tdt/SOURCE.json` |
| License | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

The pinned Parakeet TDT checkpoint and derivatives require CC-BY-4.0 attribution. The VoiceHub-owned architecture port is audited against Apache-2.0 Transformers and NeMo source. Commercial use: **allowed by the registered terms**.

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
| Family | `tdt` |
| Recipe | `single-phase` |
| Default phase | `speech_recognition` |
| Training checkpoint | `nvidia/parakeet-tdt-0.6b-v3` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `speech_recognition` | objective | `model.encoder`, `model.encoder_projector`, `model.decoder`, `model.joint` | `input_features`, `attention_mask`, `labels`, `decoder_input_ids` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('asr_parakeet_tdt')` |
| Load and run | `AutoModelForSpeechRecognition` |
| Configure | `ParakeetTDTASRConfig` |
| Model implementation | `ParakeetTDTForSpeechRecognition` |
| Normalized output | `ASROutput` |
| Training contract | `get_training_spec('asr_parakeet_tdt')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
