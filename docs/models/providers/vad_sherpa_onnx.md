---
description: Public API, checkpoint, training, and optimization guide for the vad_sherpa_onnx integration.
---

# `vad_sherpa_onnx` model guide

## Overview

`vad_sherpa_onnx` is a VoiceHub **voice activity detection**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `vad_sherpa_onnx` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_sherpa_onnx.ipynb).

## Quickstart

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Place a supported recording at `speech.wav`.
4. Run detection and tune the threshold against labeled validation audio.

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    'safestack/silero-vad',
    model_type='vad_sherpa_onnx',
    device="cpu",
    lazy_load=True,
)
output = model.detect("speech.wav", threshold=0.5)
for segment in output.segments:
    print(segment.start, segment.end, segment.score)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. The example selects a concrete device; verify checkpoint-specific
hardware needs and pin an immutable revision before production use.

## Supported tasks and capabilities

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `native-vad-dispatch` |
| Runtime | `VoiceHub-native` |
| Capabilities | `voice-activity-detection`, `voicehub-native`, `safetensors`, `explicit-onnx-weight-conversion`, `fine-tuning`, `streaming`, `sherpa-compatible-segmentation`, `silero`, `ten-vad` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Label boundary | Clip-, frame-, or segment-level labels |
| Required training inputs | `labels` |

Use authorized audio and preserve annotation provenance. Follow the
[ASR and VAD data workflow](../../guides/speech-data.md) for supported audio
forms, timestamp labels, frame targets, leakage-safe splits, and evaluation.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`safestack/silero-vad`](https://huggingface.co/safestack/silero-vad) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.vad_sherpa_onnx.modeling_vad_sherpa_onnx.SherpaONNXVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_sherpa_onnx.configuration_vad_sherpa_onnx.SherpaONNXVADConfig` |
| Source provenance | `voicehub/architectures/ten_vad/SOURCE.json` |
| License | [LicenseRef-TEN-VAD-Open-Source-License](https://github.com/TEN-framework/ten-vad) |

The provider's optional TEN family is governed by a non-standard license with additional deployment restrictions, including limits on competing with Agora. Review the bundled THIRD_PARTY_LICENSE before conversion, fine-tuning, distribution, or deployment. The default Silero family retains its own checkpoint terms. Commercial use: **review required**.

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
| Family | `frame-classification` |
| Recipe | `single-phase` |
| Default phase | `voice_activity_detection` |
| Training checkpoint | `safestack/silero-vad` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `voice_activity_detection` | objective | `model` | `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_sherpa_onnx')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `SherpaONNXVADConfig` |
| Model implementation | `SherpaONNXVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_sherpa_onnx')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
