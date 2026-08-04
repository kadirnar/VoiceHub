---
description: Public API, checkpoint, training, and optimization guide for the vad_webrtc integration.
---

# WebRTCVAD {.vh-model-title}

## Usage

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
    'webrtc-vad',
    model_type='vad_webrtc',
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

## Overview

WebRTCVAD uses the canonical model type `vad_webrtc` and is a
VoiceHub **voice activity detection** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code.

| Property | Value |
| --- | --- |
| Task | Voice activity detection |
| Architecture | `webrtc-vad` |
| Runtime | `VoiceHub-native` |
| Languages | Not text-language conditioned |
| Capabilities | `voice-activity-detection`, `fixed-point`, `voicehub-native`, `native-runtime`, `streaming` |
| Reusable components | — |
| Normalized output | `VADOutput` |

### Language support

The public VAD contract does not select a spoken language; validate checkpoint acoustic coverage on the target languages and recording conditions.

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('vad_webrtc')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `vad_webrtc` |
| Configuration class | `WebRTCVADConfig` |
| Architecture class | `WebRTCVADForVoiceActivityDetection` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'webrtc-vad',
    model_type='vad_webrtc',
)
print(type(processor).__name__)
```

Processor behavior remains model-owned when text normalization, audio loading,
feature extraction, or reference speech requires provider-specific semantics.

## Inference

The Usage example returns `VADOutput` through `AutoModelForVoiceActivityDetection`. Inputs are validated
against the task and data contracts below before model-specific execution.

### Input and output contract

| Property | Value |
| --- | --- |
| Label boundary | No verified training dataset contract |
| Required training inputs | — |

Use authorized audio and preserve annotation provenance. Follow the
[ASR and VAD data workflow](../../guides/speech-data.md) for supported audio
forms, timestamp labels, frame targets, leakage-safe splits, and evaluation.

## Training and optimization

All public optimizations enter this model through the shared
`BaseSpeechModel` lifecycle. Use `available_optimization_passes()` to discover
the public pass registry, then apply, inspect, serialize, or restore a plan
through the common model API. Application remains fail-closed when the active
runtime or hardware cannot satisfy a pass.

### Training contract

| Property | Value |
| --- | --- |
| Support | `inference-only` |
| Family | `upstream-native` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Training checkpoint | `webrtc-vad` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is intentionally **inference-only**. VoiceHub has no verified
gradient-bearing graph, loss, and reloadable training artifact for it. Do not
attach a generic loss to inference output. Choose a trainable model from the
[training matrix](../training-support.md), or contribute a tested training
adapter and data contract.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | `webrtc-vad` |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cpu`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.vad_webrtc.modeling_vad_webrtc.WebRTCVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_webrtc.configuration_vad_webrtc.WebRTCVADConfig` |
| Source provenance | `voicehub/architectures/webrtc_vad/SOURCE.json` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- The Usage example selects `cpu`; validate memory, precision,
  and optional dependency requirements on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not substitute for released-checkpoint evidence. Consult the
  linked release record before treating a checkpoint path as verified.

## Public API

The stable configuration and model facades keep source inspection local while
the task auto class owns pretrained loading and normalized output behavior.

### `WebRTCVADConfig`

[View `WebRTCVADConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/configuration_vad_webrtc.py)

```text
WebRTCVADConfig(**config_kwargs)
```

### `WebRTCVADForVoiceActivityDetection`

[View `WebRTCVADForVoiceActivityDetection` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/vad_webrtc/modeling_vad_webrtc.py)

```text
AutoModelForVoiceActivityDetection.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='vad_webrtc',
    config=None,
    **model_kwargs,
)
```

The loader returns `WebRTCVADForVoiceActivityDetection` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('vad_webrtc')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_webrtc')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `WebRTCVADConfig` |
| Process | `AutoProcessor` |
| Model implementation | `WebRTCVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_webrtc')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
