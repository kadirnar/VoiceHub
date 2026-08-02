---
description: Public API, checkpoint, training, and optimization guide for the vad_auditok integration.
---

# `vad_auditok` model guide

## Overview

`vad_auditok` is a VoiceHub **voice activity detection**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.

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
    'auditok-energy-vad',
    model_type='vad_auditok',
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
| Architecture | `energy-vad` |
| Runtime | `VoiceHub-native` |
| Capabilities | `voice-activity-detection`, `energy-based`, `adaptive-threshold`, `algorithmic`, `voicehub-native` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Label boundary | No verified training dataset contract |
| Required training inputs | — |

Use authorized audio and preserve annotation provenance. Follow the
[ASR and VAD data workflow](../../guides/speech-data.md) for supported audio
forms, timestamp labels, frame targets, leakage-safe splits, and evaluation.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | `auditok-energy-vad` |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.vad_auditok.modeling_vad_auditok.AuditokVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_auditok.configuration_vad_auditok.AuditokVADConfig` |
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
| Support | `inference-only` |
| Family | `upstream-native` |
| Recipe | `single-phase` |
| Default phase | `default` |
| Training checkpoint | `auditok-energy-vad` |
| Native training graph | `no` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `default` | objective | — | — | `loss`, `total_loss` |

This integration is intentionally **inference-only**. VoiceHub has no verified
gradient-bearing graph, loss, and reloadable training artifact for it. Do not
attach a generic loss to inference output. Choose a trainable model from the
[training matrix](../training-support.md), or contribute a tested training
adapter and data contract.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_auditok')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `AuditokVADConfig` |
| Model implementation | `AuditokVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_auditok')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
