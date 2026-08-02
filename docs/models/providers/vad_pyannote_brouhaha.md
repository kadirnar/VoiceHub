---
description: Public API, checkpoint, training, and optimization guide for the vad_pyannote_brouhaha integration.
---

# `vad_pyannote_brouhaha` model guide

## Overview

`vad_pyannote_brouhaha` is a VoiceHub **voice activity detection**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `vad_pyannote_brouhaha` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vad_pyannote_brouhaha.ipynb).

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
    'pyannote/brouhaha',
    model_type='vad_pyannote_brouhaha',
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
| Architecture | `pyannet` |
| Runtime | `VoiceHub-native` |
| Capabilities | `voice-activity-detection`, `gated-checkpoint`, `voicehub-native`, `trusted-checkpoint-conversion`, `safetensors`, `frame-scores`, `snr`, `c50`, `fine-tuning` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Label boundary | Clip-, frame-, or segment-level labels |
| Required training inputs | `waveforms`, `labels` |

Use authorized audio and preserve annotation provenance. Follow the
[ASR and VAD data workflow](../../guides/speech-data.md) for supported audio
forms, timestamp labels, frame targets, leakage-safe splits, and evaluation.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`pyannote/brouhaha`](https://huggingface.co/pyannote/brouhaha) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.vad_pyannote_brouhaha.modeling_vad_pyannote_brouhaha.PyannoteBrouhahaVADForVoiceActivityDetection` |
| Configuration | `voicehub.models.vad_pyannote_brouhaha.configuration_vad_pyannote_brouhaha.PyannoteBrouhahaVADConfig` |
| Source provenance | `voicehub/architectures/pyannet/SOURCE.json` |
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
| Support | `native` |
| Family | `composite` |
| Recipe | `single-phase` |
| Default phase | `vad_snr_c50` |
| Training checkpoint | `pyannote/brouhaha` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `vad_snr_c50` | objective | `model` | `waveforms`, `labels` | `loss`, `loss_vad`, `loss_snr`, `loss_c50` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vad_pyannote_brouhaha')` |
| Load and run | `AutoModelForVoiceActivityDetection` |
| Configure | `PyannoteBrouhahaVADConfig` |
| Model implementation | `PyannoteBrouhahaVADForVoiceActivityDetection` |
| Normalized output | `VADOutput` |
| Training contract | `get_training_spec('vad_pyannote_brouhaha')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
