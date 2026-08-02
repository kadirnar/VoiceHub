---
description: Public API, checkpoint, training, and optimization guide for the voxcpm integration.
---

# `voxcpm` model guide

## Overview

`voxcpm` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `voxcpm` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/voxcpm.ipynb).

## Quickstart

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Set the input text and generation options for your use case.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'openbmb/VoxCPM2',
    model_type='voxcpm',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
}
output = model.generate(
    "VoiceHub keeps model integrations consistent and easy to extend.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file=Path("output.wav"),
    ),
    **generation_kwargs,
)
print(output.file_path, output.sample_rate)
```

Use only authorized recordings for reference voice, transcription, detection,
or evaluation. The example selects a concrete device; verify checkpoint-specific
hardware needs and pin an immutable revision before production use.

## Supported tasks and capabilities

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `voxcpm2` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `audio-continuation`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `diffusion` |
| Sample rate | 16,000 Hz |
| Contract getter | `get_tts_dataset_spec('voxcpm')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-waveform` | `text` | audio / waveform | Source | at most one: audio / waveform; forbidden: audio_features |
| `audio-features` | `text`, `audio_features` | — | Prepared | forbidden: audio, waveform |

Conditional flow-matching, rectified-flow, or diffusion data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.voxcpm.modeling_voxcpm.VoxCPMForTextToSpeech` |
| Configuration | `voicehub.models.voxcpm.configuration_voxcpm.VoxCPMConfig` |
| Source provenance | `voicehub/models/voxcpm/source/SOURCE.json` |
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
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `source_flow_and_stop` |
| Training checkpoint | `openbmb/VoxCPM2` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `source_flow_and_stop` | objective | `model` | `text_tokens`, `text_mask`, `audio_feats`, `audio_mask`, `loss_mask`, `position_ids`, `labels` | `diffusion_loss`, `stop_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('voxcpm')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VoxCPMConfig` |
| Model implementation | `VoxCPMForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('voxcpm')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
