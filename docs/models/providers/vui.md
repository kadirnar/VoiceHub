---
description: Public API, checkpoint, training, and optimization guide for the vui integration.
---

# `vui` model guide

## Overview

`vui` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.

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
    'vui-abraham-100m.pt',
    model_type='vui',
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
| Architecture | `vui` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `fine-tuning`, `safetensors`, `standalone-safetensors-export`, `voicehub-native`, `native-runtime`, `preprocessed-training` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `codec-lm` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('vui')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `codec-batch` | `input_ids`, `audio_codes` | — | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | `vui-abraham-100m.pt` |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.vui.modeling_vui.VuiForTextToSpeech` |
| Configuration | `voicehub.models.vui.configuration_vui.VuiConfig` |
| Source provenance | `voicehub/models/vui/SOURCE.json` |
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
| Support | `preprocessed` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `codec_language_model` |
| Training checkpoint | `vui-abraham-100m.pt` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | `input_ids`, `audio_codes` | `loss`, `codec_ce_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vui')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VuiConfig` |
| Model implementation | `VuiForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('vui')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
