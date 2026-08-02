---
description: Public API, checkpoint, training, and optimization guide for the zonos integration.
---

# `zonos` model guide

## Overview

`zonos` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `zonos` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/zonos.ipynb).

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
    'Zyphra/Zonos-v0.1-transformer',
    model_type='zonos',
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
| Architecture | `zonos` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | `dac` |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `codec-lm` |
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('zonos')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `codec-batch` | `prefix_conditioning`, `audio_codes` | — | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`Zyphra/Zonos-v0.1-transformer`](https://huggingface.co/Zyphra/Zonos-v0.1-transformer) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.zonos.modeling_zonos.ZonosForTextToSpeech` |
| Configuration | `voicehub.models.zonos.configuration_zonos.ZonosConfig` |
| Source provenance | `voicehub/models/zonos/source/SOURCE.json` |
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
| Default phase | `reconstructed_codec_language_model` |
| Training checkpoint | `Zyphra/Zonos-v0.1-transformer` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `reconstructed_codec_language_model` | objective | `model` | `prefix_conditioning`, `audio_codes` | `loss`, `codec_ce_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('zonos')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ZonosConfig` |
| Model implementation | `ZonosForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('zonos')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
