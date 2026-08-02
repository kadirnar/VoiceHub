---
description: Public API, checkpoint, training, and optimization guide for the vits integration.
---

# `vits` model guide

## Overview

`vits` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `vits` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/vits.ipynb).

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
    'facebook/mms-tts-eng',
    model_type='vits',
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
| Architecture | `vits` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `multilingual`, `mms-tts`, `safetensors`, `fine-tuning`, `voicehub-native`, `native-runtime`, `raw-audio-training`, `preprocessed-training`, `adversarial-training`, `generator-warm-start`, `explicit-acoustic-training-config` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `vits` |
| Sample rate | Model/checkpoint specific |
| Contract getter | `get_tts_dataset_spec('vits')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-adversarial` | `text` | audio / audio_values | Source | — |
| `tokenized-raw-adversarial` | `input_ids` | audio / audio_values | Source | — |
| `precomputed-spectrogram` | `spectrogram` | text / input_ids; audio / audio_values | Prepared | — |

VITS/GAN text, waveform, spectrogram, and adversarial data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.vits.modeling_vits.VitsForTextToSpeech` |
| Configuration | `voicehub.models.vits.configuration_vits.VitsConfig` |
| Source provenance | `voicehub/architectures/vits/SOURCE.json` |
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
| Family | `vits` |
| Recipe | `adversarial` |
| Default phase | `generator` |
| Training checkpoint | `facebook/mms-tts-eng` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `discriminator` | discriminator | `training_model.discriminator` | `input_ids`, `audio_values` | `loss` |
| `generator` | generator | `training_model.native_model` | `input_ids`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('vits')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `VitsConfig` |
| Model implementation | `VitsForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('vits')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
