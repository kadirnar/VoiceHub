---
description: Public API, checkpoint, training, and optimization guide for the omnivoice integration.
---

# `omnivoice` model guide

## Overview

`omnivoice` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `omnivoice` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/omnivoice.ipynb).

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
    'k2-fsa/OmniVoice',
    model_type='omnivoice',
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
| Architecture | `omnivoice` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `voice-design`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning`, `preencoded-code-fine-tuning` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('omnivoice')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `raw-audio` | `text` | audio / waveform | Source | at most one: audio / waveform; forbidden: audio_tokens |
| `audio-tokens` | `text`, `audio_tokens` | — | Prepared | forbidden: audio, waveform |

Multi-component language-model, diffusion, acoustic, or GAN data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.omnivoice.modeling_omnivoice.OmniVoiceForTextToSpeech` |
| Configuration | `voicehub.models.omnivoice.configuration_omnivoice.OmniVoiceConfig` |
| Source provenance | `voicehub/models/omnivoice/source/SOURCE.json` |
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
| Default phase | `masked_audio` |
| Training checkpoint | `k2-fsa/OmniVoice` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `masked_audio` | objective | `model` | `input_ids`, `audio_mask`, `labels` | `loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('omnivoice')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `OmniVoiceConfig` |
| Model implementation | `OmniVoiceForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('omnivoice')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
