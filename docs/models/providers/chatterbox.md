---
description: Public API, checkpoint, training, and optimization guide for the chatterbox integration.
---

# `chatterbox` model guide

## Overview

`chatterbox` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `chatterbox` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/chatterbox.ipynb).

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
    'ResembleAI/chatterbox',
    model_type='chatterbox',
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
| Architecture | `chatterbox` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `raw-audio-fine-tuning` |
| Reusable components | `conformer` |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `hybrid` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('chatterbox')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `t3-raw` | `text` | audio / audio_path | Source | — |
| `flow-raw` | — | audio / audio_path | Source | — |
| `t3-precomputed` | `text_tokens`, `speech_tokens`, `speaker_emb` | — | Prepared | — |
| `flow-precomputed` | `speech_token`, `speech_feat`, `embedding` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`ResembleAI/chatterbox`](https://huggingface.co/ResembleAI/chatterbox) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.chatterbox.modeling_chatterbox.ChatterboxForTextToSpeech` |
| Configuration | `voicehub.models.chatterbox.configuration_chatterbox.ChatterboxConfig` |
| Source provenance | `voicehub/models/chatterbox/source/SOURCE.json` |
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
| Support | `custom` |
| Family | `composite` |
| Recipe | `multi-phase` |
| Default phase | `language_model` |
| Training checkpoint | `ResembleAI/chatterbox` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `language_model` | objective | `model.t3` | — | `loss`, `text_loss`, `speech_token_loss` |
| `flow` | objective | `model.s3gen.flow` | — | `loss`, `flow_loss`, `diffusion_loss` |

This profile uses model-specific phases; inspect and honor each phase boundary. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('chatterbox')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `ChatterboxConfig` |
| Model implementation | `ChatterboxForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('chatterbox')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
