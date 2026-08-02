---
description: Public API, checkpoint, training, and optimization guide for the f5tts integration.
---

# `f5tts` model guide

## Overview

`f5tts` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code.

## Quickstart

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Provide an authorized `reference.wav` and an exact reference transcript when the example requests them.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'F5TTS_v1_Base',
    model_type='f5tts',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
    "reference_text": REFERENCE_TEXT,
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
| Architecture | `f5tts` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `fine-tuning`, `flow-matching`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | `vocos` |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `diffusion` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('f5tts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `waveform-vocab` | `input_values`, `input_ids` | — | Prepared | — |
| `mel-features` | `input_ids` | mel / mel_spec | Prepared | — |
| `native-ready` | `inp`, `text` | — | Prepared | — |

Conditional flow-matching, rectified-flow, or diffusion data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | `F5TTS_v1_Base` |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.f5tts.modeling_f5tts.F5TTSForTextToSpeech` |
| Configuration | `voicehub.models.f5tts.configuration_f5tts.F5TTSConfig` |
| Source provenance | `voicehub/models/f5tts/source/SOURCE.json` |
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
| Family | `flow-matching` |
| Recipe | `single-phase` |
| Default phase | `flow` |
| Training checkpoint | `F5TTS_v1_Base` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `flow` | objective | `model.ema_model` | `inp`, `text` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('f5tts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `F5TTSConfig` |
| Model implementation | `F5TTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('f5tts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
