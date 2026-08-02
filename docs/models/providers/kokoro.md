---
description: Public API, checkpoint, training, and optimization guide for the kokoro integration.
---

# `kokoro` model guide

## Overview

`kokoro` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `kokoro` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/kokoro.ipynb).

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
    'hexgrad/Kokoro-82M',
    model_type='kokoro',
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
| Architecture | `kokoro` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime` |
| Reusable components | — |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `acoustic` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('kokoro')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `full-preprocessed` | `durations` | input_ids / phonemes; ref_s / voice; audio_values / audio / labels | Prepared | — |
| `duration-only` | `durations`, `training_phase` | input_ids / phonemes; ref_s / voice | Prepared | — |

Direct acoustic, mel, codec, or waveform regression data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`hexgrad/Kokoro-82M`](https://huggingface.co/hexgrad/Kokoro-82M) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.kokoro.modeling_kokoro.KokoroForTextToSpeech` |
| Configuration | `voicehub.models.kokoro.configuration_kokoro.KokoroConfig` |
| Source provenance | `voicehub/models/kokoro/source/SOURCE.json` |
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
| Family | `acoustic-regression` |
| Recipe | `multi-phase` |
| Default phase | `acoustic` |
| Training checkpoint | `hexgrad/Kokoro-82M` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `duration` | objective | `model.bert`, `model.bert_encoder`, `model.predictor` | `input_ids`, `ref_s`, `durations` | `loss` |
| `acoustic` | objective | `model.bert`, `model.bert_encoder`, `model.predictor`, `model.text_encoder`, `model.decoder` | `input_ids`, `ref_s`, `durations`, `audio_values` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('kokoro')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `KokoroConfig` |
| Model implementation | `KokoroForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('kokoro')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
