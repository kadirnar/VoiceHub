---
description: Public API, checkpoint, training, and optimization guide for the fishtts integration.
---

# `fishtts` model guide

## Overview

`fishtts` is a VoiceHub **text to speech**
integration. This page is generated from the model registry and its executable
data and training contracts, so the documented support stays aligned with code. [Open the `fishtts` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/fishtts.ipynb).

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
    'fishaudio/s2-pro',
    model_type='fishtts',
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
| Architecture | `fish-s2` |
| Runtime | `VoiceHub-native` |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `noncommercial` |
| Reusable components | `dac` |

### Data contract

| Property | Value |
| --- | --- |
| Readiness | `preprocessed` |
| Data architecture | `codec-lm` |
| Sample rate | 44,100 Hz |
| Contract getter | `get_tts_dataset_spec('fishtts')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `semantic-tokens` | `labels` | tokens / inputs | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Checkpoints, provenance, and license

| Property | Value |
| --- | --- |
| Default checkpoint | [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Implementation | `voicehub.models.fishtts.modeling_fishtts.FishTTSForTextToSpeech` |
| Configuration | `voicehub.models.fishtts.configuration_fishtts.FishTTSConfig` |
| Source provenance | `voicehub/models/fishtts/source/SOURCE.json` |
| License | [Fish-Audio-Research-License](https://github.com/fishaudio/fish-speech) |

Fine-tuned checkpoints are derivative works. Commercial use requires a separate written Fish Audio license. Distribution must include the Fish Audio Research License, retain its exact copyright notice, and prominently display “Built with Fish Audio”. The license also restricts using materials, derivatives, or outputs to create or improve non-Fish foundational generative-AI models. Commercial use: **not allowed**.

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
| Default phase | `semantic` |
| Training checkpoint | `fishaudio/s2-pro` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `semantic` | objective | `model` | `inputs`, `labels` | `loss`, `base_loss`, `semantic_loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Public API

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('fishtts')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `FishTTSConfig` |
| Model implementation | `FishTTSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('fishtts')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
