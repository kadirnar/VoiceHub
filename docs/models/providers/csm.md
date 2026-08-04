---
description: Public API, checkpoint, training, and optimization guide for the csm integration.
---

# CSM {.vh-model-title}

## Usage

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
    'sesame/csm-1b',
    model_type='csm',
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

## Overview

CSM uses the canonical model type `csm` and is a
VoiceHub **text to speech** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `csm` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/csm.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `csm` |
| Runtime | `VoiceHub-native` |
| Languages | Checkpoint-defined; not exhaustively enumerated |
| Capabilities | `text-to-speech`, `voice-cloning`, `conversation`, `safetensors`, `fine-tuning`, `raw-audio-training`, `preencoded-code-training`, `voicehub-native`, `native-runtime` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

VoiceHub does not claim one exhaustive language list across compatible checkpoints; verify the selected checkpoint card and processor metadata.

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('csm')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `csm` |
| Configuration class | `CSMConfig` |
| Architecture class | `CSMForTextToSpeech` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'sesame/csm-1b',
    model_type='csm',
)
print(type(processor).__name__)
```

Processor behavior remains model-owned when text normalization, audio loading,
feature extraction, or reference speech requires provider-specific semantics.

## Inference

The Usage example returns `TTSOutput` through `AutoModelForTextToSpeech`. Inputs are validated
against the task and data contracts below before model-specific execution.

### Input and output contract

| Property | Value |
| --- | --- |
| Readiness | `integrated-raw` |
| Data architecture | `codec-lm` |
| Sample rate | 24,000 Hz |
| Contract getter | `get_tts_dataset_spec('csm')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `conversation` | — | conversation / messages | Source | — |
| `grouped-audios` | `texts`, `speaker_ids`, `audios` | — | Source | — |
| `grouped-concatenated` | `texts`, `speaker_ids`, `audio`, `audio_cut_idxs` | — | Source | — |
| `utterance` | `text`, `audio` | — | Source | — |
| `tokenized` | `input_ids`, `labels` | — | Prepared | — |

Autoregressive text/audio-token or codec-language-model data. Follow the [shared data workflow](../../guides/data-preparation.md) for
manifest loading, audio validation, leakage-safe splits, and model-owned
preprocessing.

## Training and optimization

All public optimizations enter this model through the shared
`BaseSpeechModel` lifecycle. Use `available_optimization_passes()` to discover
the public pass registry, then apply, inspect, serialize, or restore a plan
through the common model API. Application remains fail-closed when the active
runtime or hardware cannot satisfy a pass.

### Training contract

| Property | Value |
| --- | --- |
| Support | `native` |
| Family | `causal-lm` |
| Recipe | `single-phase` |
| Default phase | `codec_language_model` |
| Training checkpoint | `sesame/csm-1b` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `codec_language_model` | objective | `model` | — | `loss`, `backbone_loss`, `depth_decoder_loss` |

The integration accepts its declared source or prepared contract directly. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`sesame/csm-1b`](https://huggingface.co/sesame/csm-1b) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.csm.modeling_csm.CSMForTextToSpeech` |
| Configuration | `voicehub.models.csm.configuration_csm.CSMConfig` |
| Source provenance | `voicehub/models/csm/source/SOURCE.json` |
| License | Checkpoint-specific |

No VoiceHub-specific license override is registered. Verify the checkpoint and upstream source terms before use.

The default checkpoint identifies the expected family, not every compatible
variant. Confirm the selected checkpoint's revision, access terms, provenance,
and license before downloading or redistributing it.

### Limitations

- No integration-specific checkpoint limitation is registered. Verify the selected checkpoint revision and its documented runtime requirements.
- The Usage example selects `cuda`; validate memory, precision,
  and optional dependency requirements on the target system.
- Public optimizations fail closed when the runtime or hardware cannot satisfy
  their validation contract; an unavailable pass is not reported as applied.
- Contract tests do not substitute for released-checkpoint evidence. Consult the
  linked release record before treating a checkpoint path as verified.

## Public API

The stable configuration and model facades keep source inspection local while
the task auto class owns pretrained loading and normalized output behavior.

### `CSMConfig`

[View `CSMConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/csm/configuration_csm.py)

```text
CSMConfig(**config_kwargs)
```

### `CSMForTextToSpeech`

[View `CSMForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/csm/modeling_csm.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='csm',
    config=None,
    **model_kwargs,
)
```

The loader returns `CSMForTextToSpeech` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('csm')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('csm')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `CSMConfig` |
| Process | `AutoProcessor` |
| Model implementation | `CSMForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('csm')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
