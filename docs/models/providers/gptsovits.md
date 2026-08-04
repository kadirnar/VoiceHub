---
description: Public API, checkpoint, training, and optimization guide for the gptsovits integration.
---

# GPTSoVITS {.vh-model-title}

## Usage

```bash
python -m pip install voicehub
```

1. Install VoiceHub and the provider extra shown above.
2. Choose a checkpoint that matches this integration.
3. Provide an authorized `reference.wav` and an exact reference transcript when the example requests them.
4. Generate audio and inspect the returned sample rate and metadata.

```python
from pathlib import Path

REFERENCE_AUDIO = Path("reference.wav")
REFERENCE_TEXT = "This transcript must exactly match the authorized reference audio."


from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

model = AutoModelForTextToSpeech.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
    device="cuda",
    lazy_load=True,
)
generation_kwargs = {
    "speaker_audio_path": str(REFERENCE_AUDIO),
    "prompt_text": REFERENCE_TEXT,
    "text_language": "en",
    "prompt_language": "en",
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

GPTSoVITS uses the canonical model type `gptsovits` and is a
VoiceHub **text to speech** integration. This page is
generated from the model registry and its executable data and training
contracts, so the documented support stays aligned with code. [Open the `gptsovits` Colab notebook](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/models/gptsovits.ipynb).

| Property | Value |
| --- | --- |
| Task | Text to speech |
| Architecture | `gptsovits` |
| Runtime | `VoiceHub-native` |
| Languages | 5 enumerated languages |
| Capabilities | `text-to-speech`, `voice-cloning`, `multilingual`, `fine-tuning`, `safetensors`, `voicehub-native`, `native-runtime`, `preprocessed-training`, `gpt-sovits-v1`, `gpt-sovits-v2`, `gpt-sovits-v2-pro`, `gpt-sovits-v2-pro-plus`, `prepared-pro-speaker-conditioning`, `variant-aware-safetensors-export` |
| Reusable components | — |
| Normalized output | `TTSOutput` |

### Language support

<details class="vh-language-support" markdown>
<summary>5 documented languages</summary>

`zh`, `en`, `ja`, `ko`, `yue`

Korean and Cantonese support applies to V2 and later variants.

</details>

## Configuration

Load the registered configuration without constructing the model. The canonical
key remains serializable even though the page uses a presentation label.

```python
from voicehub import AutoConfig

config = AutoConfig.for_model('gptsovits')
print(config.model_type)
```

| Property | Value |
| --- | --- |
| Canonical model type | `gptsovits` |
| Configuration class | `GPTSoVITSConfig` |
| Architecture class | `GPTSoVITSForTextToSpeech` |

## Processing

`AutoProcessor` resolves the processor declared by the registered model. Creating
the processor does not allocate model weights.

```python
from voicehub import AutoProcessor

processor = AutoProcessor.from_pretrained(
    'lj1995/GPT-SoVITS',
    model_type='gptsovits',
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
| Readiness | `preprocessed` |
| Data architecture | `hybrid` |
| Sample rate | 32,000 Hz |
| Contract getter | `get_tts_dataset_spec('gptsovits')` |

| Variant | Required fields | One of | Boundary | Other rules |
| --- | --- | --- | --- | --- |
| `s1-preprocessed` | `phoneme_ids`, `semantic_ids`, `bert_features` | — | Prepared | — |
| `s2-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids` | — | Prepared | — |
| `s2-pro-preprocessed` | `ssl_features`, `spectrogram`, `audio_values`, `phoneme_ids`, `speaker_embedding` | — | Prepared | — |

Multi-component language-model, diffusion, acoustic, or GAN data. Follow the [shared data workflow](../../guides/data-preparation.md) for
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
| Support | `preprocessed` |
| Family | `composite` |
| Recipe | `adversarial` |
| Default phase | `s1` |
| Training checkpoint | `lj1995/GPT-SoVITS` |
| Native training graph | `yes` |

| Phase | Kind | Components | Required inputs | Loss keys |
| --- | --- | --- | --- | --- |
| `s1` | objective | `training_model.s1` | `phoneme_ids`, `phoneme_lengths`, `semantic_ids`, `semantic_lengths`, `bert_features` | `loss` |
| `s2_generator` | generator | `training_model.s2.generator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |
| `s2_discriminator` | discriminator | `training_model.s2.discriminator` | `ssl_features`, `spectrogram`, `spectrogram_lengths`, `audio_values`, `phoneme_ids`, `phoneme_lengths` | `loss` |

Prepare the exact tensors listed in the data contract before this step. Call `model.validate_training_support()` before constructing a
trainer. Follow the [shared training workflow](../../guides/training.md) for a
one-step smoke test, validation, checkpoint resume, optimization, and portable
export.

## Checkpoints, provenance, license, and limitations

| Property | Value |
| --- | --- |
| Default checkpoint | [`lj1995/GPT-SoVITS`](https://huggingface.co/lj1995/GPT-SoVITS) |
| Checkpoint status | Registry default; pin an immutable revision for production and reproducible evidence |
| Optional dependency extra | Core package |
| Hardware and runtime | Usage selects `cuda`; verify checkpoint-specific requirements |
| Real-checkpoint evidence | [Release evidence](../../project/release-readiness.md); a registry default alone is not execution evidence |
| Implementation | `voicehub.models.gptsovits.modeling_gptsovits.GPTSoVITSForTextToSpeech` |
| Configuration | `voicehub.models.gptsovits.configuration_gptsovits.GPTSoVITSConfig` |
| Source provenance | `voicehub/models/gptsovits/source/SOURCE.json` |
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

### `GPTSoVITSConfig`

[View `GPTSoVITSConfig` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/configuration_gptsovits.py)

```text
GPTSoVITSConfig(**config_kwargs)
```

### `GPTSoVITSForTextToSpeech`

[View `GPTSoVITSForTextToSpeech` source](https://github.com/kadirnar/voicehub/blob/main/voicehub/models/gptsovits/modeling_gptsovits.py)

```text
AutoModelForTextToSpeech.from_pretrained(
    pretrained_model_name_or_path,
    *,
    model_type='gptsovits',
    config=None,
    **model_kwargs,
)
```

The loader returns `GPTSoVITSForTextToSpeech` through the shared task-specific factory.

```python
from voicehub import get_model_spec

spec = get_model_spec('gptsovits')
print(spec.display_name, spec.task.value)
```

| Purpose | Public object |
| --- | --- |
| Discover | `get_model_spec('gptsovits')` |
| Load and run | `AutoModelForTextToSpeech` |
| Configure | `GPTSoVITSConfig` |
| Process | `AutoProcessor` |
| Model implementation | `GPTSoVITSForTextToSpeech` |
| Normalized output | `TTSOutput` |
| Training contract | `get_training_spec('gptsovits')` |
| Optimization lifecycle | `available_optimization_passes`, `apply_optimization_plan`, `optimization_manifest`, `restore_optimization_plan` |

Related shared documentation:

- [All model guides](index.md)
- [Shared inference guides](../../guides/index.md)
- [Model and training support matrices](../training-support.md)
