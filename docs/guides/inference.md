---
description: Run reproducible TTS inference and inspect normalized VoiceHub outputs.
---

# TTS inference

VoiceHub gives every TTS model the same lifecycle:

1. choose a registered `model_type`;
2. construct a lazy model;
3. generate a `TTSOutput`; and
4. measure the returned waveform.

Conditioning still belongs to the selected architecture. A dialogue model,
voice-cloning model, and description-conditioned model require different
fields. Use the [TTS capability matrix](../models/tts-capabilities.md) before
changing models.

## Install

```bash
python -m pip install voicehub
```

GPU users should install the correct PyTorch build first. See
[Installation](../getting-started/installation.md).

## Discover models

Discovery does not load checkpoints:

```python
from voicehub import AutoModelForTextToSpeech

for spec in AutoModelForTextToSpeech.available_models():
    print(
        spec.model_type,
        spec.default_model_path,
        spec.capabilities,
    )
```

Use the canonical `model_type`. A checkpoint path alone is not always enough
to identify its architecture safely.

## Generate speech

This example uses a long prompt and verifies the actual duration. Word count
cannot guarantee duration because speaking rate varies.

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

text = (
    "VoiceHub keeps speech experiments simple and reproducible. This longer "
    "sample checks pacing, pronunciation, pauses, volume, and consistent tone "
    "across several complete sentences. We will measure the returned waveform "
    "instead of guessing its duration, then preserve the prompt, seed, model "
    "revision, and output file for a fair comparison."
)

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
    lazy_load=True,
)
output = model.generate(
    text,
    description="A clear speaker talks at a steady, natural pace.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file="artifacts/parler.wav",
    ),
)

sample_count = (
    output.audio.shape[-1]
    if hasattr(output.audio, "shape")
    else len(output.audio)
)
duration = sample_count / output.sample_rate
if duration < 10:
    raise RuntimeError(f"Expected at least 10 seconds, got {duration:.2f}")
print(output.file_path, output.sample_rate, f"{duration:.2f}s")
```

Construction is lazy. The first generation call loads the checkpoint. Call
`model.load()` when a service should fail during startup rather than on its
first request.

## Conditioning

Pass only fields supported by the selected model. Unknown fields raise instead
of being silently ignored.

Common patterns include:

- `description` for description-conditioned speech;
- `speaker_audio_path` plus an exact `reference_text` for voice cloning;
- `voice` or `speaker` for a model-owned voice preset;
- dialogue tags such as `[S1]` and `[S2]`; and
- preprocessed phonemes, codec IDs, or linguistic features where the model
  contract requires them.

Use only voices you are authorized to use. Keep voice references, consent,
license, and provenance with the request or dataset.

## Reproducible requests

`TTSGenerationConfig` separates shared request controls from model-specific
conditioning:

```python
from voicehub import TTSGenerationConfig

request = TTSGenerationConfig(
    seed=42,
    speed=1.0,
    output_file="artifacts/sample.wav",
)
output = model.generate(
    "Use an identical request when comparing two runtimes.",
    generation_config=request,
    description="A neutral studio recording.",
)
```

A seed improves repeatability but does not make every device, kernel, or
floating-point path bit-identical. Record the checkpoint revision, VoiceHub
version, PyTorch version, device, precision, and generation settings.

## Use the normalized output

Every backend returns `TTSOutput`:

| Field | Meaning |
| --- | --- |
| `audio` | Generated waveform |
| `sample_rate` | Waveform sample rate |
| `file_path` | Written path, when requested |
| `metadata` | Model-specific generation details |

Save another copy or unpack the waveform:

```python
output.save("artifacts/copy.wav")
audio, sample_rate = output.to_tuple()
```

## Local and remote artifacts

Pass either a model repository ID or a local VoiceHub artifact directory:

```python
local_model = AutoModelForTextToSpeech.from_pretrained(
    "/models/my-voicehub-export",
    model_type="parlertts",
    device="cuda",
)
```

Legacy pickle, JIT, ONNX, or provider-specific artifacts are accepted only by
integrations with an explicit, verified conversion path. Never enable trust
flags for an unverified file. Normal inference should use the converted,
portable artifact documented by the model matrix.

## Optimize only after the baseline works

Keep the eager sample as the quality baseline. Then use the
[TTS optimization guide](tts-optimization.md) to inspect support, apply one
change at a time, measure warm and cold latency plus peak memory, and listen
to both outputs. Configuration alone is not a benchmark.

## Troubleshooting

- Unknown `model_type`: choose a canonical key from discovery.
- Unsupported argument: check the selected model's conditioning contract.
- Missing local path: resolve it before model construction.
- Out of memory: choose a smaller checkpoint or reduce batch size; do not
  quantize or change precision without confirming the quality boundary.
- Training validation fails: inference support does not prove that the chosen
  checkpoint reconstructs a differentiable training graph.

See the [API reference](../reference/api.md), [training matrix](../models/training-support.md),
and [notebooks](notebook.md) for the next step.
