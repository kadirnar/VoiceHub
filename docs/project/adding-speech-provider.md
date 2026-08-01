---
description: Add an ASR or VAD provider with normalized audio inputs and outputs.
---

# Add an ASR or VAD provider

Follow the common [model integration recipe](adding-a-model.md). This page
only covers the ASR and VAD differences.

## Choose the right task

| Task | Base class | Implement | Return |
| --- | --- | --- | --- |
| ASR | `PreTrainedASRModel` | `_transcribe` | `ASROutput` |
| VAD | `PreTrainedVADModel` | `_detect` | `VADOutput` |

Use a distinct provider only when a checkpoint family needs a different graph,
processor, output normalization, or training recipe. Otherwise add the
checkpoint to an existing provider.

## ASR example

```python
from voicehub import ASROutput, AutoModelForSpeechRecognition, PreTrainedASRModel


class AcmeASRForSpeechRecognition(PreTrainedASRModel):
    config_class = AcmeASRConfig

    def _load_pretrained_model(self):
        from .runtime import load_asr

        self.model = load_asr(self.config.name_or_path, device=self.device)

    def _transcribe(self, audio, **kwargs):
        result = self.model.transcribe(audio, **kwargs)
        return ASROutput(text=result.text, segments=tuple(result.segments))


AutoModelForSpeechRecognition.register(
    AcmeASRConfig,
    AcmeASRForSpeechRecognition,
    default_model_path="acme/asr-base",
    aliases=("acme-stt",),
)
```

Do not invent timestamps, confidence values, or language labels that the
runtime did not produce.

## VAD example

```python
from voicehub import AutoModelForVoiceActivityDetection, PreTrainedVADModel, VADOutput


class AcmeVADForVoiceActivityDetection(PreTrainedVADModel):
    config_class = AcmeVADConfig

    def _load_pretrained_model(self):
        from .runtime import load_vad

        self.model = load_vad(self.config.name_or_path, device=self.device)

    def _detect(self, audio, **kwargs):
        segments = self.model.detect(audio, **kwargs)
        return VADOutput(segments=tuple(segments))


AutoModelForVoiceActivityDetection.register(
    AcmeVADConfig,
    AcmeVADForVoiceActivityDetection,
    default_model_path="acme/vad-base",
)
```

VAD segments must be ordered, non-overlapping, and measured in seconds.

## Inputs and secrets

The shared audio processor accepts supported paths, arrays, tensors, mappings,
and `AudioInput` values. Resample in one documented place. Keep API keys and
tokens on the live client; never serialize them in config or model metadata.

## Training

Declare the real objective family: CTC, sequence-to-sequence, RNNT, TDT,
audio classification, or frame classification. Use a custom adapter when the
published recipe needs special phases, optimizers, augmentation, or export.
Use `INFERENCE_ONLY` for fixed, quantized, ONNX, or otherwise
non-differentiable runtimes.

## Tests

In addition to the common model tests, cover:

- file and in-memory audio inputs;
- sample-rate handling;
- normalized ASR/VAD output ordering;
- task-factory mismatch errors;
- sequential and concurrent lifecycle behavior;
- the declared training boundary.

See the [ASR and VAD support matrix](../models/asr-vad-support.md) for the
current public contract.
