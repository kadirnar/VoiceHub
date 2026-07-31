---
description: Install VoiceHub and run simple TTS, ASR, and VAD examples.
---

# Quickstart

## Install

```bash
python -m pip install voicehub
```

GPU users should install the correct PyTorch build first. See
[Installation](installation.md) for CPU, CUDA, Git, wheel, and editable
setups.

## Discover models

Discovery is lazy: it does not load checkpoints.

```python
from voicehub import list_model_specs

for spec in list_model_specs(task=None):
    print(spec.model_type, spec.task.value, spec.default_model_path)
```

## Generate at least 10 seconds of speech

The sample is deliberately long, but speaking rate varies. Always calculate
the duration of the returned waveform.

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

text = (
    "VoiceHub keeps speech experiments easy to inspect and repeat. This "
    "long sample checks pronunciation, pacing, pauses, and consistency "
    "across several complete sentences. The same prompt and seed can then "
    "be reused to compare eager inference with each supported optimization. "
    "Listen for stable volume, natural pauses, clear endings, and consistent "
    "tone throughout the complete generated recording."
)

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = model.generate(
    text,
    description="A clear speaker talks at a steady, natural pace.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file="artifacts/quickstart.wav",
    ),
)

sample_count = output.audio.shape[-1] if hasattr(output.audio, "shape") else len(output.audio)
duration = sample_count / output.sample_rate
if duration < 10:
    raise RuntimeError(f"Expected at least 10 seconds, got {duration:.2f}")
print(output.file_path, output.sample_rate, f"{duration:.2f}s")
```

## Transcribe audio

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
)
result = model.transcribe("speech.wav", language="English")
print(result.text)
```

## Detect speech regions

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    model_type="vad_silero",
)
result = model.detect("speech.wav", threshold=0.55)
for segment in result.segments:
    print(segment.start, segment.end, segment.score)
```

## Next steps

- [Inference](../guides/inference.md): conditioning, local artifacts, and
  reproducible requests.
- [TTS optimization](../guides/tts-optimization.md): inspect support, apply
  quality-preserving passes, and benchmark correctly.
- [Speech recognition](../guides/speech-recognition.md) and
  [voice activity detection](../guides/voice-activity-detection.md): model
  inputs and normalized outputs.
- [Training](../guides/training.md): data contracts, one-step smoke tests,
  resume, and export.
- [Notebooks](../guides/notebook.md): short Colab workflows.
