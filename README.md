<div align="center">
  <img width="360" alt="VoiceHub" src="https://raw.githubusercontent.com/kadirnar/voicehub/main/assets/logo.png">
  <h1>VoiceHub</h1>
  <p>One Python interface for text-to-speech, speech recognition, and voice activity detection.</p>
</div>

VoiceHub provides lazy model loading, normalized outputs, shared optimization
controls, and a common trainer. Model weights are downloaded only when a
selected model is loaded.

## Install

VoiceHub supports Python 3.10 through 3.12.

```bash
python -m pip install voicehub
```

For fine-tuning:

```bash
python -m pip install "voicehub[training]"
```

GPU users should install the correct PyTorch build for their machine first.
See the [installation guide](https://kadirnar.github.io/voicehub/getting-started/installation/).

Verify the package without downloading a checkpoint:

```bash
python -c "import voicehub; print(voicehub.__version__, len(voicehub.list_model_specs()))"
```

## TTS

Use a long sample and verify the generated duration. Speaking rate varies by
model, so duration must be checked from the waveform rather than assumed from
word count.

```python
from voicehub import AutoModelForTextToSpeech, TTSGenerationConfig

text = (
    "Welcome to VoiceHub. This sample is intentionally long enough for a "
    "meaningful speech test. It checks pronunciation, pacing, sentence "
    "transitions, and sustained audio quality while the speaker explains a "
    "simple workflow for reliable text to speech inference. During this "
    "longer passage, listen for stable volume, natural pauses, clear word "
    "endings, and consistent tone from the opening sentence through the "
    "final measurement."
)

tts_model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = tts_model.generate(
    text,
    description="A clear speaker talks at a natural, relaxed pace.",
    generation_config=TTSGenerationConfig(
        seed=42,
        output_file="tts-sample.wav",
    ),
)

samples = (
    output.audio.shape[-1] if hasattr(output.audio, "shape") else len(output.audio)
)
duration = samples / output.sample_rate
if duration < 10:
    raise RuntimeError(f"Expected at least 10 seconds, generated {duration:.2f}")
print(output.file_path, f"{duration:.2f}s")
```

Model-specific conditioning fields such as speaker references, voices, or
descriptions are listed in the
[TTS model matrix](https://kadirnar.github.io/voicehub/models/tts-capabilities/).

## ASR

```python
from voicehub import AutoModelForSpeechRecognition

asr_model = AutoModelForSpeechRecognition.from_pretrained(
    "Qwen/Qwen3-ASR-0.6B",
    model_type="asr_qwen3",
    device="cuda",
)
output = asr_model.transcribe("speech.wav", language="English")
print(output.text)
```

## VAD

```python
from voicehub import AutoModelForVoiceActivityDetection

vad_model = AutoModelForVoiceActivityDetection.from_pretrained(
    model_type="vad_silero",
)
output = vad_model.detect("speech.wav", threshold=0.55)
for segment in output.segments:
    print(segment.start, segment.end)
```

See the [ASR and VAD matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/)
for checkpoints, inputs, outputs, and training boundaries.

## Optimize TTS

Start with eager inference, then benchmark one change at a time on the same
text, seed, warm-up count, and device.

```python
from voicehub import TTSOptimizationConfig

result = tts_model.optimize(
    TTSOptimizationConfig(
        attn_implementation="auto",
        kernel_backend="auto",
        compile="auto",
    )
)
print(result.manifest())
```

The optimization result records what was applied and what stayed on the
quality-preserving fallback. Do not publish speed or memory percentages from
configuration alone; measure them on the target hardware. Use the
[optimization guide](https://kadirnar.github.io/voicehub/guides/tts-optimization/),
[TTS model benchmarks](https://kadirnar.github.io/voicehub/guides/tts-model-benchmarks/),
and [current RTX 4090 speech results](https://kadirnar.github.io/voicehub/guides/rtx-4090-speech-benchmarks/)
for reproducible comparisons.

## Fine-tune

Every trainable integration advertises its exact objective and data contract.
Check support before loading weights:

```python
from voicehub import get_training_spec

spec = get_training_spec("dia")
print(spec.support.value, spec.family_name)
```

Then begin with a one-step smoke run. The
[training guide](https://kadirnar.github.io/voicehub/guides/training/) and
[training matrix](https://kadirnar.github.io/voicehub/models/training-support/)
show the required dataset fields, frozen components, checkpoint type, and
export path. The
[data guide](https://kadirnar.github.io/voicehub/guides/data-preparation/)
and [ASR/VAD data guide](https://kadirnar.github.io/voicehub/guides/speech-data/)
cover manifests and leakage-safe splits.

## Notebooks

The notebooks use a short, top-to-bottom workflow: install, configure, run,
and inspect.

| Notebook                    | GitHub                                                                                  | Colab                                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| TTS, ASR, and VAD inference | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/inference.ipynb)        | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/inference.ipynb)        |
| Data preparation            | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/data_preparation.ipynb) |
| Fine-tuning                 | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/training.ipynb)         | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/training.ipynb)         |
| Dia end-to-end workflow     | [View](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)     | [Run](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)     |

Read the [notebook guide](https://kadirnar.github.io/voicehub/guides/notebook/)
for expected hardware and opt-in execution flags.

## Documentation

- [Quickstart](https://kadirnar.github.io/voicehub/getting-started/quickstart/)
- [TTS inference](https://kadirnar.github.io/voicehub/guides/inference/)
- [Speech recognition](https://kadirnar.github.io/voicehub/guides/speech-recognition/)
- [Voice activity detection](https://kadirnar.github.io/voicehub/guides/voice-activity-detection/)
- [Model catalog](https://kadirnar.github.io/voicehub/models/)
- [Architecture](https://kadirnar.github.io/voicehub/concepts/architecture/)
- [Add a model](https://kadirnar.github.io/voicehub/project/adding-a-model/)
- [Add an optimization](https://kadirnar.github.io/voicehub/project/adding-an-optimization/)
- [API reference](https://kadirnar.github.io/voicehub/reference/api/)

## Development

```bash
git clone https://github.com/kadirnar/voicehub.git
cd voicehub
python -m pip install -e ".[test,training]"
python -m pytest
python scripts/check_distribution.py
```

`check_distribution.py` builds the wheel and source distribution, installs
the wheel, sdist, and editable checkout in separate environments, and checks
lazy import plus required package data. It skips PyTorch downloads by default;
pass `--with-dependencies` on a release machine for full dependency installs.

## License

VoiceHub is licensed under Apache-2.0. Vendored components retain their own
license notices in their package directories. Checkpoint licenses are
separate from source-code licenses and must be reviewed before use.
