<div align="center">
  <img width="420" alt="VoiceHub" src="assets/logo.png">
  <h1>VoiceHub</h1>
  <p>One lazy, task-aware Python interface for open TTS, ASR, and VAD models.</p>
</div>

## Why VoiceHub

- **One lifecycle:** every architecture uses config, lazy loading, task-specific inference, and a normalized output.
- **One trainer:** shared arguments, callbacks, evaluation, and resumable checkpoints.
- **Fast imports:** ML frameworks and model weights are loaded only when the selected backend needs them.
- **Task-aware discovery:** TTS, speech-recognition, and voice-activity providers share one registry without cross-task factory mistakes.
- **Source policy:** TTS source stays integrated; ASR and VAD provider runtimes remain optional and lazy.
- **Small base install:** model extras contain only general runtime dependencies.
- **Actionable errors:** missing backends point to the exact installation extra.
- **47 registered integrations:** 31 TTS backends, 9 ASR providers, and 7 VAD
  providers, each representing a runtime or checkpoint family rather than one
  fixed weight file.

## Install

VoiceHub requires Python 3.10 or newer.

```bash
python -m pip install voicehub
```

Install only the backend you plan to use:

```bash
python -m pip install "voicehub[parlertts]"
python -m pip install "voicehub[f5tts]"
python -m pip install "voicehub[melotts]"
python -m pip install "voicehub[asr-transformers]"
python -m pip install "voicehub[vad-silero]"
```

Install PyTorch training support independently of any inference backend:

```bash
python -m pip install "voicehub[training]"
```

TTS implementations and their third-party licenses are included in the
VoiceHub package. Checkpoints remain separate and are downloaded lazily or
passed as local paths. See the
[model catalog](https://kadirnar.github.io/voicehub/models/).
ASR and VAD integrations wrap optional provider runtimes selected by their
extras; see the
[speech-input matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/).

## Quick start

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)

output = model(
    text="VoiceHub keeps the interface small and the model choice open.",
    description="A warm, clear speaker talks at a relaxed pace.",
    output_file="output.wav",
)
print(output.sample_rate, output.file_path)
```

Construction is intentionally cheap. The checkpoint is downloaded and loaded
on the first synthesis call. Use `model.load()` when you want to warm a model
up before serving traffic.

## Documentation

The [VoiceHub documentation site](https://kadirnar.github.io/voicehub/)
provides searchable, task-oriented guides:

- [TTS inference](https://kadirnar.github.io/voicehub/guides/inference/)
- [Speech recognition](https://kadirnar.github.io/voicehub/guides/speech-recognition/)
- [Voice activity detection](https://kadirnar.github.io/voicehub/guides/voice-activity-detection/)
- [ASR and VAD data](https://kadirnar.github.io/voicehub/guides/speech-data/)
- [Data preparation](https://kadirnar.github.io/voicehub/guides/data-preparation/)
- [Training](https://kadirnar.github.io/voicehub/guides/training/)
- [End-to-end notebook](https://kadirnar.github.io/voicehub/guides/notebook/)
- [TTS training support](https://kadirnar.github.io/voicehub/models/training-support/)
- [ASR and VAD support](https://kadirnar.github.io/voicehub/models/asr-vad-support/)

The runnable
[Jupyter notebook](https://github.com/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb)
runs the complete workflow and
[opens directly in Colab](https://colab.research.google.com/github/kadirnar/voicehub/blob/main/notebooks/tts_workflow.ipynb).

## Speech recognition

```python
from voicehub import AutoModelForSpeechRecognition

model = AutoModelForSpeechRecognition.from_pretrained(
    "openai/whisper-small",
    model_type="asr_transformers",
    device="cuda",
)
output = model.transcribe(
    "meeting.wav",
    language="en",
    return_timestamps="word",
)
print(output.text)
```

The Transformers provider covers compatible CTC, speech
sequence-to-sequence, RNN-T, and TDT checkpoints. Separate providers expose
faster-whisper, WhisperX, OpenAI Whisper, NeMo, SpeechBrain, FunASR, ESPnet,
and WeNet while returning the same `ASROutput`.

## Voice activity detection

```python
from voicehub import AutoModelForVoiceActivityDetection

model = AutoModelForVoiceActivityDetection.from_pretrained(
    "silero_vad",
    model_type="vad_silero",
)
output = model.detect(
    "meeting.wav",
    threshold=0.55,
    min_speech_duration_ms=250,
)

for segment in output.segments:
    print(segment.start, segment.end)
```

VoiceHub integrates Transformers audio/frame classification, Silero, WebRTC,
pyannote, SpeechBrain, NeMo, and FunASR FSMN VAD behind normalized
`VADOutput` speech regions.

## Supported models

This table lists TTS integrations. See the
[ASR/VAD provider matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/)
for speech-input families, optional extras, outputs, and fine-tuning
boundaries.

| Model type        | Backend         | Notable capabilities                |
| ----------------- | --------------- | ----------------------------------- |
| `orpheustts`      | Orpheus-TTS     | Expressive speech                   |
| `dia`             | Dia             | Dialogue                            |
| `vui`             | Vui             | Text to speech                      |
| `chatterbox`      | Chatterbox      | Voice cloning                       |
| `kokoro`          | Kokoro          | Multilingual                        |
| `echo`            | Echo-TTS        | Voice cloning                       |
| `conversationtts` | ConversationTTS | CC BY-NC multilingual conversation  |
| `llasa`           | LLaSA           | Multilingual synthesis and cloning  |
| `cosyvoice`       | CosyVoice 1/2/3 | Cloning, multilingual, streaming    |
| `f5tts`           | F5-TTS          | Voice cloning                       |
| `gptsovits`       | GPT-SoVITS      | Few-shot multilingual cloning       |
| `melotts`         | MeloTTS         | Fast multilingual synthesis         |
| `openvoice`       | OpenVoice V2    | Cross-lingual voice cloning         |
| `outetts`         | OuteTTS         | Speaker profiles, multiple runtimes |
| `parlertts`       | Parler-TTS      | Natural-language style control      |
| `styletts2`       | StyleTTS 2      | Style diffusion and voice cloning   |
| `mosstts`         | MOSS-TTS        | Delay, Local, v1.5, Realtime        |
| `qwen3tts`        | Qwen3-TTS       | Clone, CustomVoice, VoiceDesign     |
| `irodoritts`      | Irodori-TTS     | Reference and caption conditioning  |
| `zonos`           | Zonos 1         | Multilingual voice cloning          |
| `zonos2`          | ZONOS2          | Batched MoE synthesis and cloning   |
| `voxcpm`          | VoxCPM 1/2      | Streaming and voice cloning         |
| `omnivoice`       | OmniVoice       | Multilingual cloning and design     |
| `higgstts`        | Higgs Audio     | Expressive long-form generation     |
| `xtts`            | XTTS v2         | Multilingual voice cloning          |
| `vibevoice`       | VibeVoice       | Realtime cached-voice generation    |
| `fishtts`         | Fish Speech S2  | Multilingual cloning                |
| `csm`             | Sesame CSM      | Conversational speaker context      |
| `neutts`          | NeuTTS          | Air, Nano, multilingual, 2E         |
| `supertonic`      | Supertonic 3    | Fast multilingual ONNX inference    |
| `inflecttts`      | Inflect v2      | Compact local synthesis             |

Aliases such as `f5-tts`, `gpt-sovits`, `melo-tts`, `parler-tts`, and
`style-tts2`, `moss-tts`, `qwen3-tts`, and `higgs-tts` are accepted.

Discover models without importing their ML stacks:

```python
from voicehub import AutoInferenceModel

for spec in AutoInferenceModel.available_models():
    print(spec.model_type, spec.capabilities, spec.components)
    if spec.license:
        print(spec.license.license_id, spec.license.commercial_use)
```

For task-aware discovery:

```python
from voicehub import list_model_specs

for spec in list_model_specs(task="asr"):
    print(spec.model_type, spec.architecture, spec.install_extra)
```

## Common API

The Transformers-style API loads architecture-specific configuration:

```python
from voicehub import (
    AutoConfig,
    AutoModelForTextToSpeech,
    AutoProcessor,
    TTSGenerationConfig,
)

config = AutoConfig.for_model("f5tts", ode_method="euler")
processor = AutoProcessor.from_config(config)
model = AutoModelForTextToSpeech.from_pretrained(
    "F5TTS_v1_Base",
    config=config,
    device="cuda",
)
inputs = processor(
    "VoiceHub has one public API contract.",
    speaker_audio_path="reference.wav",
    reference_text="Reference transcript.",
)
output = model.generate(
    **inputs,
    generation_config=TTSGenerationConfig(speed=1.0, seed=42),
)
```

Every synthesis call returns `TTSOutput`, containing `audio`, `sample_rate`,
`file_path`, and backend metadata. Every TTS registry class is named
`<Architecture>ForTextToSpeech`, every config is named
`<Architecture>Config`, and all TTS wrappers inherit the same `forward` and
`generate` signatures.

ASR uses `AutoModelForSpeechRecognition` and returns `ASROutput`; VAD uses
`AutoModelForVoiceActivityDetection` and returns `VADOutput`. Both accept file,
array, tensor, mapping, or `AudioInput` audio. See the
[ASR guide](https://kadirnar.github.io/voicehub/guides/speech-recognition/)
and [VAD guide](https://kadirnar.github.io/voicehub/guides/voice-activity-detection/).

See the
[model catalog](https://kadirnar.github.io/voicehub/models/)
and [architecture guide](https://kadirnar.github.io/voicehub/concepts/architecture/).

## Training

`Trainer` and `TrainingArguments` follow the Transformers training vocabulary
without adding Transformers or PyTorch to the base installation. Every
registered backend has an audited `ModelTrainingSpec`. Directly runnable
profiles use the built-in family adapters, while source-native recipes that
need architecture-specific orchestration require a specialized adapter:

```python
from voicehub import AutoModelForTextToSpeech, Trainer, TrainingArguments

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
    lazy_load=True,
)
model.validate_training_support()

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="runs/my-tts-model",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        load_best_model_at_end=True,
    ),
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
trainer.train()
```

This example assumes that dataset items already contain Parler's
backend-shaped training tensors. The generic trainer does not silently turn
raw text/audio into architecture-specific codec, alignment, flow, or
adversarial targets.

Use `trainer.train(resume_from_checkpoint=True)` only when `output_dir`
already contains a complete VoiceHub checkpoint.

VoiceHub adapters normalize TTS phases to `TTSTrainingOutput` and ASR/VAD
phases to `SpeechTrainingOutput`; both put `loss` first. Native forwards may
instead return a mapping with `loss` or a tuple with loss first.
Architecture-specific objectives can be connected through `compute_loss_func`
for a true single-phase model. Declarative recipes cover causal codec LMs,
sequence-to-sequence models, flow/diffusion models, acoustic regression, VITS,
composite/GAN systems, CTC, speech sequence-to-sequence, RNN-T, TDT, and audio
or frame classification. Composite source modules receive separately routed,
named optimizer and scheduler states. Variable-length TTS batches use
`DataCollatorForTTSTraining`; ASR and VAD batches use
`DataCollatorForAudioTraining`.

Support is deliberately variant-aware. Differentiable native and preprocessed
profiles run directly; custom profiles fail until their specialized adapter is
registered; ONNX/GGUF, fused, quantized, or inference-pruned variants are
rejected before loading when they cannot preserve a verified gradient path.

Speech-input training follows the same rule. Compatible unquantized
Transformers ASR and VAD graphs use the task-neutral trainer families. NeMo,
SpeechBrain, FunASR ASR and FSMN VAD, ESPnet, WeNet, and pyannote currently
keep their upstream-custom recipe ownership; faster-whisper, WhisperX, OpenAI
Whisper, Silero, and WebRTC integrations are inference-only. The exact
boundary is in the
[ASR/VAD support matrix](https://kadirnar.github.io/voicehub/models/asr-vad-support/#fine-tuning-boundaries).

The common loop includes gradient accumulation and clipping, AdamW schedules,
mixed precision, callbacks, evaluation/prediction, best-model selection,
checkpoint rotation, and atomic resumable model/optimizer/scheduler/RNG state.

See the
[trainer guide](https://kadirnar.github.io/voicehub/concepts/trainer/)
and
[training model matrix](https://kadirnar.github.io/voicehub/models/training-support/)
for objective families, native upstream recipes, and extension points.

## Source policy

Upstream implementation snapshots live under `voicehub/models/*/source`.
Reusable code is organized by role under `voicehub/components`: codecs,
vocoders, watermarking, and neural blocks. `ModelSpec.components` connects
each backend to the shared components it uses. Every snapshot records its
exact revision and license, and a static test rejects imports of external TTS
packages.

Non-commercial licenses are supported and exposed through
`ModelSpec.license`; they are not treated as an integration failure.
ConversationTTS and LLaSA/XCodec2 are CC BY-NC 4.0 and Fish Speech uses the
Fish Audio Research License. These backends are included, but they must not
be used commercially without the required additional permission.

Some newly included source families also carry restrictions outside
VoiceHub's Apache-2.0 license: Fish Speech uses the Fish Audio Research
License, NeuTTS and XTTS checkpoints use their respective custom licenses,
and VibeVoice checkpoints have responsible-use conditions. Review
[`SOURCE.json`](voicehub/models) and the selected checkpoint card before use.
Fish Speech attribution must include “Built with Fish Audio”.

Build and dependency metadata has one source of truth in `pyproject.toml`.
The repository no longer executes `setup.py` during installation or release.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest
pre-commit run --from-ref origin/main --to-ref HEAD
```

## License

VoiceHub is released under Apache-2.0. Model code, checkpoints, voices, and
generated audio can have additional licenses or usage conditions; review the
selected backend before distribution.
