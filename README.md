<div align="center">
  <img width="420" alt="VoiceHub" src="assets/logo.png">
  <h1>VoiceHub</h1>
  <p>One lazy, discoverable Python interface for open text-to-speech models.</p>
</div>

## Why VoiceHub

- **One lifecycle:** every architecture uses config, lazy loading, generation, and a normalized output.
- **Fast imports:** ML frameworks and model weights are loaded only when the selected backend needs them.
- **Source included:** VoiceHub never delegates synthesis to a separately installed TTS package.
- **Small base install:** model extras contain only general runtime dependencies.
- **Actionable errors:** missing backends point to the exact installation extra.
- **16 backends:** classic TTS, voice cloning, multilingual, dialogue, and prompted-style models.

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
```

TTS implementations and their third-party licenses are included in the
VoiceHub package. Checkpoints remain separate and are downloaded lazily or
passed as local paths. See the
[model guide](https://github.com/kadirnar/VoiceHub/blob/main/docs/models.md).

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

## Supported models

| Model type        | Backend         | Notable capabilities                |
| ----------------- | --------------- | ----------------------------------- |
| `orpheustts`      | Orpheus-TTS     | Expressive speech                   |
| `dia`             | Dia             | Dialogue                            |
| `vui`             | Vui             | Text to speech                      |
| `chatterbox`      | Chatterbox      | Voice cloning                       |
| `kokoro`          | Kokoro          | Multilingual                        |
| `echo`            | Echo-TTS        | Voice cloning                       |
| `conversationtts` | ConversationTTS | Blocked: upstream source unlicensed |
| `llasa`           | LLaSA           | Multilingual synthesis and cloning  |
| `cosyvoice`       | CosyVoice 1/2/3 | Cloning, multilingual, streaming    |
| `f5tts`           | F5-TTS          | Voice cloning                       |
| `gptsovits`       | GPT-SoVITS      | Few-shot multilingual cloning       |
| `melotts`         | MeloTTS         | Fast multilingual synthesis         |
| `openvoice`       | OpenVoice V2    | Cross-lingual voice cloning         |
| `outetts`         | OuteTTS         | Speaker profiles, multiple runtimes |
| `parlertts`       | Parler-TTS      | Natural-language style control      |
| `styletts2`       | StyleTTS 2      | Style diffusion and voice cloning   |

Aliases such as `f5-tts`, `gpt-sovits`, `melo-tts`, `parler-tts`, and
`style-tts2` are accepted.

Discover models without importing their ML stacks:

```python
from voicehub import AutoInferenceModel

for spec in AutoInferenceModel.available_models():
    print(spec.model_type, spec.capabilities)
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
`file_path`, and backend metadata. Every registry class is named
`<Architecture>ForTextToSpeech`, every config is named
`<Architecture>Config`, and all models inherit the same `forward` and
`generate` signatures.

See the
[model guide](https://github.com/kadirnar/VoiceHub/blob/main/docs/models.md)
and [architecture guide](https://github.com/kadirnar/VoiceHub/blob/main/docs/architecture.md).

## Source policy

Upstream implementation snapshots live under `voicehub/models/*/source`;
shared neural components live under `voicehub/third_party`. Every vendored
snapshot records its exact revision and license. A static test rejects imports
of external TTS packages.

ConversationTTS is the sole exception: its public repository has no source
license, so redistribution is blocked until upstream grants one.
LLaSA's vendored XCodec2 component is separately licensed under
CC BY-NC 4.0 and is restricted to non-commercial use.

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
