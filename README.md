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
- **31 backends:** classic TTS, voice cloning, multilingual, dialogue,
  realtime, and prompted-style models.

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
