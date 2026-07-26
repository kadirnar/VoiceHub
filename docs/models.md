# Model guide

VoiceHub ships the model implementation source inside its own wheel. Extras
install only general runtimes such as PyTorch, Transformers, phonemizers, and
audio I/O libraries. They never install a separate TTS implementation package.
Model checkpoints are still downloaded lazily or supplied as local paths.

```python
from voicehub import AutoModelForTextToSpeech

model = AutoModelForTextToSpeech.from_pretrained(
    "parler-tts/parler-tts-mini-v1",
    model_type="parlertts",
    device="cuda",
)
output = model(
    "VoiceHub uses one model lifecycle.",
    description="A warm speaker in a clean studio.",
    output_file="speech.wav",
)
print(output.sample_rate, output.file_path)
```

`AutoInferenceModel` remains as a compatible factory. Construction is lazy;
call `model.load()` once during service startup to warm the checkpoint.

## Source status

| Model type | Implementation used by VoiceHub | Source status |
|---|---|---|
| `orpheustts` | VoiceHub generation code + vendored SNAC | Included |
| `dia` | VoiceHub Dia source + vendored DAC | Included |
| `vui` | VoiceHub Vui and Fluac source | Included |
| `chatterbox` | VoiceHub Chatterbox + vendored S3Tokenizer/Perth | Included |
| `kokoro` | VoiceHub Kokoro source | Included |
| `echo` | VoiceHub Echo-TTS source | Included |
| `conversationtts` | Upstream has no source license | Blocked |
| `llasa` | LLaSA adapter + XCodec2 architecture source | Vendored (CC BY-NC 4.0) |
| `cosyvoice` | CosyVoice + Matcha-TTS | Vendored |
| `f5tts` | F5-TTS + BigVGAN + Vocos | Vendored |
| `gptsovits` | GPT-SoVITS inference tree | Vendored |
| `melotts` | MeloTTS | Vendored |
| `openvoice` | OpenVoice + shared MeloTTS | Vendored |
| `outetts` | OuteTTS + shared DAC | Vendored |
| `parlertts` | Parler-TTS + shared DAC | Vendored |
| `styletts2` | StyleTTS 2 + monotonic alignment source | Vendored |

Each vendored directory contains `SOURCE.json` and `THIRD_PARTY_LICENSE`.
`scripts/vendor_tts_sources.py` rebuilds deterministic snapshots from pinned
upstream revisions. Pretrained weights are not copied except for Perth's
small runtime watermark checkpoint.

LLaSA uses the vendored XCodec2 architecture instead of the `xcodec2` pip
package. The XCodec2 source and model are licensed CC BY-NC 4.0, so review
the non-commercial restriction before selecting this backend:

```python
model = AutoInferenceModel.from_pretrained("llasa")
output = model(
    "VoiceHub decodes LLaSA speech tokens locally.",
    speaker_audio_path="reference.wav",
    reference_text="Transcript of the reference.",
    output_file="llasa.wav",
)
```

## Voice cloning models

F5-TTS:

```python
from voicehub import AutoInferenceModel

model = AutoInferenceModel.from_pretrained("f5tts")
output = model(
    "Flow matching makes speech generation sound natural.",
    speaker_audio_path="reference.wav",
    reference_text="Transcript of the reference audio.",
    output_file="f5.wav",
)
```

CosyVoice:

```python
model = AutoInferenceModel.from_pretrained("cosyvoice")
output = model(
    "Voice cloning works from a short reference.",
    speaker_audio_path="reference.wav",
    prompt_text="This is the transcript of the reference.",
    output_file="cosyvoice.wav",
)
```

OpenVoice:

```python
model = AutoInferenceModel.from_pretrained(
    "openvoice",
    model_path="/models/openvoice/checkpoints_v2",
)
output = model(
    "This sentence uses the reference voice.",
    speaker_audio_path="reference.wav",
    language="EN_NEWEST",
    output_file="openvoice.wav",
)
```

GPT-SoVITS needs a local inference YAML whose checkpoint paths point to the
downloaded GPT, SoVITS, BERT, and CN-HuBERT assets:

```python
model = AutoInferenceModel.from_pretrained(
    "gpt-sovits",
    model_path="/models/gpt-sovits/tts_infer.yaml",
)
output = model(
    "Merhaba, bugün nasılsın?",
    text_language="tr",
    speaker_audio_path="reference.wav",
    prompt_language="tr",
    prompt_text="Referans kaydın metni.",
    output_file="gpt-sovits.wav",
)
```

StyleTTS 2 needs its main checkpoint plus the ASR, JDC, and PLBERT assets. Put
the official directory layout beside the checkpoint, or pass
`assets_directory` and `config_path`:

```python
model = AutoInferenceModel.from_pretrained(
    "style-tts2",
    model_path="/models/styletts2/epochs_2nd_00020.pth",
    assets_directory="/models/styletts2",
)
output = model(
    "Style diffusion controls timbre and prosody.",
    speaker_audio_path="reference.wav",
    output_file="styletts2.wav",
)
```

Only clone a voice with the speaker's permission and follow the selected
checkpoint's license and disclosure requirements.

## ConversationTTS

The public ConversationTTS repository had no license at audited revision
`b3851f7`. Unlicensed code cannot legally be redistributed, so VoiceHub does
not silently clone it or depend on an installable package. The registry entry
is present, but `load()` raises `SourceLicenseError` until upstream publishes
a compatible license.
