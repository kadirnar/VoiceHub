# Model catalog

VoiceHub ships the model implementation source inside its own wheel. Extras
install only general runtimes such as PyTorch, Transformers, phonemizers, and
audio I/O libraries. They never install a separate TTS implementation package.
Model checkpoints are still downloaded lazily or supplied as local paths.

## Choose a model

Start from the capability you need, then inspect the exact checkpoint and
conditioning contract before loading weights. The optional extra matches the
registry key.

| Model type | Good fit | Install extra |
| --- | --- | --- |
| `orpheustts` | Expressive speech | `voicehub[orpheustts]` |
| `dia` | Multi-speaker dialogue | `voicehub[dia]` |
| `vui` | Compact text to speech | `voicehub[vui]` |
| `chatterbox` | Voice cloning | `voicehub[chatterbox]` |
| `kokoro` | Lightweight multilingual speech | `voicehub[kokoro]` |
| `echo` | Reference-conditioned cloning | `voicehub[echo]` |
| `conversationtts` | Multilingual conversation | `voicehub[conversationtts]` |
| `llasa` | Multilingual codec-LM cloning | `voicehub[llasa]` |
| `cosyvoice` | Cloning, multilingual speech, streaming | `voicehub[cosyvoice]` |
| `f5tts` | Flow-matching voice cloning | `voicehub[f5tts]` |
| `gptsovits` | Few-shot multilingual cloning | `voicehub[gptsovits]` |
| `melotts` | Fast multilingual synthesis | `voicehub[melotts]` |
| `openvoice` | Cross-lingual voice transfer | `voicehub[openvoice]` |
| `outetts` | Speaker profiles and multiple runtimes | `voicehub[outetts]` |
| `parlertts` | Natural-language style control | `voicehub[parlertts]` |
| `styletts2` | Style diffusion and voice cloning | `voicehub[styletts2]` |
| `mosstts` | Delay, local, and realtime generation | `voicehub[mosstts]` |
| `qwen3tts` | Custom voices, design, and cloning | `voicehub[qwen3tts]` |
| `irodoritts` | Reference and caption conditioning | `voicehub[irodoritts]` |
| `zonos` | Multilingual voice cloning | `voicehub[zonos]` |
| `zonos2` | Batched mixture-of-experts synthesis | `voicehub[zonos2]` |
| `voxcpm` | Streaming voice cloning | `voicehub[voxcpm]` |
| `omnivoice` | Multilingual cloning and voice design | `voicehub[omnivoice]` |
| `higgstts` | Expressive long-form generation | `voicehub[higgstts]` |
| `xtts` | Multilingual voice cloning | `voicehub[xtts]` |
| `vibevoice` | Realtime cached-voice generation | `voicehub[vibevoice]` |
| `fishtts` | Multilingual semantic-token cloning | `voicehub[fishtts]` |
| `csm` | Conversational speaker context | `voicehub[csm]` |
| `neutts` | Compact local and multilingual variants | `voicehub[neutts]` |
| `supertonic` | Fast multilingual ONNX inference | `voicehub[supertonic]` |
| `inflecttts` | Compact local synthesis | `voicehub[inflecttts]` |

Training capability is checkpoint-aware. Check the
[training support matrix](training-support.md) before selecting an artifact or
designing a dataset.

```python
from voicehub import AutoInferenceModel, AutoModelForTextToSpeech

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
| `conversationtts` | ConversationTTS + bundled MimiCodec | Vendored (CC BY-NC 4.0) |
| `llasa` | LLaSA adapter + XCodec2 architecture source | Vendored (CC BY-NC 4.0) |
| `cosyvoice` | CosyVoice + Matcha-TTS | Vendored |
| `f5tts` | F5-TTS + BigVGAN + Vocos | Vendored |
| `gptsovits` | GPT-SoVITS inference tree | Vendored |
| `melotts` | MeloTTS | Vendored |
| `openvoice` | OpenVoice + shared MeloTTS | Vendored |
| `outetts` | OuteTTS + shared DAC | Vendored |
| `parlertts` | Parler-TTS + shared DAC | Vendored |
| `styletts2` | StyleTTS 2 + monotonic alignment source | Vendored |
| `mosstts` | MOSS-TTS + MOSS Audio Tokenizer | Vendored |
| `qwen3tts` | Qwen3-TTS | Vendored |
| `irodoritts` | Irodori-TTS + DACVAE + SilentCipher | Vendored |
| `zonos` | Zonos v0.1 | Vendored |
| `zonos2` | ZONOS2 + shared DAC | Vendored |
| `voxcpm` | VoxCPM and VoxCPM2 | Vendored |
| `omnivoice` | OmniVoice | Vendored |
| `higgstts` | Higgs Audio v2/v2.5 | Vendored |
| `xtts` | Coqui XTTS v2 architecture | Vendored (MPL-2.0) |
| `vibevoice` | VibeVoice realtime | Vendored |
| `fishtts` | Fish Speech S2/OpenAudio + DAC | Vendored (research license) |
| `csm` | Sesame CSM + Moshi/Mimi + SilentCipher | Vendored |
| `neutts` | NeuTTS + NeuCodec + Perth | Vendored (custom model license) |
| `supertonic` | Supertonic 3 ONNX runtime | Vendored |
| `inflecttts` | Inflect Micro/Nano v2 | Vendored |

Each vendored directory contains `SOURCE.json` and `THIRD_PARTY_LICENSE`.
`scripts/vendor_tts_sources.py` rebuilds deterministic snapshots from pinned
upstream revisions. Pretrained weights are not copied except for Perth's
small runtime watermark checkpoint.

## Current-generation families

The current backends keep checkpoint variants behind one architecture key:

| Backend | Supported family variants |
|---|---|
| `mosstts` | Delay, Local, Local v1.5, Realtime, MOSS-TTS and MOSS-TTS-Nano checkpoints |
| `qwen3tts` | 0.6B/1.7B Base, CustomVoice, VoiceDesign, and voice cloning |
| `irodoritts` | v2, v3, and VoiceDesign-compatible checkpoints |
| `zonos` / `zonos2` | Zonos v0.1 Transformer/Hybrid and ZONOS2 |
| `voxcpm` | VoxCPM and VoxCPM2 |
| `higgstts` | Higgs Audio v2/v2.5 source architecture |
| `neutts` | Air, Nano, multilingual Nano, and 2E backbones |
| `inflecttts` | Inflect Micro v2 and Nano v2 |

Model weights, cached voice prompts, preset speaker embeddings, and ONNX
graphs are not embedded in the wheel. They are resolved from a checkpoint
repository or accepted as local paths.

MOSS example:

```python
model = AutoInferenceModel.from_pretrained(
    "moss-tts",
    model_path="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    variant="local_v1_5",
    device="cuda",
)
output = model(
    "Source-integrated speech generation.",
    speaker_audio_path="reference.wav",
    output_file="moss.wav",
)
```

Qwen3-TTS voice design:

```python
model = AutoInferenceModel.from_pretrained(
    "qwen3-tts",
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device="cuda",
)
output = model(
    "Merhaba, VoiceHub'a hoş geldiniz.",
    mode="voice_design",
    language="Turkish",
    instruct="A calm, confident adult speaker.",
)
```

## Training-safe runtime selection

Fine-tuning starts through `load_for_training()`, which can select a different
construction path from inference and reject an incompatible backend:

- Dia fine-tuning selects the official Transformers
  `DiaForConditionalGeneration` checkpoint and processor. The original Nari
  `Dia-1.6B` runtime remains inference-only.
- Sesame CSM fine-tuning selects the official Transformers CSM graph, creates
  labels with `CsmProcessor`, and keeps the Mimi codec frozen.
- OmniVoice keeps `torch_dtype="float16"` as its inference default but uses
  `training_torch_dtype="float32"` by default for training. Its registered
  collator schema treats codebook tensors as codebook-first and time-last.
- VoxCPM passes `training=True` through the vendored pipeline, disables the
  denoiser and inference optimization, and applies the source freezing policy,
  including the AudioVAE.
- OuteTTS generic fine-tuning requires the HF backend and rejects
  `load_in_4bit`, `load_in_8bit`, and non-`None` `quantization_config` options.
- NeuTTS rejects a GGUF backbone for fine-tuning. An ONNX codec decoder does
  not block preprocessed HF-backbone training because the codec is not an
  optimized component and can remain frozen.
- Qwen3-TTS fine-tuning starts from
  `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, exposed as
  `model.training_default_model_name_or_path`. CustomVoice and VoiceDesign
  checkpoints are inference/export targets for this recipe.
- MOSS-TTS Delay, Local, and Realtime variants expose native loss paths. Local
  v1.5 uses the integrated channel-wise supervised fine-tuning objective for
  `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`.

Safetensors are weight containers, not proof of a trainable graph. When a
model repository also publishes GGUF or another serving artifact, select the
compatible unquantized PyTorch/Transformers checkpoint. Exact training resume
uses a VoiceHub checkpoint because optimizer, scheduler, RNG, sampler, and
recipe state are not present in a safetensors weight export.

The exact model-by-model boundary is maintained in the
[training model matrix](training-support.md).

The [current model research](../project/model-audit.md) records the dated Hugging
Face audit, download/trending signals, upstream source, licensing, and
source-only inclusion decisions.

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

Fish Speech is distributed under the Fish Audio Research License. Its
required attribution is: **Built with Fish Audio**.

## ConversationTTS

ConversationTTS revision `b3851f7` declares its source, checkpoints, datasets,
and evaluation tools under CC BY-NC 4.0. VoiceHub therefore includes its
executable model, inference, text-tokenizer, and MimiCodec runtime source. The
license does not permit commercial use:

```python
model = AutoInferenceModel.from_pretrained(
    "conversationtts",
    model_path="AudioFoundation/SpeechFoundation",
    device="cuda",
)
output = model(
    "A source-integrated conversational model.",
    speaker_audio_path="reference.wav",
    reference_text="Transcript of the reference speaker.",
    output_file="conversation.wav",
)
```

The main checkpoint and Mimi tokenizer weights remain external Hub artifacts.
