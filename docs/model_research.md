# Current TTS model audit

Audit date: 2026-07-26.

The Hugging Face page was queried with its actual API ordering,
`sort=trendingScore`, and separately with `sort=downloads`. Counts are a
point-in-time signal, not a quality ranking. Finetunes, language-only
derivatives, GGUF/ONNX conversions, and duplicate parameter sizes map to an
existing architecture instead of creating duplicate VoiceHub backends.

## Trending and downloaded architectures

| Architecture/checkpoint | Downloads at audit | VoiceHub decision |
|---|---:|---|
| Kokoro 82M | 10,417,369 | Existing `kokoro` backend |
| XTTS v2 | 9,344,273 | Added as `xtts` |
| Chatterbox | 2,566,656 | Existing `chatterbox` backend |
| Qwen3-TTS CustomVoice 1.7B | 2,487,040 | Added as `qwen3tts`; 0.6B and VoiceDesign share it |
| VoxCPM2 | 1,031,834 | Added as `voxcpm` |
| OmniVoice | 834,165 | Added as `omnivoice` |
| F5-TTS | 775,187 | Existing `f5tts` backend |
| VibeVoice Realtime 0.5B | 646,209 | Added as `vibevoice` |
| MOSS-TTS | 453,643 | Added as `mosstts` |
| Higgs TTS 3 4B | 370,423 | Audited; v2/v2.5 source added as `higgstts` |
| Fish Speech S2 Pro | 251,994 | Added as `fishtts` |
| Sesame CSM 1B | 244,792 | Added as `csm` |
| Supertonic 3 | 42,457 | Added as `supertonic` |
| NeuTTS 2E | 4,812 | Added as `neutts` |
| Inflect Micro/Nano v2 | 298 / 252 | Added as `inflecttts` |

The audit also covered Irodori-TTS, Zonos v0.1, ZONOS2, and the complete MOSS
and Qwen3 checkpoint families requested for this release.

## Source and license decisions

| Family | Upstream source | Source/license result |
|---|---|---|
| ConversationTTS | <https://github.com/Audio-Foundation-Models/ConversationTTS> | CC BY-NC 4.0; model, inference, tokenizer, and MimiCodec runtime source included |
| MOSS-TTS | <https://github.com/OpenMOSS/MOSS-TTS> | Apache-2.0; MOSS Audio Tokenizer source included |
| Qwen3-TTS | <https://github.com/QwenLM/Qwen3-TTS> | Apache-2.0 |
| Irodori-TTS | <https://github.com/Aratako/Irodori-TTS> | MIT; DACVAE and SilentCipher source included |
| Zonos / ZONOS2 | <https://github.com/Zyphra/Zonos>, <https://github.com/Zyphra/ZONOS2> | Apache-2.0 / MIT; DAC source included |
| VoxCPM | <https://github.com/OpenBMB/VoxCPM> | Apache-2.0 |
| OmniVoice | <https://github.com/k2-fsa/OmniVoice> | Apache-2.0 |
| Higgs Audio v2 | <https://github.com/boson-ai/higgs-audio> | Apache-2.0 source |
| XTTS | <https://github.com/coqui-ai/TTS> | MPL-2.0 source; XTTS weights use CPML |
| VibeVoice | <https://github.com/microsoft/VibeVoice> | MIT source; checkpoint card carries responsible-use limitations |
| Fish Speech | <https://github.com/fishaudio/fish-speech> | Fish Audio Research License; non-commercial and attribution restrictions |
| Sesame CSM | <https://github.com/SesameAILabs/csm> | Apache-2.0; Moshi/Mimi and SilentCipher source included |
| NeuTTS | <https://github.com/neuphonic/neutts> | Custom NeuTTS model license; Apache-2.0 NeuCodec and MIT Perth source included |
| Supertonic | <https://github.com/supertone-inc/supertonic> | MIT runtime source; OpenRAIL-M weights |
| Inflect v2 | <https://huggingface.co/owensong/Inflect-Micro-v2> | Apache-2.0 model-specific source included |

**Built with Fish Audio**

Commercial-use restrictions are recorded as metadata, not used as an
exclusion rule. `conversationtts`, `fishtts`, and `llasa` remain discoverable
and report `commercial_use=False`.

## Audited but not registered

Gepard 1.0 and NVIDIA MagpieTTS currently require NVIDIA NeMo TTS/codec
runtime source outside their small inference repositories. Voxtral 4B TTS and
Higgs Audio v3 currently document SGLang-Omni/vLLM serving paths. Registering
any of these through those installable TTS/omni runtimes would violate
VoiceHub's source-only rule, so they are recorded here but are not presented
as working local backends. They can be added once their complete executable
model and codec source is vendored and tested under the same contract.
