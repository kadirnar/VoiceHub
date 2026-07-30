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
| Chatterbox | 2,566,656 | Native `chatterbox` inference and separate T3/S3Gen fine-tuning |
| Qwen3-TTS CustomVoice 1.7B | 2,487,040 | Added as `qwen3tts`; 0.6B and VoiceDesign share it |
| VoxCPM2 | 1,031,834 | Added as native `voxcpm` inference, full SFT, and LoRA |
| OmniVoice | 834,165 | Added as `omnivoice` |
| F5-TTS | 775,187 | Existing `f5tts` backend |
| VibeVoice Realtime 0.5B | 646,209 | Added as `vibevoice` |
| MOSS-TTS | 453,643 | VoiceHub-native four-variant inference and full semantic-model fine-tuning |
| Higgs TTS 3 4B | 370,423 | Audited; v2/v2.5 source added as `higgstts` |
| Fish Speech S2 Pro | 251,994 | Added as `fishtts` |
| Sesame CSM 1B | 244,792 | Added as `csm` |
| Supertonic 3 | 42,457 | Added as `supertonic` |
| NeuTTS 2E | 4,812 | Added as `neutts` |
| OuteTTS 1.0 | — | Replaced provider runtimes with native Llama/Qwen, byte-BPE, V3 prompting, and DAC |
| Inflect Micro/Nano v2 | 298 / 252 | Added as `inflecttts` |

The audit also covered Irodori-TTS, Zonos v0.1, ZONOS2, and the complete MOSS
and Qwen3 checkpoint families requested for this release.

## Training and data audit

Training paths were re-audited on 2026-07-30 across all 34 registered
TTS model types. Registry presence was checked against the actual loaded
training graph, objective, data boundary, optimizer topology, and export
semantics.

| Registry support | Count | Practical meaning |
| --- | ---: | --- |
| `native` | 14 | A backend-native differentiable objective exists; raw-data preparation may still be external |
| `preprocessed` | 17 | A verified objective accepts source-shaped tensors or tokens |
| `custom` | 3 | Source-specific graph or orchestration is required |
| `inference-only` | 0 | Every registered TTS model type has a verified fine-tuning route; unsupported checkpoint variants still fail closed |

The principal findings were:

- full end-to-end VITS training was not present; the Transformers VITS route
  is an explicit waveform-reconstruction experiment, while MeloTTS,
  GPT-SoVITS, and StyleTTS2 require source-owned trainable graphs;
- diffusion/flow objectives existed model by model without one strict shared
  noise/timestep/target API;
- codec/LLM TTS was the strongest family, but codebook layout, codec
  preprocessing, and raw-data support varied by model;
- broad family schemas could incorrectly imply that a model accepted raw
  text/audio when it required prepared tensors; and
- the trainer accumulated all adversarial phase losses before one optimizer
  boundary, so it could not represent discriminator-step followed by a fresh
  generator forward.

The shared layer now addresses those cross-cutting boundaries with
model-specific dataset readiness, manifest loading and content fingerprints,
strict multi-codebook/diffusion/VITS objective primitives, exact masked losses,
and opt-in sequential named-optimizer phase steps. Model support remains
fail-closed: the shared math does not relabel an inference-only or incomplete
source integration as trainable. The current per-model result is maintained in
the [TTS training support matrix](../models/training-support.md).

## Source and license decisions

| Family | Upstream source | Source/license result |
|---|---|---|
| Chatterbox | <https://github.com/resemble-ai/chatterbox> | MIT; VoiceHub-native T3, S3Gen, S3Tokenizer, voice encoder, audio frontend, and Perth runtime. Pinned checkpoint inventory: VE 16 tensors / 1,423,618 values; T3 292 / 532,405,248; S3Gen 2,489 / 264,041,793; total 2,797 tensors / 797,870,659 values. |
| ConversationTTS | <https://github.com/Audio-Foundation-Models/ConversationTTS> | CC BY-NC 4.0; model, inference, tokenizer, and MimiCodec runtime source included |
| Vui | <https://github.com/fluxions-ai/vui> | MIT; VoiceHub-native pinned Vui 100M, byte tokenizer, and frozen Fluac graph. Restricted official PyTorch import and strict standalone model-plus-codec Safetensors export/reload are both covered; the reconstructed preprocessed objective does not claim parity with an unpublished author training loop. |
| MOSS-TTS | <https://github.com/OpenMOSS/MOSS-TTS> | Apache-2.0; VoiceHub-native Delay, Local, Local v1.5, and Realtime graphs plus MOSS Audio Tokenizer v1/v2, pinned to source revisions `58b20a0` and `8c50ac4`. Audited semantic inventories are Delay 463 tensors / 8,489,841,664 values, Local 556 / 3,060,606,464, Local v1.5 438 / 4,550,403,584, and Realtime 403 / 2,331,940,864. Codec v1 is 1,600 F32 tensors / 1,774,566,400 values; codec v2 is 2,094 / 2,123,701,248. All official repositories are immutable-revision Safetensors with strict header and shape validation. Raw-audio and pre-encoded full semantic-model fine-tuning freeze the codec. Realtime generation is buffered; no incremental streaming claim or accuracy claim is made. |
| Qwen3-TTS | <https://github.com/QwenLM/Qwen3-TTS> | Apache-2.0 |
| Irodori-TTS | <https://github.com/Aratako/Irodori-TTS> | MIT; VoiceHub-native RF-DiT, duration predictor, unigram/byte-fallback tokenizer, frozen Semantic-DACVAE, raw-audio flow/duration objectives, and strict v2/v3/VoiceDesign Safetensors lifecycle. SilentCipher numerical parity is not claimed. |
| Zonos / ZONOS2 | <https://github.com/Zyphra/Zonos>, <https://github.com/Zyphra/ZONOS2> | Apache-2.0 / MIT; DAC source included |
| VoxCPM2 | <https://github.com/OpenBMB/VoxCPM> | Apache-2.0; source and checkpoint revisions pinned, 577 model tensors and 312 AudioVAE V2 tensors audited |
| OmniVoice | <https://github.com/k2-fsa/OmniVoice> | Apache-2.0; VoiceHub-native 313-tensor / 612,577,288-parameter graph plus pinned 527-tensor frozen Higgs Audio v2 codec. Raw-audio or preencoded-code full fine-tuning uses the published weighted masked cross-entropy. |
| Higgs Audio v2 | <https://github.com/boson-ai/higgs-audio> | Apache-2.0 source; custom-license checkpoint. VoiceHub owns the audited 397-tensor / 5,771,283,456-parameter decoder and 527-tensor / 201,400,553-parameter frozen codec, full SFT objective, strict loading, and export. |
| XTTS | <https://github.com/coqui-ai/TTS> | MPL-2.0 source; XTTS weights use CPML. VoiceHub owns the audited 963-tensor / 466,900,598-parameter native graph, exact tokenizer assets, GPT objective, strict Safetensors runtime, and explicit restricted legacy conversion. DVAE target extraction remains offline. |
| VibeVoice | <https://github.com/microsoft/VibeVoice> | VoiceHub-native family pinned to MIT source `94da20d`. The ASR checkpoint revision `f22241c` contains 901 BF16 tensors / 8,330,325,888 values; the non-streaming 1.5B TTS revision `c00898d` contains 1,204 / 2,704,021,987; realtime 0.5B revision `6bce5f0` contains 605 / 1,017,626,724. Exact graphs, codecs, byte-BPE processing, diffusion/DPM, strict Safetensors loading, ASR fine-tuning, non-streaming TTS fine-tuning, and portable export are native. Realtime unified fine-tuning and high-level cached-prompt synthesis fail closed pending independent parity; no accuracy claim is made. |
| Fish Speech | <https://github.com/fishaudio/fish-speech> | VoiceHub-native S2 DualAR + ModifiedDAC pinned to source `e5e2926` and checkpoint `1de9996`; Fish Audio Research License, non-commercial derivative, notice, and attribution restrictions |
| Sesame CSM | <https://github.com/SesameAILabs/csm> | VoiceHub-native graph pinned to source `daed31e`; gated `sesame/csm-1b` Safetensors rev `c92a71e` (187 tensors, 1,552,791,552 parameters, Apache-2.0) plus frozen native Mimi rev `2bfc9ae` (318 tensors, 96,151,393 parameters, CC-BY-4.0); SilentCipher remains an explicit postprocessor boundary |
| NeuTTS | <https://github.com/neuphonic/neutts> | Native Qwen/Llama LM, tokenizer, and Apache-2.0 NeuCodec; Air checkpoint is Apache-2.0, while other variants use the NeuTTS Open License with its USD 5M commercial threshold |
| OuteTTS 1.0 | <https://github.com/edwko/OuteTTS> | Apache-2.0 source pinned to `f5eac6e`; VoiceHub-native Llama/Qwen causal LM, exact V3 prompt/token protocol, and IBM DAC graph. The default 1B checkpoint is CC-BY-NC-SA-4.0; the 0.6B checkpoint is Apache-2.0. |
| GPT-SoVITS V1/V2/Pro | <https://github.com/RVC-Boss/GPT-SoVITS> | MIT; native V1, V2, V2Pro, and V2ProPlus S1 semantic and classic-S2 VITS/GAN graphs. All 12 released component inventories are pinned exactly; Pro uses required prepared 20,480-D ERes2NetV2 conditioning and the seven-period discriminator. Staged variant-aware Safetensors export reloads for inference. V3/V4 flow-matching and LoRA layouts fail closed. |
| OpenVoice V2 | <https://github.com/myshell-ai/OpenVoice> | MIT; VoiceHub-native checkpoint-exact tone-color converter pinned to source `74a1d147` and checkpoint `f36e7edf`. The audited release contains 486 F32 tensors / 32,792,226 values with checkpoint SHA-256 `9652c27…ab9e`. The official pickle crosses a one-time digest-checked `weights_only=True` boundary; normal inference, reconstructed paired-waveform training, and export use Safetensors. Upstream publishes no converter training loop, discriminator, dataset, or loss, so VoiceHub explicitly records no upstream-training-parity or quality-improvement claim. |
| StyleTTS 2 | <https://github.com/yl4579/StyleTTS2> | MIT; VoiceHub-native PL-BERT, diffusion, HiFi-GAN/iSTFTNet, strict checkpoint adapter, and preprocessed generator/MPD/MSD objectives pinned to `5cedc71` |
| Supertonic | <https://github.com/supertone-inc/supertonic> | MIT runtime source; OpenRAIL-M weights |
| Inflect v2 | <https://huggingface.co/owensong/Inflect-Micro-v2> | Apache-2.0 model-specific source included |
| Granite Speech 4.1 | <https://huggingface.co/ibm-granite/granite-speech-4.1-2b> | Apache-2.0; VoiceHub-native Conformer, Q-Former, Granite decoder, byte-BPE tokenizer, HTK frontend, and source-compatible projector/LoRA fine-tuning. Pinned checkpoint inventory: 954 tensors / 2,313,207,148 parameters / header fingerprint `8889064…74001`. |
| Parakeet TDT 0.6B v3 | <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3> | VoiceHub-native FastConformer/TDT source port audited against immutable Transformers and NeMo revisions; Apache-2.0 architecture references and CC-BY-4.0 checkpoint. Pinned inventory: 723 tensors / 627,057,286 learned parameters / 627,057,310 state values / header fingerprint `f861cd8…e6b`. |
| Nemotron 3.5 ASR | <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b> | VoiceHub-native prompt-conditioned FastConformer/RNN-T port audited against immutable Apache-2.0 Transformers source; OpenMDW-1.1 checkpoint. Pinned single-file inventory: 655 tensors / 637,997,088 parameters / header fingerprint `c50ff50…517`. Greedy decoding and the cache-aware graph are verified; the shared public session remains buffered. |
| Cohere Transcribe 03-2026 | <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026> | VoiceHub-native 48-layer FastConformer, eight-layer autoregressive cross-attention decoder, 128-bin log-mel frontend, byte-fallback BPE tokenizer, prompt processor, and quiet-boundary long-form path audited against immutable Apache-2.0 source. Pinned gated Safetensors inventory: 2,152 tensors / 2,065,804,096 persistent values / 2,047,822,080 learned parameters / header fingerprint `06a76e1…292`. Full-model fine-tuning and portable export are supported; decoding is greedy and no WER or accuracy-improvement claim is made. |
| SeamlessM4T-v2 Large S2T | <https://huggingface.co/facebook/seamless-m4t-v2-large> | VoiceHub-native stacked Kaldi-style frontend, 24-layer relative-key Conformer, adapter, 24-layer decoder, SentencePiece BPE, and 98-language prompt table audited against immutable Apache-2.0 Transformers source `a08ace4`. The pinned CC-BY-NC-4.0 two-shard checkpoint contains 2,232 tensors / 2,309,249,669 values; the executable S2T projection persists 1,429 tensors / 1,501,842,240 values with header fingerprint `2f12727…bef`. Full-model teacher-forced fine-tuning, gradient checkpointing, and portable S2T export are verified. Recognition is greedy-only, and no WER or accuracy-improvement claim is made. |
| Google MedASR | <https://huggingface.co/google/medasr> | VoiceHub-native LASR CTC port audited against immutable Apache-2.0 source and Google's published full-model fine-tuning notebook. The gated checkpoint remains subject to the Health AI Developer Foundations terms. Pinned inventory: 368 tensors / 105,282,833 persistent elements / header fingerprint `c302fca…090`. |

**Built with Fish Audio**

Commercial-use restrictions are recorded as metadata, not used as an
exclusion rule. `conversationtts`, `fishtts`, `llasa`, and the default
`outetts` checkpoint remain discoverable and report `commercial_use=False`.

## Audited but not registered

Gepard 1.0 and NVIDIA MagpieTTS currently require NVIDIA NeMo TTS/codec
runtime source outside their small inference repositories. Voxtral 4B TTS and
Higgs Audio v3 currently document SGLang-Omni/vLLM serving paths. Registering
any of these through those installable TTS/omni runtimes would violate
VoiceHub's source-only rule, so they are recorded here but are not presented
as working local backends. They can be added once their complete executable
model and codec source is vendored and tested under the same contract.
