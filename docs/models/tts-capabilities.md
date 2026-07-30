# TTS capabilities and adaptation boundaries

This matrix describes the current VoiceHub provider contract. **Diffusion**
includes flow-matching models. Task codes are **C** voice cloning from a
reference/profile/embedding, **V** text-guided voice design, **D** dialogue,
and **S** explicit style, emotion, or description control. A language entry of
**checkpoint-defined/not enumerated** means VoiceHub does not claim one
exhaustive set across the provider's accepted checkpoints.

| Model type | Family | Verified languages | C/V/D/S | Fine-tuning and LoRA boundary |
| --- | --- | --- | --- | --- |
| `orpheustts` | LLM | checkpoint-defined/not enumerated | S | Raw text/audio or SNAC codes train the LM; SNAC stays frozen. LoRA: no. |
| `dia` | LLM | checkpoint-defined/not enumerated | D | Raw text/audio trains the `Dia-1.6B-0626` encoder/decoder; native DAC prepares targets. The legacy checkpoint is unsupported. LoRA: no. |
| `vui` | LLM | en | C (codec prompt) | Prepared text IDs and Fluac codes train a reconstructed LM objective; Fluac stays frozen. LoRA: no. |
| `chatterbox` | Hybrid | en | C, S | Raw or prepared data trains T3 LM and S3Gen flow in separate jobs, not one joint job. LoRA: T3 LM only. |
| `kokoro` | Acoustic | en-US, en-GB, es, fr, hi, it, pt-BR, ja, zh | S | Prepared partial PL-BERT, duration/prosody, and decoder objectives; not the raw-data/full author recipe. LoRA: no. |
| `echo` | Diffusion | checkpoint-defined/not enumerated | C | Prepared Fish-codec latents train an experimental reconstructed DiT-flow objective; codec stays frozen. LoRA: no. |
| `conversationtts` | LLM | checkpoint-defined/not enumerated | C, D | Raw data or Mimi codes train the language/depth model; Mimi stays frozen. LoRA: no. |
| `llasa` | LLM | checkpoint-defined/not enumerated | C | Raw data or XCodec2 codes train only the LM; XCodec2 stays frozen. LoRA: no. |
| `cosyvoice` | Hybrid | zh (+18 dialects), en, ja, ko, de, es, fr, it, ru | C, S | Raw/prepared data trains LM, flow, HiFT generator, or discriminator in separate jobs; S3Tokenizer stays frozen and CAMPPlus embeddings remain explicit. LoRA: no. |
| `f5tts` | Diffusion | en, zh | C | Waveform/mel plus prepared text IDs train the full DiT flow; Vocos stays frozen. Chinese needs explicit pinyin or a native normalizer. LoRA: no. |
| `gptsovits` | Hybrid | zh, en, ja; V2+ also ko, yue | C | Prepared staged S1, S2-generator, and S2-discriminator jobs cover V1/V2/V2Pro/V2ProPlus. V3, V4, and LoRA are unsupported. |
| `melotts` | VITS | en, fr, ja, es, zh, ko | — | Prepared linguistic/BERT/spectrogram/audio tensors train full VITS/GAN phases; raw multilingual preparation and author-resumable state are not provided. LoRA: no. |
| `openvoice` | VITS | en, es, fr, zh, ja, ko | C | Opt-in reconstructed paired-waveform converter training only; no released upstream recipe or quality-parity claim. LoRA: no. |
| `outetts` | LLM | checkpoint-defined/not enumerated | C (profile) | Prepared V3 profiles or token labels train the full LM; DAC stays frozen and raw audio is unsupported. LoRA: no. |
| `parlertts` | LLM | checkpoint-defined/not enumerated | S | Raw description/text/audio or DAC codes train the decoder and T5 by default; T5 may be frozen and DAC always is. LoRA: no. |
| `styletts2` | Diffusion | en-US | C, S | Prepared reconstructed generator/GAN phases; no raw G2P/alignment, WavLM objective, or author-resume parity. LoRA: no. |
| `mosstts` | LLM | checkpoint-defined/not enumerated | C | Raw audio or RVQ codes train the complete semantic graph for each supported variant; its codec stays frozen. LoRA: no. |
| `qwen3tts` | LLM | zh, en, ja, ko, de, fr, ru, pt, es, it | C, V, S | Base checkpoints only: prepared 16-codebook SFT trains talker/residual predictor fully by default, or native LoRA adapts their attention/MLP projections. Speaker encoder stays frozen; CustomVoice/VoiceDesign are not training starts. |
| `irodoritts` | Diffusion | checkpoint-defined/not enumerated | C, V, S | Raw audio or latents train RF-DiT and optional duration prediction; Semantic-DACVAE stays frozen. LoRA: no. |
| `zonos` | LLM | checkpoint-defined/not enumerated | C, S | Raw/codes with explicit or injected phonemes train a reconstructed dense-Transformer LM objective; DAC stays frozen. Mamba-2 is unsupported. LoRA: no. |
| `zonos2` | LLM | checkpoint-defined/not enumerated | C, S | Raw audio or codes train the dense/MoE acoustic LM under a reconstructed objective; codec is outside the trainable graph. LoRA: no. |
| `voxcpm` | Hybrid | checkpoint-defined/not enumerated | C, V, S | Raw audio or latents train the MiniCPM/flow graph, or native LoRA targets LM, DiT, and optional projections. AudioVAE stays frozen. |
| `omnivoice` | LLM | checkpoint-defined/not enumerated | C, V | Raw audio or codes train the complete masked-token model; Higgs Audio codec stays frozen. LoRA: no. |
| `higgstts` | LLM | checkpoint-defined/not enumerated | C, S | Raw audio or codes train the dual-FFN decoder; HuBERT/DAC tokenizer stays frozen and VoiceHub owns the unpublished optimizer schedule. LoRA: no. |
| `xtts` | LLM | en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-CN, hu, ko, ja, hi | C | Raw audio or codes train only the GPT; DVAE, speaker encoder, and HiFi-GAN stay frozen. No vocoder/GAN phase or LoRA. |
| `vibevoice` | Hybrid | checkpoint-defined/not enumerated | —* | Prepared latents train the non-streaming 1.5B LM/connectors/diffusion head; codecs stay frozen. Realtime/default-checkpoint FT and high-level TTS fail closed. LoRA: no. |
| `fishtts` | LLM | checkpoint-defined/not enumerated | C | Prepared tokens train the slow and fast semantic model only; ModifiedDAC stays frozen. Provider LoRA is rejected; derivatives are non-commercial. |
| `csm` | LLM | checkpoint-defined/not enumerated | C, D | Raw conversation/audio or Mimi codes train the CSM backbone/depth decoder; Mimi stays frozen. LoRA: no. |
| `neutts` | LLM | checkpoint-defined/not enumerated | C, S | NeuTTS-Air only: raw audio or codes train the LM while NeuCodec stays frozen. Nano, multilingual Nano, and default 2E FT fail closed. LoRA: no. |
| `supertonic` | Diffusion | ar, bg, cs, da, de, el, en, es, et, fi, fr, hi, hr, hu, id, it, ja, ko, lt, lv, na, nl, pl, pt, ro, ru, sk, sl, sv, tr, uk, vi | S | Prepared style, duration, and latent targets train reconstructed graph losses; no raw-data or complete author recipe. LoRA: no. |
| `inflecttts` | VITS | en-US | — | Prepared phonemes/spectrogram/audio train a full-VITS warm start with newly initialized posterior/discriminator; it is not author-resumable. LoRA: no. |
| `bark` | LLM | de, en, es, fr, hi, it, ja, ko, pl, pt, ru, tr, zh | S | Prepared tokens train semantic, coarse, or fine stages separately; Encodec stays frozen and no joint raw-audio recipe is provided. LoRA: no. |
| `speecht5` | Acoustic | en | C (x-vector) | Raw text/audio trains the complete spectrogram model; HiFi-GAN stays frozen. LoRA: no. |
| `vits` | VITS | checkpoint-defined/not enumerated | — | Full raw-waveform adversarial FT requires an explicit checkpoint acoustic config; the generator-only prepared route is a partial warm start. LoRA: no. |

\* VibeVoice exposes verified low-level realtime stages, but its unified
high-level waveform-generation contract is intentionally unavailable.

See [TTS training support](training-support.md) for the detailed data,
checkpoint, objective, and export contracts.
