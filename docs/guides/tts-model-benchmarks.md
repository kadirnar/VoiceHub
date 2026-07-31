---
description: Measured RTX 4090 TTS optimization results and an honest evidence row for every VoiceHub TTS provider.
---

# TTS model benchmarks

Date: 2026-07-31. Hardware: NVIDIA GeForce RTX 4090, PyTorch 2.8.0,
CUDA 12.8.

Five of 34 registered TTS providers have real-checkpoint, end-to-end results
with at least 10 seconds of audio. The other rows are explicit about what was
and was not run.

## Read the labels

- `real-checkpoint-end-to-end`: published weights produced a long waveform.
- `tiny-graph`: a small synthetic graph ran; this is not a checkpoint benchmark.
- `static-plan`: lazy construction and optimization planning only.
- `blocked`, `not-run`, and `unsupported` never carry invented performance.
- `rejected` means a measured candidate failed the quality policy.

Every measured profile used seed 1234, a fresh process and compiler cache,
CUDA-synchronized timing, warm-ups, repeated runs, and a 10-second minimum.
Exact waveform equality was required for numerically invariant changes.
Non-exact regional kernels also needed unchanged duration/transcript metrics
and high waveform SNR. One fixture is not a general listening-quality claim.

## Results to use

- **VITS:** use eager FP32 plus `weight-norm-cache`. It was bit-exact and
  1.048x faster by mean latency (1.027x by median), but used 53.92 MiB more
  peak allocated memory. AUTO inference now stays eager; inference compile is
  quality-unsafe.
- **F5-TTS:** regional compile passed this fixture at 1.111x steady speed and
  4.01% less peak allocated memory. Cold inference rose from 6.33 to 130.16
  seconds. It needs 206 additional requests (207 total) to break even. Use it
  only for a long-lived, fixed workload and recheck quality on your data.
- **NeuTTS:** keep eager. SDPA was already native; regional compile was 1.342x
  faster but changed duration and doubled CER, so it is rejected.
- **Supertonic:** keep eager. Compile was bit-exact but only 0.589x as fast;
  reduced step counts raised WER.
- **Vui:** keep eager. Both compile policies changed sequence or transcript
  quality and now fail closed for inference.

### Real-checkpoint baselines

| Model | Audio | Mean latency | RTF | Peak allocated | Warm-up / runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vui | 21.827 s | 8.523 s | 0.39049 | 1,897.76 MiB | 2 / 5 |
| F5-TTS | 14.207 s | 5.836 s | 0.41080 | 1,600.56 MiB | 1 / 3 |
| NeuTTS | 12.440 s | 20.270 s | 1.62941 | 3,487.07 MiB | 1 / 3 |
| Supertonic | 28.369 s | 0.478 s | 0.01685 | 433.73 MiB | 3 / 10 |
| VITS | 27.312 s | 0.064 s | 0.00235 | 487.73 MiB | 5 / 15 |

### Optimization candidates

Peak allocated change uses `−` for less memory and `+` for more. “Paired”
means the candidate used the separately recorded baseline from the same run.

| Model / profile | Mean | Speed | Peak allocated change | Quality check | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Vui / compile dynamic | 5.266 s | 1.619x | +7.12% | WER 0.516 → 2.359; length changed | `rejected` |
| Vui / compile specialized | 4.316 s | 1.942x | −0.22% | WER 0.516 → 0.578; transcript changed | `rejected` |
| F5 / BF16 | 2.042 s | 2.859x | −49.66% | WER 0.485 → 1.455 | `rejected` |
| F5 / SDPA | 5.916 s | 0.986x | 0.00% | bit-exact | `no-benefit` |
| F5 / Triton | 5.814 s | 1.004x | 0.00% | same WER; 54.07 dB SNR | `no-benefit` |
| F5 / SDPA + Triton | 5.860 s | 0.996x | 0.00% | same WER; 54.07 dB SNR | `no-benefit` |
| F5 / NFE 16 | 3.088 s | 1.890x | −1.36% | WER 0.485 → 0.606 | `rejected` |
| F5 / DBCache 0.05 | 5.997 s | 0.973x | +4.01% | exact; 0% cache hits | `no-benefit` |
| F5 / DBCache 0.12 | 2.904 s | 2.010x | +4.01% | same WER; changed waveform lacks paired SNR | `rejected` |
| F5 / outer compile | — | — | — | terminated after more than 376 s | `blocked` |
| F5 / regional compile (paired) | 5.406 s | 1.111x | −4.01% | same WER; 51.31 dB SNR | `accepted` |
| NeuTTS / SDPA | 20.486 s | 0.989x | 0.00% | bit-exact | `no-benefit` |
| NeuTTS / regional compile (paired) | 15.225 s | 1.342x | +0.001% | duration −0.46 s; CER doubled | `rejected` |
| Supertonic / compile | 0.811 s | 0.589x | −0.19% | bit-exact | `no-benefit` |
| Supertonic / 3 steps | 0.357 s | 1.339x | +0.0002% | WER 0.031 → 0.063 | `rejected` |
| Supertonic / 2 steps | 0.297 s | 1.612x | +0.0002% | WER 0.031 → 0.266 | `rejected` |
| VITS / weight-norm cache (paired) | 0.068 s | 1.048x | +11.06% | bit-exact; replicated | `accepted` |
| VITS / Triton | 0.064 s | 0.999x | 0.00% | not bit-exact | `rejected` |
| VITS / compile | 0.070 s | 0.914x | +32.45% | not bit-exact | `rejected` |
| VITS / Triton + compile | 0.058 s | 1.099x | +32.45% | not bit-exact | `rejected` |
| VITS / FP16 cache | 0.055 s | 1.166x | −38.45% | duration and waveform changed | `rejected` |
| VITS / BF16 cache | 0.044 s | 1.464x | −38.45% | substantial waveform change | `rejected` |

## Full model evidence

All 34 rows passed the offline static-plan audit. “Evidence” below is the
highest tier reached, not a claim that every lower-level test was end to end.
Full checkpoint revisions and evidence paths are in the JSON artifact.

| Model type | Evidence | Real checkpoint | What was actually established |
| --- | --- | --- | --- |
| `orpheustts` | `tiny-graph` | `blocked` | Tiny round trip; gated weights and voice fixture unavailable |
| `dia` | `tiny-graph` | `not-run` | Tiny local generation/training/export |
| `vui` | `real-checkpoint-end-to-end` | measured | 21.827-second eager and compile runs |
| `chatterbox` | `tiny-graph` | `not-run` | Tiny T3 inference and component training |
| `kokoro` | `tiny-graph` | `not-run` | Tiny acoustic/frontend graph |
| `echo` | `static-plan` | `blocked` | No immutable default-checkpoint revision recorded |
| `conversationtts` | `tiny-graph` | `not-run` | Tiny cached decoder and training graph |
| `llasa` | `tiny-graph` | `not-run` | Tiny codec/generation/training graph |
| `cosyvoice` | `tiny-graph` | `blocked` | Missing speaker-embedding and prompt-token fixture |
| `f5tts` | `real-checkpoint-end-to-end` | measured | 14.207-second baseline and nine candidates |
| `gptsovits` | `tiny-graph` | `not-run` | Prepared-input native graph |
| `melotts` | `tiny-graph` | `not-run` | Tiny inference/export/training graph |
| `openvoice` | `tiny-graph` | `blocked` | Missing matched base and target-speaker audio |
| `outetts` | `tiny-graph` | `not-run` | Tiny language-model objective/export |
| `parlertts` | `tiny-graph` | `not-run` | Tiny acoustic-token and DAC graph |
| `styletts2` | `tiny-graph` | `blocked` | No default checkpoint; reviewed artifact and phonemes required |
| `mosstts` | `tiny-graph` | `not-run` | Tiny codec and buffered generation |
| `qwen3tts` | `tiny-graph` | `not-run` | Tiny talker/codec/cache/LoRA graph |
| `irodoritts` | `tiny-graph` | `not-run` | Tiny RF-DiT/duration graph |
| `zonos` | `tiny-graph` | `not-run` | Tiny sampling/codec/training graph |
| `zonos2` | `tiny-graph` | `not-run` | Tiny dense/MoE sampling graph |
| `voxcpm` | `tiny-graph` | `not-run` | Tiny generation/SFT/LoRA graph |
| `omnivoice` | `tiny-graph` | `not-run` | Tiny token generation and codec graph |
| `higgstts` | `tiny-graph` | `not-run` | Tiny constrained generation and codec graph |
| `xtts` | `tiny-graph` | `not-run` | Tiny token generation/conditioning graph |
| `vibevoice` | `tiny-graph` | `unsupported` | Low-level stages only; high-level waveform parity fails closed |
| `fishtts` | `tiny-graph` | `not-run` | Tiny semantic/cache and codec graph |
| `csm` | `tiny-graph` | `blocked` | Tiny graph; gated checkpoint access unavailable |
| `neutts` | `real-checkpoint-end-to-end` | measured | 12.440-second official Emily fixture |
| `supertonic` | `real-checkpoint-end-to-end` | measured | 28.369-second baseline and three candidates |
| `inflecttts` | `tiny-graph` | `not-run` | Tiny VITS/frontend graph |
| `bark` | `tiny-graph` | `not-run` | Tiny end-to-end generation and codec |
| `speecht5` | `tiny-graph` | `not-run` | Tiny acoustic/vocoder generation |
| `vits` | `real-checkpoint-end-to-end` | measured | 27.312-second baseline and six candidates |

## Reproduce a measured pair

```bash
python scripts/benchmark_tts_inference.py \
  --model-type vits \
  --model facebook/mms-tts-eng \
  --profiles baseline,weight-norm-cache \
  --config-kwargs '{"revision":"c71de0fe7204c83f1c10820a7d696d0b450048ba"}' \
  --seed 1234 \
  --minimum-audio-seconds 10 \
  --output result.json
```

Keep the eager `baseline`, pin every checkpoint, and compare one change at a
time. The machine-readable ledger is
`benchmarks/tts_optimization_rtx4090_2026-07-31.json`.
