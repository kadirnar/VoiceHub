---
description: Reproducible RTX 4090 TTS, ASR, VAD, and fine-tuning measurements with explicit quality checks.
---

# RTX 4090 speech benchmarks

Measured on 2026-07-31 with an NVIDIA GeForce RTX 4090, driver 570.169,
PyTorch 2.8.0+cu128, CUDA 12.8, cuDNN 9.10.2, Python 3.11.10, and an
AMD EPYC 7K62 CPU. GPU clocks were not locked.

These are checkpoint- and machine-specific results, not portable thresholds.
Lower latency and real-time factor (RTF) are better.

## Coverage

| Task | Registered | Registry/contract result | Released checkpoints run end to end |
| --- | ---: | --- | --- |
| TTS | 34 | 34 lazy loads and 34 baseline/optimized plans passed | MMS-VITS |
| ASR | 23 | 23 lazy contracts passed | Moonshine tiny, Wav2Vec2 base, and Whisper tiny through five adapters |
| VAD | 11 | 11 lazy contracts passed | Silero, Sherpa/Silero, WebRTC, Auditok |

The complete test suite exercises the remaining native graphs with small
deterministic weights. It would be misleading to claim that every public
checkpoint was downloaded: several are gated, license-restricted, or many
gigabytes. Registry/graph coverage and released-checkpoint coverage are
reported separately throughout this page.

## Method

Canonical values are stored in
`benchmarks/tts_vits_rtx4090_2026-07-31.json` and
`benchmarks/asr_vad_rtx4090_2026-07-31.json`. Tables below round those values
for readability.

- TTS profiles used the same text, seed, generation arguments, and isolated
  process. Compiler caches were fresh for each profile.
- ASR and VAD used the same 12.485-second LibriSpeech sample.
- CUDA timings were synchronized. GPU memory is peak allocated tensor memory.
- CPU memory is process peak RSS, not model-only memory.
- Cold/setup time and steady-state time are reported separately.
- An optimization is called quality-preserving only when its compared output
  met the stated equality check. Timing alone is never treated as quality
  evidence.

## TTS

The released `facebook/mms-tts-eng` checkpoint has 36,284,592 parameters.
It was resolved at revision
`c71de0fe7204c83f1c10820a7d696d0b450048ba`; the model file SHA-256 was
`69cf8b651c1493f1801dfd2311c298d694a38357bc9a1e41f410491ea6f0e1be`.
This 64-word input produced 27.312 seconds of 16 kHz audio:

> VoiceHub makes speech model inference easier to understand and reproduce.
> This sample is intentionally long enough to produce more than ten seconds
> of clear spoken audio. It measures the complete text tokenizer, acoustic
> model, normalizing flow, and neural vocoder pipeline on one graphics
> processor. The same sentence and random seed are used for every benchmark
> so that baseline and optimized runs remain directly comparable.

### Accepted VITS optimization

The primary paired run used three warm-ups and 30 measurements. An independent
repeat used two warm-ups and 30 measurements.

| FP32 profile | Median | RTF | Peak allocated | Change | Waveform |
| --- | ---: | ---: | ---: | --- | --- |
| Unoptimized baseline | 64.119 ms | 0.00235 | 487.73 MiB | Reference | — |
| Weight-normalization cache | 62.407 ms | 0.00228 | 541.65 MiB | 1.027x; 2.67% less latency; +53.92 MiB | Bit-exact |

The cache is the quality-preserving winner for repeated VITS requests when
the extra 53.92 MiB is acceptable. A second independent 30-run measurement
showed a 4.52% median reduction, so the observed improvement range was
2.67–4.52%. Load and cold-request timings varied and are not used for this
choice.

### TTS candidates not accepted by the strict gate

The following clean matrix used five warm-ups and 15 measurements per
isolated process. The baseline had no weight cache. “Not accepted” means the
waveform changed; it is not a claim about perceptual MOS.

| Profile | Mean / median | Mean / median speed | Peak allocated | Output check | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| FP32 baseline | 64.183 / 64.154 ms | Reference | 487.73 MiB | Bit-exact repeats | Reference |
| Triton | 64.217 / 64.054 ms | 0.999x / 1.002x | 487.73 MiB | Max error 0.0602; SNR 45.07 dB | No material speedup; non-exact |
| Compile | 70.196 / 54.192 ms | 0.914x / 1.184x | 645.99 MiB | Max error 0.1147; SNR 31.95 dB | Variable, non-exact, and +158.26 MiB |
| Triton + compile | 58.392 / 51.008 ms | 1.099x / 1.258x | 645.99 MiB | Max error 0.1076; SNR 32.14 dB | Faster median, but non-exact |
| FP16 + cache | 55.025 / 42.265 ms | 1.166x / 1.518x | 300.17 MiB | Duration changed by 16 ms | Saves 38.45% memory, but fails strict gate |
| BF16 + cache | 43.844 / 43.797 ms | 1.464x / 1.465x | 300.19 MiB | Max error 1.2328; SNR -2.60 dB | Saves 38.45% memory, but fails strict gate |

Compile and Triton + compile had 29.16% and 25.60% latency coefficients of
variation, so their medians should not be read without their means. Compile
also required 26.61 seconds on its first request with a fresh cache. None of
these candidates is enabled as a no-quality-change default.

## ASR

All ASR profiles produced finite, deterministic transcripts. The reference
contains 32 normalized words; the released checkpoints each measured 3.125%
WER on this single sample. One utterance is a smoke test, not a corpus-level
accuracy result.

### Precision

| Checkpoint | Profile | Mean / median | Peak allocated | Memory change | Transcript |
| --- | --- | ---: | ---: | ---: | --- |
| Moonshine tiny | FP32 | 301.35 / 301.52 ms | 134.72 MiB | Reference | Reference |
| Moonshine tiny | FP16 | 316.43 / 314.15 ms | 80.42 MiB | -40.30% | Exact |
| Moonshine tiny | BF16 | 329.96 / 332.57 ms | 84.52 MiB | -37.26% | Exact |
| Whisper tiny | FP32 | 173.59 / 167.29 ms | 321.25 MiB | Reference | Reference |
| Whisper tiny | FP16 | 200.17 / 200.55 ms | 269.04 MiB | -16.25% | Exact |
| Whisper tiny | BF16 | 201.23 / 199.21 ms | 269.04 MiB | -16.25% | Punctuation changed; WER unchanged |

Reduced precision saved memory but was slower for these two sequence-to-
sequence checkpoints. It is a memory option only after dataset-level
accuracy validation.

Wav2Vec2 base reduced peak allocation from 569.23 MiB to 316.15 MiB
(44.46%). Its run was strongly bimodal for every dtype (75–79% latency
coefficient of variation). BF16's mean was 1.140x faster than FP32, but that
single high-variance run is not a robust precision winner. The script records
mean, median, range, standard deviation, and a high-variability flag to prevent
a median-threshold artifact from being reported as a speedup.

### Moonshine compile

After five discarded stabilization requests, 15 measurements gave:

| Profile | Mean / median | Peak allocated | Output |
| --- | ---: | ---: | --- |
| FP32 eager | 301.354 / 301.518 ms | 134.72 MiB | Reference |
| FP32 compile | 176.222 / 175.544 ms | 130.22 MiB | Identical transcript and WER |

Compile was 1.710x faster by mean (41.52% less latency) and used 3.33% less
steady peak memory. The first compile/inference took 75.23 seconds and five
stabilization requests took 75.95 seconds in total. The setup cost breaks even
after roughly 608 repeated utterances, so compile is useful for a long-lived,
high-volume Moonshine service—not a short job.

## VAD

The CPU thread count mattered more than compilation for the learned VAD
models on this 96-thread host.

| Runtime | Baseline mean / median | Candidate mean / median | Mean speed | Output |
| --- | ---: | ---: | ---: | --- |
| Silero | 1,506.67 / 1,500.51 ms, 48 threads | 224.70 / 224.65 ms, one thread | 6.705x | Same segments and scores |
| Sherpa/Silero | 1,640.20 / 1,605.46 ms, 48 threads | 281.12 / 280.25 ms, one thread | 5.835x | Same segments |
| WebRTC | 482.18 / 477.40 ms, scalar PCM | 411.99 / 408.29 ms, vectorized PCM | 1.170x | Identical boundaries |
| Auditok | 4.009 / 0.892 ms, 48 threads | 1.358 / 1.357 ms, one thread | 2.953x mean | Deterministic segments |

The WebRTC change is implemented in VoiceHub and preserves the original
clamping, non-finite handling, and Python tie-rounding behavior.

Auditok's 48-thread mean contains an 83 ms outlier and its coefficient of
variation was 384%; its sub-millisecond median does not support a thread-count
winner.

Silero compile with one CPU thread averaged 191.53 ms (190.32 ms median),
another 1.173x over one-thread eager, with identical boundaries and a maximum
score difference of one float32 epsilon. It raised process peak RSS from
520.78 to 856.77 MiB and needed 9.33 seconds on the first request. Prefer the
exact one-thread setting; compile is an opt-in service experiment.

## Fine-tuning

All 34 TTS providers have an explicit training and dataset profile:
14 `native`, 17 `preprocessed`, and 3 `custom`; none silently falls back to
an invented generic objective.

The checked-in VITS smoke utility ran a real one-step generator warm-start on
`facebook/mms-tts-eng`:

- one aligned self-synthesized example, 5.904 seconds;
- loss `7.002128` to `6.994179`;
- finite gradient norm and a changed model-state fingerprint;
- native Safetensors export and bit-exact state reload; and
- 27.456 seconds of reloaded validation audio.

MMS metadata does not publish the original adversarial acoustic recipe, so
this is correctly labeled a preprocessed generator warm-start, not a full
adversarial reproduction. Separate tiny-graph tests cover backward,
optimizer, export, and reload paths across VITS, Qwen3-TTS LoRA, F5-TTS,
Dia, ConversationTTS, and FishTTS.

## Reproduce

```bash
# Offline provider audits
python scripts/benchmark_tts_inference.py --audit-registry
python scripts/benchmark_asr_vad.py --audit-registry

# Real TTS profiles; exits nonzero if the waveform gate fails
python scripts/benchmark_tts_inference.py \
  --model-type vits \
  --model facebook/mms-tts-eng \
  --profiles baseline,weight-norm-cache,float16-cache,bfloat16-cache,compile \
  --audio-dir runs/tts-audio \
  --output runs/tts-benchmark.json

# Real ASR or VAD; input must be at least 10 seconds
python scripts/benchmark_asr_vad.py \
  --task asr \
  --audio speech.wav \
  --reference "REFERENCE TRANSCRIPT" \
  --output runs/asr-benchmark.json

# One real checkpoint training/export/reload smoke
python scripts/smoke_finetune_vits.py --output-dir runs/vits-smoke
```

Use the JSON reports rather than copying these percentages to different
hardware. Keep only profiles that pass the quality metric appropriate to the
target dataset and deployment.
