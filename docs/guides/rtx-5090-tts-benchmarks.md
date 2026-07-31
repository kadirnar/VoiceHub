---
description: Measured RTX 5090 inference and tiny-training results for VoiceHub VITS, LLM, and diffusion TTS paths.
---

# RTX 5090 TTS benchmarks

Measured on 2026-07-30 with one NVIDIA GeForce RTX 5090 (32,607 MiB,
compute capability 12.0), driver 570.195.03, PyTorch 2.8.0+cu128, CUDA
12.8, cuDNN 9.10.2, Python 3.12.3, and two Intel Xeon Gold 6530 CPUs.
GPU clocks were not locked. Unless noted, results use batch size 1,
`torch.inference_mode()`, a CUDA synchronization around each sample, one
discarded warm-up, and the median of three or more warm measurements.

This page is a dated lab snapshot, not a packaged benchmark suite. It records
the exact implementation decisions and immutable public checkpoint revisions,
but the reference audio and one-off timing harness are not shipped. Use the
tables to compare the tested techniques on this machine, not as a portable
regression threshold.

Real-time factor (RTF) is wall time divided by generated audio duration;
lower is better. These are latency results, not MOS claims. Approximate
sampler and cache changes require listening, WER/CER, and speaker-similarity
evaluation on the intended checkpoint and language before production use.

## Coverage

VoiceHub has 34 registered TTS providers. All 34 were exercised by the
repository inference, checkpoint, optimization-resolution, and output-contract
tests. Public-weight GPU runs below add one released end-to-end checkpoint for
each requested execution family. The remaining provider tests use their exact
native graph with tiny deterministic weights because downloading every
multi-gigabyte, gated, or license-restricted checkpoint is not a reproducible
CI requirement.

| Execution family | Registered coverage | Released GPU checkpoint |
| --- | ---: | --- |
| VITS/GAN | 5 shared VITS optimization surfaces | `facebook/mms-tts-eng@c71de0f` |
| LLM-TTS | All registered causal/codec-LM provider contracts | `neuphonic/neutts-2e@412aaab` plus `neuphonic/neucodec@30c1fdd` |
| Diffusion/flow | 9 registered sampler surfaces | `SWivid/F5-TTS@84e5a41` plus `charactr/vocos-mel-24khz@0feb3fd` |
| Other acoustic TTS | Kokoro and SpeechT5 provider contracts | Tiny exact-graph tests |

The model-by-model languages, cloning/design/dialogue controls, and
fine-tuning boundaries are in the
[TTS capability matrix](../models/tts-capabilities.md).

## VITS

### Released checkpoint

Fixed text: “VoiceHub benchmarks text to speech inference on the RTX 5090.”
With seed 2026, the MMS checkpoint generated 3.888 seconds of finite 16 kHz
audio. The local-checkpoint load took 1.161 seconds.

| Configuration | Precision | Cold/setup | Warm median | RTF | Peak allocated |
| --- | --- | ---: | ---: | ---: | ---: |
| Eager baseline | FP16 | 1,222.7 ms first request | 24.856 ms | 0.00639 | 105.57 MiB |
| Weight-normalization inference cache | FP16 | 5.57 ms setup | 22.775 ms | 0.00586 | 132.22 MiB |

The paired 60-sample medians show a 1.09x speedup (8.37% less latency).
Outputs were bit-exact. The non-persistent 26.648 MiB cache leaves all
checkpoint keys unchanged and invalidates after training, parameter mutation,
or device/dtype conversion.

### Optimization experiments

These paired microbenchmarks isolate the active neural boundary. “Regional
compile” compiles stable text-encoder, reverse-flow, and decoder tensor
regions while leaving validation, random generators, and duration-dependent
request logic eager.

| Graph | Change | Baseline | Optimized | Speedup | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Native VITS, tiny BF16 | Regional `torch.compile`, dynamic default | 4.792 ms | 3.990 ms | 1.20x | Tiny-graph result only; rejected as an inference default after the real-checkpoint quality gate |
| Native VITS, tiny BF16 | Regional max-autotune, no CUDA graphs | 4.792 ms | 3.965 ms | 1.21x | Tiny-graph result only; 29.11 s first call and about 90 s process autotuning |
| MeloTTS released-dimension synthetic graph, BF16 | Reversible eval weight-normalization cache | 19.107 ms | 18.309 ms | 1.04x | Keep; maximum BF16 delta `9.77e-4` |
| OpenVoice released topology | Reversible eval weight-normalization cache | 10.994 ms | 9.601 ms | 1.15x | Keep; bit-exact |
| InflectTTS Nano topology | Reversible eval weight-normalization cache | 8.029 ms | 7.113 ms | 1.13x | Keep; bit-exact |
| GPT-SoVITS S2 topology | Reversible eval weight-normalization cache | 13.578 ms | 12.625 ms | 1.08x | Keep; bit-exact |

Compiling the complete request wrapper was rejected: the first call took
91.09 seconds and steady state improved only 5.1%. Fixed-shape CUDA graphs
remain an explicit bucketed-serving option; predicted durations make them an
unsafe general default. A later long-audio real-checkpoint run also changed
the fixed-seed waveform, so current VITS inference plans stay eager and retain
compilation only for training.

## LLM-TTS

### Released checkpoint

The NeuTTS run uses `neuphonic/neutts-2e@412aaab`, the released
`neuphonic/neucodec@30c1fdd`, the same 4.848-second reference WAV and
transcript for each request, BF16 for the language model, and FP32 for the
codec. The uncommitted lab fixture was 155,180 bytes with SHA-256
`42e4e9366557e58ec50d868b0534158374edfb29c2f3f66412e70b645aff1e7d`.

| Configuration | Generated tokens | Warm median | Audio duration | RTF | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native cache-aware generation | 64 | 928.99 ms | 1.280 s | 0.7258 | 3,230.05 MiB |

The real run found and fixed two interoperability bugs: logical Safetensors
symlink names are now preserved during checkpoint classification, and the
NeuCodec frontend now applies the released shortest-stream frame alignment
when its two encoders round an utterance to adjacent frame counts.
Cold model load was 9.76–12.38 seconds and one-time encoding of the
4.848-second reference took 0.69–0.79 seconds. The reported inference median
uses four warm-ups, seven samples, an eight-thread host pool, and excludes
reference encoding.

### Fused GQA attention

A 152.1M-parameter BF16 Qwen-style causal model measured the shared attention
path used by NeuTTS, OuteTTS, CosyVoice, and other native causal-LM adapters.
The optimized path uses PyTorch scaled-dot-product attention without
materializing repeated grouped-query key/value heads. Padding masks and
attention-weight requests retain the explicit FP32 reference path.

| Operation | Shape | Manual attention | Fused SDPA/GQA | Speedup |
| --- | --- | ---: | ---: | ---: |
| Prefill | 64 tokens | 13.7089 ms | 10.1977 ms | 1.34x |
| Prefill | 256 tokens | 14.7134 ms | 10.9695 ms | 1.34x |
| Prefill | 1,024 tokens | 16.5621 ms | 12.0643 ms | 1.37x |
| Cached decode | 64-token context | 7.7785 ms/token | 5.8589 ms/token | 1.33x |
| Cached decode | 256-token context | 7.7869 ms/token | 5.9485 ms/token | 1.31x |
| Cached decode | 1,024-token context | 8.0652 ms/token | 5.8697 ms/token | 1.37x |
| Generate | 256 prompt + 64 output | 417.283 ms | 379.816 ms | 1.10x |

At 1,024-token prefill the measured transient allocation fell from
172.63 MiB to 50.00 MiB (3.45x lower). CPU reference maximum absolute error
was `1.86e-8`; BF16-logit cosine similarity was `0.99985`.

A Python preallocated KV cache was rejected for the current eager decoder:
512 update steps took 284.15 ms versus 204.05 ms for the existing dynamic
cache. The asymptotic allocation pattern is better, but Python indexed writes
lost on this workload. Static-cache compilation remains a future
whole-generation optimization, not a default regression.

## Diffusion TTS

### Released F5-TTS checkpoint

Fixed target: “VoiceHub tests diffusion text to speech on the RTX 5090.”
The same reference audio and transcript, seed, sway sampling, and CFG settings
were reused. The output duration was 3.328 seconds.

| Setting | Warm median | RTF | Peak allocated | Relative to 32 NFE |
| --- | ---: | ---: | ---: | ---: |
| BF16, 32 NFE | 855.374 ms | 0.25702 | 726.83 MiB | 1.00x |
| BF16, 16 NFE | 416.470 ms | 0.12514 | 724.59 MiB | 2.05x |
| BF16, 8 NFE | 199.917 ms | 0.06007 | 723.47 MiB | 4.28x |
| BF16, 4 NFE | 198.903 ms | 0.05977 | 722.91 MiB | 4.30x |
| BF16, 32 NFE plus DBCache | 711.4 ms | 0.21375 | — | 1.20x |

Eight NFE is the latency candidate, not a new quality default. Four NFE did
not materially improve latency because the fixed vocoder cost dominated and
it removes more solver evaluations. DBCache used front/back depth 1,
threshold 0.08, two warm-up steps, at most two consecutive reuse steps, and
an error budget of 0.20; it reused 12 of 32 steps (37.5%).

All approximate outputs were finite and had the same 79,872 samples.
Against the deterministic 32-NFE waveform, cosine similarity was 0.703 at
16 NFE, 0.275 at 8 NFE, and 0.032 at 4 NFE. DBCache at 32 NFE measured
0.772 cosine and 0.0330 mean absolute waveform error. Waveform similarity is
phase-sensitive and is not a perceptual score, but these differences confirm
that neither NFE reduction nor caching is an exact optimization. They remain
off by default.

A paired 16-NFE dtype run measured:

| Precision | Warm median | RTF | Peak allocated |
| --- | ---: | ---: | ---: |
| FP32 | 529.697 ms | 0.15916 | 1,438.34 MiB |
| BF16 | 462.576 ms | 0.13900 | 725.59 MiB |

BF16 was 1.15x faster and used 49.5% less peak allocated memory. This run
also exposed and fixed mixed-dtype STFT, Vocos, and ISTFT boundaries; FFT
operations promote internally to FP32 and return the expected runtime dtype.

Compiling the entire sampler reached 125.1–176.3 ms steady state but required
60.86 seconds on the first request and another 4.39-second recompile. It is a
deploy-only option after shape bucketing. Compiling only the F5 transformer
with CUDA graphs was rejected because its cross-step text cache changes tensor
ownership; the no-graph max-autotune variant needed 135.55 seconds to compile
and then varied from 130.2 to 240.4 ms, with no reliable win over eager.

## Technique disposition

The implementation survey covered the techniques that can preserve the
current checkpoints' mathematics. Training-dependent distillation and
multi-GPU methods are listed explicitly instead of being mislabeled as
single-GPU inference switches.

| Technique | VITS | LLM-TTS | Diffusion/flow | Result |
| --- | --- | --- | --- | --- |
| BF16/FP16 autocast | Supported | Supported per checkpoint/codec | Measured BF16 | Keep with numerically sensitive FP32 boundaries |
| Weight-normalization materialization | Measured | Not applicable | Not applicable | Reversible eval cache added |
| Regional `torch.compile` | Rejected for inference | Graph-specific | Measured | Architecture-specific; keep only after the checkpoint quality gate |
| CUDA graphs | Static buckets only | Static cache/buckets only | Static buckets only | Never automatic for dynamic requests |
| SDPA/FlashAttention kernel dispatch | Not attention-bound | Measured fused GQA | Existing selectable SDPA | Fused unmasked causal path added |
| Dynamic KV cache | Not applicable | Existing | Outer LMs only | Keep |
| Python static KV cache | Not applicable | Measured slower | Not applicable | Rejected |
| NFE/schedule reduction | Not applicable | Not applicable | Measured | Opt-in approximate acceleration |
| Limited/adaptive CFG | Not applicable | Not applicable | Implemented where branches are separable | Requires checkpoint calibration |
| FORA/TeaCache/SmoothCache/Taylor reuse | Not applicable | Not applicable | Implemented, fail-closed calibration | No production default without quality data |
| DBCache/first-block cache | Not applicable | Not applicable | Measured DBCache | Opt-in approximate acceleration |
| STORK-2 | Not applicable | Not applicable | Implemented for direct velocity fields | Use only after quality validation |
| TensorRT/export engines | Backend-specific | External serving supported | Upstream F5 server is external | Not merged into the portable native graph |
| AWQ/GPTQ/FP8/low-bit kernels | Conv-heavy graph mismatch | Needs checkpoint calibration | Mixed conv/attention graph | No unvalidated quantized default |
| Speculative/Medusa decoding | Not applicable | Requires a compatible draft/head | Not applicable | Training/checkpoint work, not a generic flag |
| Progressive/consistency distillation | Not applicable | Not applicable | Requires a new trained checkpoint | Future training recipe |
| Tensor/sequence/context parallelism | One GPU available | One GPU available | One GPU available | Not applicable to this run |

Primary implementation references include
[VITS](https://arxiv.org/abs/2106.06103),
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention),
[FlashAttention](https://arxiv.org/abs/2205.14135),
[F5-TTS](https://github.com/SWivid/F5-TTS),
[DeepCache](https://arxiv.org/abs/2312.00858),
[TeaCache](https://arxiv.org/abs/2411.19108),
[SmoothCache](https://arxiv.org/abs/2411.10510),
[TaylorSeer](https://arxiv.org/abs/2503.06923), and
[LoRA](https://arxiv.org/abs/2106.09685). See the
[diffusion optimization guide](diffusion-optimization.md) for the exact
compatibility gates.

## Tiny fine-tuning smoke

Each smoke uses a deliberately tiny deterministic dataset. Passing means a
forward loss, backward pass, optimizer update, export, and reload boundary
completed; it is not an audio-quality claim.

| Family | Route | Records/steps | Trainable parameters | Result |
| --- | --- | ---: | ---: | --- |
| VITS | Full adversarial generator + discriminator | 1 record / 5 alternating steps | 7,249 + 11 (tiny graphs) | Both parameter sets changed; all losses finite |
| LLM-TTS | Qwen3-TTS native LoRA | 1 synthetic batch / 25 steps | 320 (tiny graph) | Loss 4.41021 → 3.67348; frozen-base drift 0; merged-logit error `2.38e-7` |
| Diffusion | F5-TTS full DiT flow | 1 record / 25 steps | 47,752 (tiny graph) | Fixed-path loss 1.87107 → 0.13962 |

Qwen3-TTS full SFT remains the default. Opt-in LoRA freezes the Base model and
targets the talker and residual-code-predictor attention and MLP projections.
Adapter-only Safetensors can resume training; export creates a clean merged
CustomVoice runtime without mutating the live training model. The complete
fine-tuning limits for every provider are in
[TTS training support](../models/training-support.md).
