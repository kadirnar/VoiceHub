# MOSS-TTS-Delay: llama.cpp Inference Backend

[English](README.md) | [简体中文](README_zh.md)

This package provides a **torch-free** (or torch-optional) end-to-end TTS inference pipeline for MOSS-TTS-Delay using:

- **llama.cpp** for the Qwen3 backbone (GGUF format, GPU/CPU)
- **NumPy** for embeddings, LM heads, delay state machine, and sampling
- **ONNX Runtime** or **TensorRT** for the audio tokenizer

When PyTorch is available, LM heads can optionally be GPU-accelerated (~30x faster).

## Prerequisites

1. **llama.cpp** — compiled from source with shared library support
2. **Python >= 3.10**

## Installation

### Minimal (torch-free, ONNX audio)

```bash
pip install -e ".[llama-cpp-onnx]"
```

### With TensorRT audio (max performance)

```bash
pip install -e ".[llama-cpp-trt]"
```

### With PyTorch LM heads acceleration

```bash
pip install -e ".[llama-cpp-trt,llama-cpp-torch]"
```

## Weight Preparation

> To convert weights from the original MOSS-TTS model yourself (instead of downloading pre-quantized ones), see the [conversion guide](conversion/README.md).

### Step 1: Download pre-quantized TTS backbone & weights

We provide pre-quantized GGUF backbone, embedding tables, and LM head matrices on HuggingFace:

```bash
# Download pre-built GGUF + embeddings + lm_heads
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights/MOSS-TTS-GGUF
```

This gives you:
- `weights/MOSS-TTS-GGUF/MOSS_TTS_Q4_K_M.gguf` — Q4_K_M quantized backbone
- `weights/MOSS-TTS-GGUF/embeddings/` — 33 embedding `.npy` files
- `weights/MOSS-TTS-GGUF/lm_heads/` — 33 LM head `.npy` files
- `weights/MOSS-TTS-GGUF/tokenizer/` — BPE tokenizer files

### Step 2: Download ONNX audio tokenizer

We provide ONNX models for the audio tokenizer. **TensorRT engines are not provided** because they are tied to specific GPU architectures and TensorRT versions.

```bash
# Download ONNX encoder & decoder
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir weights/MOSS-Audio-Tokenizer-ONNX
```

### Step 3: Build the C bridge

```bash
# Clone and build llama.cpp (if not already done)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j
cd ..

# Build the C bridge shared library
cd moss_tts_delay/llama_cpp
bash build_bridge.sh /path/to/llama.cpp
```

### Step 4 (Optional): Build TensorRT engines

> **Note:** Only needed if you want to use `audio_backend: trt` for maximum audio tokenizer performance. Most users should use the ONNX backend.

```bash
bash moss_audio_tokenizer/trt/build_engine.sh \
    weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx \
    weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx \
    weights/MOSS-Audio-Tokenizer-TRT
```

> **⚠️ maxShapes determines the maximum audio length your engine can handle.**
> The default builds support up to **40 seconds** of audio. If you need longer audio,
> edit `MAX_AUDIO_SECONDS` in `build_engine.sh` before building.
> See the detailed shape ↔ duration table in the script's comments.

## Usage

### CLI

```bash
# Basic generation
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello, world!" \
    --output output.wav

# With reference audio (voice cloning)
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello!" \
    --reference ref.wav \
    --output output.wav

# Force numpy LM heads (torch-free)
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello!" \
    --heads-backend numpy

# With profiling
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello!" \
    --profile
```

### Python API

```python
from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig

config = PipelineConfig.from_yaml("configs/llama_cpp/default.yaml")

with LlamaCppPipeline(config) as pipeline:
    waveform = pipeline.generate(
        text="Hello, world!",
        reference_audio="ref.wav",  # optional
        language="en",
    )

import soundfile as sf
sf.write("output.wav", waveform, 24000)
```

### Batch Evaluation

```bash
python scripts/batch_eval_llama_cpp.py \
    --config configs/llama_cpp/default.yaml \
    --benchmark-dir /path/to/eval/tts \
    --result-dir results/llama_cpp_run \
    --suite seed-tts
```

## Benchmark

Quantization quality evaluated on [Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval) zero-shot benchmark. Baseline is the original HuggingFace model; GGUF variants use the llama.cpp backend with TensorRT audio tokenizer.

| Quantization | EN WER (%) ↓ | EN SIM (%) ↑ | ZH CER (%) ↓ | ZH SIM (%) ↑ |
|---|---:|---:|---:|---:|
| Baseline (HuggingFace) | 1.79 | 71.46 | 1.32 | 77.05 |
| Q8_0 | 3.21 | 68.61 | 1.56 | 76.03 |
| Q6_K | 3.11 | 68.77 | 1.44 | 76.06 |
| Q5_K_M | 2.95 | 68.55 | 1.50 | 75.96 |
| Q4_K_M | 2.83 | 68.15 | 1.58 | 75.71 |

## Configuration

### Config Files

| Config | Audio Backend | Use Case |
|--------|--------------|----------|
| `configs/llama_cpp/default.yaml` | ONNX | Recommended starting point |
| `configs/llama_cpp/trt.yaml` | TensorRT | Maximum throughput |
| `configs/llama_cpp/cpu-only.yaml` | ONNX (CPU) | No GPU required |

### Key Options

| Option | Values | Description |
|--------|--------|-------------|
| `heads_backend` | `auto` / `numpy` / `torch` | LM heads computation backend. `auto` uses torch if available |
| `audio_backend` | `onnx` / `trt` / `torch` | Audio tokenizer backend |
| `n_gpu_layers` | `-1` / `0` / `N` | GPU offload layers. -1 = all, 0 = CPU only |
| `n_ctx` | int | Context window size (prompt + generation) |
| `max_new_tokens` | int | Maximum generation steps |

## Architecture

```
Input text
  │
  ▼
Tokenizer (Rust BPE, tokenizers library)
  │
  ▼
build_generation_prompt() → input_ids (S, 33)
  │
  ▼
EmbeddingLookup (NumPy .npy) → embeddings (S, H)
  │
  ▼
LlamaCppBackbone (GGUF, C bridge) → hidden_state (H,)
  │
  ├─ [heads_backend=torch] TorchLMHeads (nn.Linear, GPU)
  │                          └─ audio_logits (32, 1025)
  │
  └─ [heads_backend=numpy] NumpyLMHeads (CPU matmul)
                             └─ audio_logits (32, 1025)
  │
  ▼
delay_step() + sampling (NumPy) → next_ids (33,)
  │
  ▼ (loop until EOS)
  │
Audio codes → AudioTokenizer (ONNX/TRT/Torch) → waveform
```

## File Structure

```
moss_tts_delay/llama_cpp/
├── __init__.py          # Package entry, exports LlamaCppPipeline
├── __main__.py          # python -m moss_tts_delay.llama_cpp
├── _constants.py        # Token IDs (from config.json, torch-free)
├── pipeline.py          # LlamaCppPipeline (main entry)
├── backbone.py          # LlamaCppBackbone (C bridge wrapper)
├── backbone_bridge.c    # C bridge source
├── build_bridge.sh      # Build script
├── embedding.py         # EmbeddingLookup (NumPy)
├── lm_heads.py          # NumpyLMHeads + TorchLMHeads
├── delay_state.py       # Delay state machine (NumPy)
├── sampling.py          # top-k/p sampling (NumPy)
├── processor.py         # Tokenizer + prompt builder
├── README.md            # This file
├── README_zh.md         # Chinese documentation
└── conversion/
    ├── extract_weights.py  # Weight extraction script
    ├── README.md           # Conversion guide (English)
    └── README_zh.md        # Conversion guide (Chinese)
```
