# Converting MOSS-TTS Weights to GGUF

[English](README.md) | [简体中文](README_zh.md)

This guide walks through converting the original MOSS-TTS (HuggingFace) weights into the GGUF format used by the llama.cpp inference backend. If you just want to **use** the pre-converted weights, skip this guide and download them directly:

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights/MOSS-TTS-GGUF
```

## Overview

The conversion pipeline has three steps:

1. **Extract weights** — split the MOSS-TTS model into a standalone Qwen3 backbone (safetensors), embedding tables (`.npy`), and LM head matrices (`.npy`).
2. **Convert to GGUF** — convert the Qwen3 backbone safetensors to a full-precision (f16) GGUF file using llama.cpp's `convert_hf_to_gguf.py`.
3. **Quantize** — quantize the f16 GGUF to a smaller format (e.g. Q4_K_M) using `llama-quantize`.

```
OpenMOSS-Team/MOSS-TTS (HuggingFace)
  │
  ▼  Step 1: extract_weights.py
  ├── qwen3_backbone/     (safetensors + config.json)
  ├── embeddings/          (33 × .npy)
  └── lm_heads/            (33 × .npy)
        │
        ▼  Step 2: convert_hf_to_gguf.py
        backbone_f16.gguf
        │
        ▼  Step 3: llama-quantize
        backbone_q4km.gguf
```

## Prerequisites

- Python >= 3.10
- `safetensors`, `numpy`, `torch`, `huggingface_hub` (`pip install safetensors numpy torch huggingface_hub`)
- A compiled [llama.cpp](https://github.com/ggerganov/llama.cpp) tree (for `convert_hf_to_gguf.py` and `llama-quantize`)

### Building llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
cd ..
```

After building, you will have:
- `llama.cpp/convert_hf_to_gguf.py` — HF-to-GGUF conversion script
- `llama.cpp/build/bin/llama-quantize` — quantization tool

## Step 1: Extract Weights

This splits the full MOSS-TTS model into three component groups. The script downloads the model from HuggingFace automatically if a local path is not provided.

```bash
python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model OpenMOSS-Team/MOSS-TTS \
    --output weights/extracted
```

To use a **local** model directory instead of downloading:

```bash
python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model /path/to/MOSS-TTS \
    --output weights/extracted
```

### Output structure

```
weights/extracted/
├── qwen3_backbone/
│   ├── config.json                          # Qwen3ForCausalLM config
│   ├── model-00001-of-00004.safetensors     # backbone shards
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── model.safetensors.index.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── embeddings/
│   ├── embed_tokens.npy      # shared text embedding table
│   ├── emb_ext_00.npy        # audio embedding codebook 0
│   ├── emb_ext_01.npy
│   └── ...                   # (32 audio codebooks total)
├── lm_heads/
│   ├── lm_head_text.npy      # text LM head
│   ├── lm_head_audio_00.npy  # audio LM head 0
│   ├── lm_head_audio_01.npy
│   └── ...                   # (32 audio heads total)
└── extraction_meta.json       # metadata (vocab sizes, paths, etc.)
```

## Step 2: Convert Backbone to GGUF

Use llama.cpp's conversion script to turn the extracted Qwen3 backbone into a GGUF file:

```bash
python llama.cpp/convert_hf_to_gguf.py \
    weights/extracted/qwen3_backbone \
    --outfile weights/backbone_f16.gguf \
    --outtype f16
```

This produces a ~16 GB f16 GGUF file.

## Step 3: Quantize

Quantize the f16 GGUF to a smaller format. `Q4_K_M` is a good balance of quality and size:

```bash
llama.cpp/build/bin/llama-quantize \
    weights/backbone_f16.gguf \
    weights/backbone_q4km.gguf \
    Q4_K_M
```

This reduces the file from ~16 GB to ~4.8 GB.

### Other quantization options

| Type | Approx. Size | BPW | Notes |
|------|-------------|-----|-------|
| `Q4_K_M` | ~4.8 GB | 4.91 | Recommended default |
| `Q5_K_M` | ~5.7 GB | 5.69 | Slightly better quality |
| `Q6_K` | ~6.6 GB | 6.56 | Near-lossless for most uses |
| `Q8_0` | ~8.7 GB | 8.50 | Highest quality quantization |

## All-in-One Example

```bash
# 0. Build llama.cpp (one-time)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j && cd ..

# 1. Extract weights
python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model OpenMOSS-Team/MOSS-TTS \
    --output weights/extracted

# 2. Convert to f16 GGUF
python llama.cpp/convert_hf_to_gguf.py \
    weights/extracted/qwen3_backbone \
    --outfile weights/backbone_f16.gguf \
    --outtype f16

# 3. Quantize to Q4_K_M
llama.cpp/build/bin/llama-quantize \
    weights/backbone_f16.gguf \
    weights/backbone_q4km.gguf \
    Q4_K_M

# Done! Use the quantized backbone + embeddings + lm_heads for inference.
# See the llama.cpp backend README for usage instructions.
```

## Using the Converted Weights

After conversion, arrange the weights for the llama.cpp backend:

```
weights/
├── backbone_q4km.gguf          # from Step 3
├── embeddings/                  # from Step 1 (weights/extracted/embeddings/)
│   ├── embed_tokens.npy
│   └── emb_ext_*.npy
├── lm_heads/                    # from Step 1 (weights/extracted/lm_heads/)
│   ├── lm_head_text.npy
│   └── lm_head_audio_*.npy
└── tokenizer/                   # from Step 1 (weights/extracted/qwen3_backbone/)
    ├── tokenizer.json
    └── tokenizer_config.json
```

Then update your config YAML (e.g. `configs/llama_cpp/default.yaml`) to point to these paths and run inference:

```bash
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello, world!" \
    --output output.wav
```

## Troubleshooting

- **`convert_hf_to_gguf.py` fails with "unknown model architecture"**: Make sure you are converting the `qwen3_backbone/` directory (not the original MOSS-TTS directory). The `config.json` must declare `"architectures": ["Qwen3ForCausalLM"]`.
- **Out of memory during extraction**: The extraction script uses lazy loading, so peak memory should be roughly one safetensors shard (~5 GB). If memory is still tight, close other applications.
- **Quantization produces unexpected size**: Verify you are quantizing the f16 GGUF (not an already-quantized file). Double-check the quantization type argument.
