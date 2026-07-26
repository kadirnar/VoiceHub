# MOSS-TTS 权重转换为 GGUF 格式

[English](README.md) | [简体中文](README_zh.md)

本指南介绍如何将原始 MOSS-TTS（HuggingFace）权重转换为 llama.cpp 推理后端使用的 GGUF 格式。如果你只想 **直接使用** 预转换的权重，可以跳过本指南，直接下载：

```bash
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights/MOSS-TTS-GGUF
```

## 流程概览

转换流程分为三步：

1. **提取权重** — 将 MOSS-TTS 模型拆分为独立的 Qwen3 backbone（safetensors）、embedding 表（`.npy`）和 LM head 矩阵（`.npy`）。
2. **转换为 GGUF** — 使用 llama.cpp 的 `convert_hf_to_gguf.py` 将 Qwen3 backbone safetensors 转换为全精度（f16）GGUF 文件。
3. **量化** — 使用 `llama-quantize` 将 f16 GGUF 量化为更小的格式（如 Q4_K_M）。

```
OpenMOSS-Team/MOSS-TTS（HuggingFace）
  │
  ▼  第 1 步：extract_weights.py
  ├── qwen3_backbone/     （safetensors + config.json）
  ├── embeddings/          （33 × .npy）
  └── lm_heads/            （33 × .npy）
        │
        ▼  第 2 步：convert_hf_to_gguf.py
        backbone_f16.gguf
        │
        ▼  第 3 步：llama-quantize
        backbone_q4km.gguf
```

## 前置条件

- Python >= 3.10
- `safetensors`、`numpy`、`torch`、`huggingface_hub`（`pip install safetensors numpy torch huggingface_hub`）
- 编译好的 [llama.cpp](https://github.com/ggerganov/llama.cpp)（用于 `convert_hf_to_gguf.py` 和 `llama-quantize`）

### 编译 llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
cd ..
```

编译完成后你将获得：
- `llama.cpp/convert_hf_to_gguf.py` — HF 转 GGUF 的脚本
- `llama.cpp/build/bin/llama-quantize` — 量化工具

## 第 1 步：提取权重

将完整的 MOSS-TTS 模型拆分为三组。如果未提供本地路径，脚本会自动从 HuggingFace 下载模型。

```bash
python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model OpenMOSS-Team/MOSS-TTS \
    --output weights/extracted
```

使用 **本地** 模型目录（跳过下载）：

```bash
python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model /path/to/MOSS-TTS \
    --output weights/extracted
```

### 输出结构

```
weights/extracted/
├── qwen3_backbone/
│   ├── config.json                          # Qwen3ForCausalLM 配置
│   ├── model-00001-of-00004.safetensors     # backbone 分片
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   ├── model.safetensors.index.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── ...
├── embeddings/
│   ├── embed_tokens.npy      # 共享文本 embedding 表
│   ├── emb_ext_00.npy        # 音频 embedding codebook 0
│   ├── emb_ext_01.npy
│   └── ...                   # （共 32 个音频 codebook）
├── lm_heads/
│   ├── lm_head_text.npy      # 文本 LM head
│   ├── lm_head_audio_00.npy  # 音频 LM head 0
│   ├── lm_head_audio_01.npy
│   └── ...                   # （共 32 个音频 head）
└── extraction_meta.json       # 元数据（词表大小、路径等）
```

## 第 2 步：将 Backbone 转换为 GGUF

使用 llama.cpp 的转换脚本将提取出的 Qwen3 backbone 转为 GGUF 文件：

```bash
python llama.cpp/convert_hf_to_gguf.py \
    weights/extracted/qwen3_backbone \
    --outfile weights/backbone_f16.gguf \
    --outtype f16
```

这会生成一个约 16 GB 的 f16 GGUF 文件。

## 第 3 步：量化

将 f16 GGUF 量化为更小的格式。`Q4_K_M` 在质量和大小之间取得了较好的平衡：

```bash
llama.cpp/build/bin/llama-quantize \
    weights/backbone_f16.gguf \
    weights/backbone_q4km.gguf \
    Q4_K_M
```

这会将文件从约 16 GB 缩减到约 4.8 GB。

### 其他量化选项

| 类型 | 近似大小 | BPW | 说明 |
|------|---------|-----|------|
| `Q4_K_M` | ~4.8 GB | 4.91 | 推荐默认选项 |
| `Q5_K_M` | ~5.7 GB | 5.69 | 质量稍好 |
| `Q6_K` | ~6.6 GB | 6.56 | 大多数场景近乎无损 |
| `Q8_0` | ~8.7 GB | 8.50 | 最高质量的量化 |

## 完整示例

```bash
# 0. 编译 llama.cpp（一次性）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j && cd ..

# 1. 提取权重
python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model OpenMOSS-Team/MOSS-TTS \
    --output weights/extracted

# 2. 转换为 f16 GGUF
python llama.cpp/convert_hf_to_gguf.py \
    weights/extracted/qwen3_backbone \
    --outfile weights/backbone_f16.gguf \
    --outtype f16

# 3. 量化为 Q4_K_M
llama.cpp/build/bin/llama-quantize \
    weights/backbone_f16.gguf \
    weights/backbone_q4km.gguf \
    Q4_K_M

# 完成！使用量化后的 backbone + embeddings + lm_heads 进行推理。
# 用法详见 llama.cpp 后端 README。
```

## 使用转换后的权重

转换完成后，按以下结构组织权重供 llama.cpp 后端使用：

```
weights/
├── backbone_q4km.gguf          # 来自第 3 步
├── embeddings/                  # 来自第 1 步（weights/extracted/embeddings/）
│   ├── embed_tokens.npy
│   └── emb_ext_*.npy
├── lm_heads/                    # 来自第 1 步（weights/extracted/lm_heads/）
│   ├── lm_head_text.npy
│   └── lm_head_audio_*.npy
└── tokenizer/                   # 来自第 1 步（weights/extracted/qwen3_backbone/）
    ├── tokenizer.json
    └── tokenizer_config.json
```

然后更新你的配置 YAML（如 `configs/llama_cpp/default.yaml`），将路径指向这些文件，即可运行推理：

```bash
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "你好世界！" \
    --output output.wav
```

## 常见问题

- **`convert_hf_to_gguf.py` 报错 "unknown model architecture"**：请确认你转换的是 `qwen3_backbone/` 目录（而非原始 MOSS-TTS 目录）。`config.json` 中必须声明 `"architectures": ["Qwen3ForCausalLM"]`。
- **提取时内存不足**：提取脚本使用懒加载，峰值内存约等于一个 safetensors 分片（约 5 GB）。如果内存仍然不够，请关闭其他应用。
- **量化后文件大小异常**：请确认你量化的是 f16 GGUF（而非已经量化过的文件），并仔细检查量化类型参数。
