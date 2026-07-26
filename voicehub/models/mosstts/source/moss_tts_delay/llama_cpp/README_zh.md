# MOSS-TTS-Delay: llama.cpp 推理后端

[English](README.md) | [简体中文](README_zh.md)

本模块提供 MOSS-TTS-Delay 的 **无 PyTorch 依赖**（或 PyTorch 可选加速）端到端 TTS 推理流水线：

- **llama.cpp** 运行 Qwen3 backbone（GGUF 格式，GPU/CPU）
- **NumPy** 负责 embedding 查找、LM heads、delay 状态机和采样
- **ONNX Runtime** 或 **TensorRT** 运行音频编解码器

当 PyTorch 可用时，LM heads 可选择 GPU 加速（约 30 倍提速）。

## 前置条件

1. **llama.cpp** — 编译为共享库
2. **Python >= 3.10**

## 安装

### 最小安装（无 PyTorch，ONNX 音频）

```bash
pip install -e ".[llama-cpp-onnx]"
```

### TensorRT 音频（最高性能）

```bash
pip install -e ".[llama-cpp-trt]"
```

### PyTorch 加速 LM heads

```bash
pip install -e ".[llama-cpp-onnx,llama-cpp-torch]"
```

## 权重准备

> 如需从原始 MOSS-TTS 模型自行转换权重（而非下载预量化版本），请参阅 [转换指南](conversion/README_zh.md)。

### 第一步：下载预量化的 TTS Backbone 和权重

我们在 HuggingFace 上提供了预量化的 GGUF backbone、embedding 表和 LM head 矩阵：

```bash
# 下载预量化的 GGUF + embeddings + lm_heads
huggingface-cli download OpenMOSS-Team/MOSS-TTS-GGUF --local-dir weights/MOSS-TTS-GGUF
```

下载后的目录结构：
- `weights/MOSS-TTS-GGUF/MOSS_TTS_Q4_K_M.gguf` — Q4_K_M 量化的 backbone
- `weights/MOSS-TTS-GGUF/embeddings/` — 33 个 embedding `.npy` 文件
- `weights/MOSS-TTS-GGUF/lm_heads/` — 33 个 LM head `.npy` 文件
- `weights/MOSS-TTS-GGUF/tokenizer/` — BPE tokenizer 文件

### 第二步：下载 ONNX 音频编解码器

我们提供 ONNX 格式的音频编解码器模型。**我们不提供预编译的 TensorRT engine**，因为 TRT engine 与 GPU 架构和 TensorRT 版本强绑定。

```bash
# 下载 ONNX encoder & decoder
huggingface-cli download OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX --local-dir weights/MOSS-Audio-Tokenizer-ONNX
```

### 第三步：编译 C bridge

```bash
# 克隆并编译 llama.cpp（如果尚未完成）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j
cd ..

# 编译 C bridge 共享库
cd moss_tts_delay/llama_cpp
bash build_bridge.sh /path/to/llama.cpp
```

### 第四步（可选）：编译 TensorRT Engine

> **注意：** 仅当你想使用 `audio_backend: trt` 以获取最高音频编解码器性能时需要。大多数用户使用 ONNX 后端即可。

```bash
bash moss_audio_tokenizer/trt/build_engine.sh \
    weights/MOSS-Audio-Tokenizer-ONNX/encoder.onnx \
    weights/MOSS-Audio-Tokenizer-ONNX/decoder.onnx \
    weights/MOSS-Audio-Tokenizer-TRT
```

> **⚠️ maxShapes 决定了 engine 能处理的最长音频时长。**
> 默认配置支持最长 **40 秒** 的音频。如果需要更长的音频，
> 请在编译前修改 `build_engine.sh` 中的 `MAX_AUDIO_SECONDS` 变量。
> 详细的形状 ↔ 时长对照表请查看脚本注释。
>
> **快速计算公式：**
> - Encoder: `最大采样点数 = 秒数 × 24000`（须为 1920 的倍数）
> - Decoder: `最大帧数 = 最大采样点数 / 1920`

## 使用方法

### 命令行

```bash
# 基础生成
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "你好世界！" \
    --output output.wav

# 带参考音频（语音克隆）
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "你好！" \
    --reference ref.wav \
    --output output.wav

# 强制使用 numpy LM heads（纯无 torch）
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "你好！" \
    --heads-backend numpy

# 带性能分析
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "你好！" \
    --profile
```

### Python API

```python
from moss_tts_delay.llama_cpp import LlamaCppPipeline, PipelineConfig

config = PipelineConfig.from_yaml("configs/llama_cpp/default.yaml")

with LlamaCppPipeline(config) as pipeline:
    waveform = pipeline.generate(
        text="你好世界！",
        reference_audio="ref.wav",  # 可选
        language="zh",
    )

import soundfile as sf
sf.write("output.wav", waveform, 24000)
```

### 批量评测

```bash
python scripts/batch_eval_llama_cpp.py \
    --config configs/llama_cpp/default.yaml \
    --benchmark-dir /path/to/eval/tts \
    --result-dir results/llama_cpp_run \
    --suite seed-tts
```

## 量化评测

在 [Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval) zero-shot 评测集上的量化质量对比。Baseline 为 HuggingFace 原始模型；GGUF 变体使用 llama.cpp 后端 + TensorRT 音频编解码器。

| 量化方案 | EN WER (%) ↓ | EN SIM (%) ↑ | ZH CER (%) ↓ | ZH SIM (%) ↑ |
|---|---:|---:|---:|---:|
| Baseline (HuggingFace) | 1.79 | 71.46 | 1.32 | 77.05 |
| Q8_0 | 3.21 | 68.61 | 1.56 | 76.03 |
| Q6_K | 3.11 | 68.77 | 1.44 | 76.06 |
| Q5_K_M | 2.95 | 68.55 | 1.50 | 75.96 |
| Q4_K_M | 2.83 | 68.15 | 1.58 | 75.71 |

## 配置

### 预设配置

| 配置 | 音频后端 | 适用场景 |
|------|---------|---------|
| `configs/llama_cpp/default.yaml` | ONNX | 推荐入门 |
| `configs/llama_cpp/trt.yaml` | TensorRT | 最大吞吐 |
| `configs/llama_cpp/cpu-only.yaml` | ONNX (CPU) | 无需 GPU |

### 关键选项

| 选项 | 取值 | 说明 |
|------|------|------|
| `heads_backend` | `auto` / `numpy` / `torch` | LM heads 计算后端。`auto` 自动检测 torch |
| `audio_backend` | `onnx` / `trt` / `torch` | 音频编解码器后端 |
| `n_gpu_layers` | `-1` / `0` / `N` | GPU offload 层数。-1 = 全部，0 = 纯 CPU |
| `n_ctx` | int | 上下文窗口大小（prompt + 生成） |
| `max_new_tokens` | int | 最大生成步数 |

## 架构

```
输入文本
  │
  ▼
Tokenizer（Rust BPE，tokenizers 库）
  │
  ▼
build_generation_prompt() → input_ids (S, 33)
  │
  ▼
EmbeddingLookup（NumPy .npy）→ embeddings (S, H)
  │
  ▼
LlamaCppBackbone（GGUF，C bridge）→ hidden_state (H,)
  │
  ├─ [heads_backend=torch] TorchLMHeads（nn.Linear，GPU）
  │                          └─ audio_logits (32, 1025)
  │
  └─ [heads_backend=numpy] NumpyLMHeads（CPU matmul）
                             └─ audio_logits (32, 1025)
  │
  ▼
delay_step() + 采样（NumPy）→ next_ids (33,)
  │
  ▼（循环直到 EOS）
  │
Audio codes → AudioTokenizer（ONNX/TRT/Torch）→ 波形
```

## 文件结构

```
moss_tts_delay/llama_cpp/
├── __init__.py          # 包入口，导出 LlamaCppPipeline
├── __main__.py          # python -m moss_tts_delay.llama_cpp
├── _constants.py        # Token ID（从 config.json 加载，无 torch 依赖）
├── pipeline.py          # LlamaCppPipeline（主入口）
├── backbone.py          # LlamaCppBackbone（C bridge 封装）
├── backbone_bridge.c    # C bridge 源码
├── build_bridge.sh      # 编译脚本
├── embedding.py         # EmbeddingLookup（NumPy）
├── lm_heads.py          # NumpyLMHeads + TorchLMHeads
├── delay_state.py       # Delay 状态机（NumPy）
├── sampling.py          # top-k/p 采样（NumPy）
├── processor.py         # Tokenizer + prompt 构建器
├── README.md            # 英文文档
├── README_zh.md         # 本文件
└── conversion/
    ├── extract_weights.py  # 权重提取脚本
    ├── README.md           # 转换指南（英文）
    └── README_zh.md        # 转换指南（中文）
```
