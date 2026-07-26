# MOSS-TTS Local Transformer v1.5 微调教程

本目录提供 `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` 的完整监督微调流程。

相比旧版 `MOSS-TTS-Local-Transformer`，v1.5 的关键区别是：

- 使用 `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
- 原生 48 kHz 双通道音频
- 固定 12 层 RVQ
- 使用完整语言标签，例如 `"English"`、`"Chinese"`

目录中的脚本包括：

- `prepare_data.py`: 预提取目标音频的 `audio_codes`，支持按 rank 切分数据并分别保存结果
- `dataset.py`: 将文本、可选单参考音频、语言标签等字段打包为 teacher-forcing 样本
- `sft.py`: 支持单卡、数据并行，以及可选 DeepSpeed ZeRO-3 训练
- `run_train.sh`: 一键预处理并训练

## 1. 环境准备

先安装训练依赖：

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune]"
```

如果你的环境支持 FlashAttention 2，也可以继续沿用根目录 README 里的安装方式。
使用 `--attn-implementation auto` 时，脚本会优先尝试 FlashAttention 2；当前环境不满足条件时，
CUDA 上回退到 SDPA，CPU 上回退到 eager。

如果准备使用 **DeepSpeed ZeRO-3**，请额外安装：

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune-deepspeed]"
```

## 2. 输入 JSONL 格式

所有记录共享一套基础思路：

- `audio`: 目标训练音频路径，`prepare_data.py` 会把它编码为 `audio_codes`
- `text`: 需要合成的文本
- `language`: 可选但建议填写；使用 `"English"`、`"Chinese"`、`"French"` 这样的完整语言标签
- `ref_audio`: 可选的单条参考音频路径，用于音色克隆

v1.5 codec 内部使用 48 kHz 双通道音频。输入音频可以是其他采样率或单通道；
processor 会在编码前完成加载和归一化。

### 2.1 纯 `text, speech` pair

这种格式不需要参考音频：

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","language":"Chinese"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","language":"English"}
{"audio":"./data/utt0003.wav","text":"Bonjour, je voudrais essayer une voix francaise naturelle et stable.","language":"French"}
```

### 2.2 音色克隆 / 参考音频条件训练

使用 `ref_audio` 放一条克隆参考音频：

```jsonl
{"audio":"./data/utt0001.wav","text":"请用同一个声音说出这句话。","ref_audio":"./data/ref.wav","language":"Chinese"}
{"audio":"./data/utt0002.wav","text":"A warm line spoken in the same voice.","ref_audio":"./data/ref.wav","language":"English"}
```

说明：

- `ref_audio` 应该是单个路径，不是列表
- `prepare_data.py` 默认会把 `ref_audio` 编码为 `ref_audio_codes`
- 如果 JSONL 已经包含 `audio_codes` 和 `ref_audio_codes`，训练阶段可以跳过 codec 预处理


## 3. 预处理数据

### 3.1 单进程

```bash
python moss_tts_local_v1.5/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
    --codec-weight-dtype fp32 \
    --codec-compute-dtype bf16 \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl
```

默认情况下，`prepare_data.py` 会自动预编码参考音频；如果只想编码目标音频，可以显式关闭：

```bash
python moss_tts_local_v1.5/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
    --codec-weight-dtype fp32 \
    --codec-compute-dtype bf16 \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl train_with_codes.jsonl \
    --skip-reference-audio-codes
```

v1.5 微调流程中，codec 权重加载精度固定为 `fp32`。

### 3.2 多机多卡并行编码

`prepare_data.py` 直接按 `accelerate launch` 的多进程语义切分数据。
例如 2 台节点、16 张卡，总共切 16 份，每个 rank 单独输出一个 shard：

```bash
accelerate launch --num_processes 16 moss_tts_local_v1.5/finetuning/prepare_data.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-path OpenMOSS-Team/MOSS-Audio-Tokenizer-v2 \
    --codec-weight-dtype fp32 \
    --codec-compute-dtype bf16 \
    --device auto \
    --input-jsonl train_raw.jsonl \
    --output-jsonl prepared/train_with_codes.jsonl
```

输出会类似：

- `prepared/train_with_codes.rank00000-of-00016.jsonl`
- `prepared/train_with_codes.rank00001-of-00016.jsonl`
- ...
- `prepared/train_with_codes.rank00015-of-00016.jsonl`

后续训练阶段，`sft.py` 可以直接读取：

- 单个 JSONL
- 一个目录
- 一个 glob，例如 `prepared/train_with_codes.rank*.jsonl`
- 或逗号分隔的多个文件

使用预分片文件做 DDP 训练时，每个 rank 读取自己的 shard。训练器会按最短
shard 对齐所有 rank，只丢弃无法对齐的尾部 micro-batch，确保所有 rank 的
forward/backward 次数一致。

## 4. 启动训练

### 4.1 单卡基线

```bash
accelerate launch moss_tts_local_v1.5/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-weight-dtype fp32 \
    --codec-compute-dtype bf16 \
    --train-jsonl train_with_codes.jsonl \
    --output-dir output/moss_tts_local_v1_5_sft \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 2.0e-5 \
    --warmup-ratio 0 \
    --lr-scheduler-type constant \
    --num-epochs 3 \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

### 4.2 数据并行

单机 8 卡数据并行可直接使用模板：

```bash
accelerate launch \
    --config_file moss_tts_local_v1.5/finetuning/configs/accelerate_ddp_8gpu.yaml \
    moss_tts_local_v1.5/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-weight-dtype fp32 \
    --codec-compute-dtype bf16 \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_local_v1_5_sft_ddp \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2.0e-5 \
    --warmup-ratio 0 \
    --lr-scheduler-type constant \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

### 4.3 可选 DeepSpeed ZeRO-3 训练

v1.5 SFT 路径支持 DDP 和 DeepSpeed ZeRO-3。

```bash
accelerate launch \
    --config_file moss_tts_local_v1.5/finetuning/configs/accelerate_zero3_1.7b.yaml \
    moss_tts_local_v1.5/finetuning/sft.py \
    --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 \
    --codec-weight-dtype fp32 \
    --codec-compute-dtype bf16 \
    --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
    --output-dir output/moss_tts_local_v1_5_sft_zero3 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2.0e-5 \
    --warmup-ratio 0 \
    --lr-scheduler-type constant \
    --mixed-precision bf16 \
    --channelwise-loss-weight 1,32 \
    --gradient-checkpointing
```

ZeRO-3 需要 `deepspeed` 包；如果只使用单卡或 DDP，则不需要额外安装它。

### 4.4 常用可调超参数

`sft.py` 将常见训练超参数都直接开放出来：

- 优化器：`--learning-rate`、`--weight-decay`、`--adam-beta1`、`--adam-beta2`、`--adam-eps`
- 学习率调度：`--lr-scheduler-type`、`--warmup-steps`、`--warmup-ratio`
- 稳定性相关：`--max-grad-norm`、`--gradient-checkpointing`、`--mixed-precision`
- RVQ 多头 loss 加权：`--channelwise-loss-weight`

`--channelwise-loss-weight` 支持两种写法：

- `n_vq + 1` 个值：`text_head,vq0,...,vq11`
- 两个值：`text_weight,total_audio_weight`

默认值是 `1,32`，表示总音频 loss 权重为 32，并平均分到 12 个 RVQ head。

训练日志会直接打印：

- 带时间戳的日志前缀
- `global_batch_size` 及其计算公式
- `step_time`
- `steps_per_sec`
- `samples_per_sec`
- `eta`

### 4.5 多机训练

将配置文件里的以下字段改成你的集群值即可：

- `num_machines`
- `num_processes`
- `machine_rank`
- `main_process_ip`
- `main_process_port`

例如 2 节点 16 卡，可以在两台机器分别设置：

- 节点 0: `machine_rank: 0`
- 节点 1: `machine_rank: 1`
- `num_machines: 2`
- `num_processes: 16`

其余训练命令保持不变。

## 5. 快速推理验证

`sft.py` 保存的每个 checkpoint 目录都会附带模型配置、运行时 Python 文件、
tokenizer 文件和 processor 元数据，因此可以直接对这个目录调用 `from_pretrained`：

```python
from pathlib import Path
import importlib.util

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor

torch.backends.cuda.enable_cudnn_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)


def resolve_attn_implementation(device: str, dtype: torch.dtype) -> str:
    if (
        device == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    if device == "cuda":
        return "sdpa"
    return "eager"


model_path = "output/moss_tts_local_v1_5_sft/checkpoint-last"
reference_audio = "./assets/audio/reference_zh_0.wav"
text = "今天我们继续把 MOSS-TTS Local v1.5 的微调流程跑通。"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
attn_implementation = resolve_attn_implementation(device, dtype)

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    codec_weight_dtype="fp32",
    codec_compute_dtype="bf16",
)
processor.audio_tokenizer = processor.audio_tokenizer.to(device)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype,
    attn_implementation=attn_implementation,
).to(device)
model.eval()

conversation = [[
    processor.build_user_message(
        text=text,
        reference=[reference_audio],
        language="Chinese",
    )
]]

batch = processor(conversation, mode="generation")
outputs = model.generate(
    input_ids=batch["input_ids"].to(device),
    attention_mask=batch["attention_mask"].to(device),
    max_new_tokens=4096,
    do_sample=True,
    audio_temperature=1.2,
    audio_top_p=1.0,
    audio_top_k=25,
    audio_repetition_penalty=1.0,
)

message = processor.decode(outputs)[0]
audio = message.audio_codes_list[0]
Path("demo_outputs").mkdir(parents=True, exist_ok=True)
torchaudio.save("demo_outputs/finetuned_v1_5_sample.wav", audio, processor.model_config.sampling_rate)
```

MOSS-TTS Local v1.5 输出双通道音频。`message.audio_codes_list[0]` 可以直接作为双通道张量保存。

## 6. 一键脚本

直接使用：

```bash
bash moss_tts_local_v1.5/finetuning/run_train.sh
```

常见环境变量：

- `RAW_JSONL`: 原始训练 JSONL
- `PREPARED_JSONL`: `prepare_data.py` 输出文件
- `TRAIN_JSONL`: 可选；训练输入，可以是单文件、目录或 glob。默认会自动从 `PREPARED_JSONL` 推断
- `OUTPUT_DIR`: 训练输出目录
- `ACCELERATE_CONFIG_FILE`: 可选，填 DDP 或 ZeRO-3 配置
- `SKIP_PREPARE`: 设为 `1` 时跳过预处理，直接用现有的 `TRAIN_JSONL` / `PREPARED_JSONL` 进入训练
- `CODEC_WEIGHT_DTYPE`: 必须是 `fp32`
- `CODEC_COMPUTE_DTYPE`: 默认 `bf16`
- `PREP_EXTRA_ARGS_STR`: 额外传给 `prepare_data.py`
- `PREP_ACCELERATE_ARGS_STR`: 如果希望预处理也通过 `accelerate launch` 并行启动，可设置这组参数，例如 `--num_processes 16` 或 `--config_file moss_tts_local_v1.5/finetuning/configs/accelerate_ddp_8gpu.yaml`
- `TRAIN_EXTRA_ARGS_STR`: 额外传给 `sft.py`

例如用 ZeRO-3 一键启动：

```bash
RAW_JSONL=train_raw.jsonl \
PREPARED_JSONL=prepared/train_with_codes.jsonl \
OUTPUT_DIR=output/moss_tts_local_v1_5_sft_zero3 \
CODEC_WEIGHT_DTYPE=fp32 \
CODEC_COMPUTE_DTYPE=bf16 \
ACCELERATE_CONFIG_FILE=moss_tts_local_v1.5/finetuning/configs/accelerate_zero3_1.7b.yaml \
PREP_ACCELERATE_ARGS_STR='--config_file moss_tts_local_v1.5/finetuning/configs/accelerate_ddp_8gpu.yaml' \
PREP_EXTRA_ARGS_STR='' \
TRAIN_EXTRA_ARGS_STR='--per-device-batch-size 1 --gradient-accumulation-steps 4 --learning-rate 2.0e-5 --warmup-ratio 0 --lr-scheduler-type constant --num-epochs 3 --mixed-precision bf16 --channelwise-loss-weight 1,32 --gradient-checkpointing' \
bash moss_tts_local_v1.5/finetuning/run_train.sh
```

## 7. 格式补充说明

这套微调流程面向 `MOSS-TTS-Local-Transformer-v1.5`。

推荐字段：

- `audio`: 必填目标音频，除非已经提前写入 `audio_codes`
- `text`: 目标 transcript 或合成文本
- `language`: 可选但推荐填写完整语言标签
- `ref_audio`: 可选的单条参考音频路径，用于音色克隆
- `tokens`: 可选的期望音频 token 数，用于时长控制

兼容字段：

- `reference_audio`: 预处理阶段接受它作为参考音频字段别名
- `reference`: processor 可以接受该字段，但本公开版 v1.5 微调 README 不将多参考或多说话人训练作为支持流程说明
- `instruction`、`ambient_sound`、`quality`、`sound_event`: 如果记录中存在，会被转发给 user message

共享脚本：

- 数据准备统一使用 `prepare_data.py`
- 训练统一使用 `sft.py`
- `train-jsonl` 支持单文件、目录、glob、多文件列表
