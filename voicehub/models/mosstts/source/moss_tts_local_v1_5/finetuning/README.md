# Fine-Tuning MOSS-TTS Local Transformer v1.5

This directory provides a complete supervised finetuning workflow for
`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`.

Compared with the earlier `MOSS-TTS-Local-Transformer` release, v1.5 uses:

- `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`
- native 48 kHz stereo audio
- a fixed 12-layer RVQ layout
- full language tags such as `"English"` and `"Chinese"`

The scripts in this directory are:

- `prepare_data.py`: pre-extract target `audio_codes`, with rank-sharded output support
- `dataset.py`: pack text, optional single-reference audio, language tags, and related fields into teacher-forcing samples
- `sft.py`: supports single-GPU training, data parallel training, and optional DeepSpeed ZeRO-3 training
- `run_train.sh`: one-click prepare-and-train launcher

## 1. Install

Install training dependencies first:

```bash
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune]"
```

If your environment supports FlashAttention 2, you can also follow the
installation notes in the root README. With `--attn-implementation auto`, the
scripts prefer FlashAttention 2 when the package and GPU support it, otherwise
they fall back to SDPA on CUDA and eager attention on CPU.

If you plan to use **DeepSpeed ZeRO-3**, install the extra dependency group as
well:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e ".[torch-runtime,finetune-deepspeed]"
```

## 2. Input JSONL format

All records share the same basic idea:

- `audio`: target training audio path; `prepare_data.py` will encode it into `audio_codes`
- `text`: text to synthesize
- `language`: optional but recommended; use full tags such as `"English"`, `"Chinese"`, and `"French"`
- `ref_audio`: optional single reference audio path for voice cloning

The v1.5 codec expects 48 kHz stereo internally. Input audio can be another
sample rate or mono; the processor handles loading and normalization before
encoding.

### 2.1 Plain `text, speech` pairs

This format does not require reference audio:

```jsonl
{"audio":"./data/utt0001.wav","text":"Actually, I noticed that I am very sensitive to other people's emotions.","language":"English"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","language":"English"}
{"audio":"./data/utt0003.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","language":"Chinese"}
```

### 2.2 Voice cloning / reference-conditioned training

Use `ref_audio` for one cloning reference:

```jsonl
{"audio":"./data/utt0001.wav","text":"A warm line spoken in the same voice.","ref_audio":"./data/ref.wav","language":"English"}
{"audio":"./data/utt0002.wav","text":"请用同一个声音说出这句话。","ref_audio":"./data/ref.wav","language":"Chinese"}
```

Notes:

- `ref_audio` should be a single path, not a list
- `prepare_data.py` encodes `ref_audio` into `ref_audio_codes` by default
- if you already have `audio_codes` and `ref_audio_codes`, training can skip codec preprocessing


## 3. Prepare data

### 3.1 Single process

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

By default, `prepare_data.py` pre-encodes reference audio as well. If you only
want target audio codes, disable it explicitly:

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

Codec weight dtype is intentionally fixed to `fp32` in the v1.5
finetuning path. 

### 3.2 Multi-node / multi-GPU parallel preprocessing

`prepare_data.py` follows the `accelerate launch` multi-process model directly.
For example, with 2 nodes and 16 GPUs in total, the dataset is split into 16
shards and each rank writes one shard:

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

The output will look like:

- `prepared/train_with_codes.rank00000-of-00016.jsonl`
- `prepared/train_with_codes.rank00001-of-00016.jsonl`
- ...
- `prepared/train_with_codes.rank00015-of-00016.jsonl`

During training, `sft.py` can read:

- a single JSONL
- a directory
- a glob such as `prepared/train_with_codes.rank*.jsonl`
- or a comma-separated list of files

During DDP training from pre-sharded files, each rank reads its own shard. The
trainer aligns all ranks to the shortest shard and drops only the unmatched
tail micro-batches, so all ranks run the same number of forward/backward steps.

## 4. Train

### 4.1 Single-GPU baseline

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

### 4.2 Data parallel

For single-node 8-GPU data parallel training, use:

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

### 4.3 Optional DeepSpeed ZeRO-3 training

The v1.5 SFT path supports DDP and DeepSpeed ZeRO-3.

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

ZeRO-3 requires the `deepspeed` package. If you only use single-GPU or DDP
training, you do not need it.

### 4.4 Common tunable hyperparameters

`sft.py` exposes the common training hyperparameters directly:

- optimizer: `--learning-rate`, `--weight-decay`, `--adam-beta1`, `--adam-beta2`, `--adam-eps`
- LR schedule: `--lr-scheduler-type`, `--warmup-steps`, `--warmup-ratio`
- stability: `--max-grad-norm`, `--gradient-checkpointing`, `--mixed-precision`
- RVQ multi-head loss weighting: `--channelwise-loss-weight`

`--channelwise-loss-weight` supports two forms:

- `n_vq + 1` values: `text_head,vq0,...,vq11`
- two values: `text_weight,total_audio_weight`

The default is `1,32`, which means the total audio loss weight is 32 and is
evenly split across the 12 RVQ heads.

Training logs print:

- timestamped log prefixes
- `global_batch_size` and its formula
- `step_time`
- `steps_per_sec`
- `samples_per_sec`
- `eta`

### 4.5 Multi-node training

Update the following fields in the config file for your cluster:

- `num_machines`
- `num_processes`
- `machine_rank`
- `main_process_ip`
- `main_process_port`

For example, for 2 nodes and 16 GPUs:

- node 0: `machine_rank: 0`
- node 1: `machine_rank: 1`
- `num_machines: 2`
- `num_processes: 16`

The training command itself can stay unchanged.

## 5. Quick inference test

Each checkpoint saved by `sft.py` contains model config, runtime Python files,
tokenizer files, and processor metadata, so you can call `from_pretrained`
directly on that checkpoint directory:

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
reference_audio = "./assets/audio/reference_en_0.mp3"
text = "This is a quick finetuning smoke test for MOSS-TTS Local v1.5."

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
        language="English",
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

MOSS-TTS Local v1.5 outputs stereo audio. `message.audio_codes_list[0]` is
saved directly as a two-channel tensor.

## 6. One-click launcher

Run directly:

```bash
bash moss_tts_local_v1.5/finetuning/run_train.sh
```

Common environment variables:

- `RAW_JSONL`: raw training JSONL
- `PREPARED_JSONL`: output file from `prepare_data.py`
- `TRAIN_JSONL`: optional; training input, which can be a single file, directory, or glob. If unset, it is inferred automatically from `PREPARED_JSONL`
- `OUTPUT_DIR`: training output directory
- `ACCELERATE_CONFIG_FILE`: optional; DDP or ZeRO-3 config file
- `SKIP_PREPARE`: set to `1` to skip preprocessing and train directly from existing `TRAIN_JSONL` / `PREPARED_JSONL`
- `CODEC_WEIGHT_DTYPE`: must be `fp32`
- `CODEC_COMPUTE_DTYPE`: defaults to `bf16`
- `PREP_EXTRA_ARGS_STR`: extra arguments passed to `prepare_data.py`
- `PREP_ACCELERATE_ARGS_STR`: if you want preprocessing to also launch through `accelerate`, set this, for example `--num_processes 16` or `--config_file moss_tts_local_v1.5/finetuning/configs/accelerate_ddp_8gpu.yaml`
- `TRAIN_EXTRA_ARGS_STR`: extra arguments passed to `sft.py`

For example, to launch with ZeRO-3:

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

## 7. Additional format notes

This finetuning path targets `MOSS-TTS-Local-Transformer-v1.5`.

Recommended fields:

- `audio`: required target audio unless `audio_codes` is already present
- `text`: target transcript or synthesis text
- `language`: optional but recommended full language tag
- `ref_audio`: optional single reference audio path for voice cloning
- `tokens`: optional expected audio-token count for duration control

Compatibility fields:

- `reference_audio`: accepted as a reference-audio alias during preprocessing
- `reference`: accepted by the processor, but this v1.5 finetuning README does not document multi-reference or multi-speaker training as a supported public workflow
- `instruction`, `ambient_sound`, `quality`, `sound_event`: forwarded to the user message when present

Shared scripts:

- use `prepare_data.py` for codec preprocessing
- use `sft.py` for training
- `train-jsonl` supports a single file, directory, glob, or comma-separated multi-file list
