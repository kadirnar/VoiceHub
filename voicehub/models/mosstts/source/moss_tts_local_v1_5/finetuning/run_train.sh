#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5}"
CODEC_PATH="${CODEC_PATH:-OpenMOSS-Team/MOSS-Audio-Tokenizer-v2}"

RAW_JSONL="${RAW_JSONL:-train_raw.jsonl}"
PREPARED_JSONL="${PREPARED_JSONL:-train_with_codes.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/moss_tts_local_v1_5_sft}"

PREP_DEVICE="${PREP_DEVICE:-auto}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

CODEC_WEIGHT_DTYPE="${CODEC_WEIGHT_DTYPE:-fp32}"
CODEC_COMPUTE_DTYPE="${CODEC_COMPUTE_DTYPE:-bf16}"
CODEC_ATTN_IMPLEMENTATION="${CODEC_ATTN_IMPLEMENTATION:-auto}"

PREP_ACCELERATE_ARGS_STR="${PREP_ACCELERATE_ARGS_STR:-}"
PREP_EXTRA_ARGS_STR="${PREP_EXTRA_ARGS_STR:-}"
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR:---per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 2.0e-5 --warmup-ratio 0 --lr-scheduler-type constant --num-epochs 3 --mixed-precision bf16 --channelwise-loss-weight 1,32 --gradient-checkpointing}"

PREP_ACCELERATE_ARGS=()
PREP_EXTRA_ARGS=()
TRAIN_EXTRA_ARGS=()

if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
  read -r -a PREP_ACCELERATE_ARGS <<< "${PREP_ACCELERATE_ARGS_STR}"
fi
if [[ -n "${PREP_EXTRA_ARGS_STR}" ]]; then
  read -r -a PREP_EXTRA_ARGS <<< "${PREP_EXTRA_ARGS_STR}"
fi
if [[ -n "${TRAIN_EXTRA_ARGS_STR}" ]]; then
  read -r -a TRAIN_EXTRA_ARGS <<< "${TRAIN_EXTRA_ARGS_STR}"
fi

if [[ "${CODEC_WEIGHT_DTYPE}" != "fp32" ]]; then
  echo "[ERROR] CODEC_WEIGHT_DTYPE=${CODEC_WEIGHT_DTYPE} is not supported by the public v1.5 finetuning path. Use CODEC_WEIGHT_DTYPE=fp32." >&2
  exit 1
fi

derive_shard_glob() {
  local path="$1"
  if [[ "$path" == *.jsonl ]]; then
    printf '%s\n' "${path%.jsonl}.rank*.jsonl"
  else
    printf '%s\n' "${path}.rank*"
  fi
}

if [[ -z "${TRAIN_JSONL}" ]]; then
  TRAIN_JSONL="${PREPARED_JSONL}"
  if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
    TRAIN_JSONL="$(derive_shard_glob "${PREPARED_JSONL}")"
  elif [[ ! -e "${PREPARED_JSONL}" ]]; then
    SHARD_GLOB="$(derive_shard_glob "${PREPARED_JSONL}")"
    if compgen -G "${SHARD_GLOB}" > /dev/null; then
      TRAIN_JSONL="${SHARD_GLOB}"
    fi
  fi
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
    accelerate launch "${PREP_ACCELERATE_ARGS[@]}" moss_tts_local_v1.5/finetuning/prepare_data.py \
      --model-path "${MODEL_PATH}" \
      --codec-path "${CODEC_PATH}" \
      --codec-weight-dtype "${CODEC_WEIGHT_DTYPE}" \
      --codec-compute-dtype "${CODEC_COMPUTE_DTYPE}" \
      --codec-attn-implementation "${CODEC_ATTN_IMPLEMENTATION}" \
      --device "${PREP_DEVICE}" \
      --input-jsonl "${RAW_JSONL}" \
      --output-jsonl "${PREPARED_JSONL}" \
      "${PREP_EXTRA_ARGS[@]}"
  else
    python moss_tts_local_v1.5/finetuning/prepare_data.py \
      --model-path "${MODEL_PATH}" \
      --codec-path "${CODEC_PATH}" \
      --codec-weight-dtype "${CODEC_WEIGHT_DTYPE}" \
      --codec-compute-dtype "${CODEC_COMPUTE_DTYPE}" \
      --codec-attn-implementation "${CODEC_ATTN_IMPLEMENTATION}" \
      --device "${PREP_DEVICE}" \
      --input-jsonl "${RAW_JSONL}" \
      --output-jsonl "${PREPARED_JSONL}" \
      "${PREP_EXTRA_ARGS[@]}"
  fi
fi

if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  accelerate launch --config_file "${ACCELERATE_CONFIG_FILE}" moss_tts_local_v1.5/finetuning/sft.py \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --codec-weight-dtype "${CODEC_WEIGHT_DTYPE}" \
    --codec-compute-dtype "${CODEC_COMPUTE_DTYPE}" \
    --codec-attn-implementation "${CODEC_ATTN_IMPLEMENTATION}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"
else
  accelerate launch moss_tts_local_v1.5/finetuning/sft.py \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --codec-weight-dtype "${CODEC_WEIGHT_DTYPE}" \
    --codec-compute-dtype "${CODEC_COMPUTE_DTYPE}" \
    --codec-attn-implementation "${CODEC_ATTN_IMPLEMENTATION}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"
fi
