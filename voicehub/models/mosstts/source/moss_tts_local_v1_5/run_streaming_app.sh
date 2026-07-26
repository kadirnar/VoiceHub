#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5}"
CODEC_DIR="${CODEC_DIR:-OpenMOSS-Team/MOSS-Audio-Tokenizer-v2}"
DEVICE="${DEVICE:-cuda}"
TTS_DEVICE="${TTS_DEVICE:-cuda:0}"
CODEC_DEVICE="${CODEC_DEVICE:-cuda:0}"
TTS_DTYPE="${TTS_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CODEC_WEIGHT_DTYPE="${CODEC_WEIGHT_DTYPE:-fp32}"
CODEC_COMPUTE_DTYPE="${CODEC_COMPUTE_DTYPE:-bf16}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/moss_tts_local_v1_5_streaming}"
UPLOAD_DIR="${UPLOAD_DIR:-outputs/moss_tts_local_v1_5_uploads}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7861}"

export MODEL_DIR CODEC_DIR DEVICE TTS_DEVICE CODEC_DEVICE TTS_DTYPE ATTN_IMPLEMENTATION CODEC_WEIGHT_DTYPE CODEC_COMPUTE_DTYPE OUTPUT_DIR UPLOAD_DIR HOST PORT

python "$(dirname "$0")/../clis/moss_tts_local_v1.5_app.py" "$@"
