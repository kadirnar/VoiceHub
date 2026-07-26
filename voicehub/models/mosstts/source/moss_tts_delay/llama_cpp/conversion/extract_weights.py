#!/usr/bin/env python3
"""
Extract MOSS-TTS-Delay weights into three groups for llama.cpp porting:

  1. Qwen3 backbone  → standalone Qwen3ForCausalLM (safetensors + config.json)
  2. Embedding tables → numpy .npy files
  3. LM head weights  → numpy .npy files

Usage:
  python moss_tts_delay/llama_cpp/conversion/extract_weights.py \
    --model OpenMOSS-Team/MOSS-TTS \
    --output weights/extracted

The script uses safetensors lazy loading so peak memory ≈ one shard at a time.
"""

import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Name mapping: MossTTSDelay → Qwen3ForCausalLM ──────────────────────────

def remap_backbone_name(name: str) -> str | None:
    """Map a MossTTSDelay tensor name to Qwen3ForCausalLM convention.

    Returns None if the tensor does not belong to the backbone.
    """
    if name.startswith("language_model."):
        # language_model.embed_tokens.weight → model.embed_tokens.weight
        # language_model.layers.N.*         → model.layers.N.*
        # language_model.norm.weight        → model.norm.weight
        return "model." + name[len("language_model."):]
    if name == "lm_heads.0.weight":
        return "lm_head.weight"
    return None

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_source_index(model_dir: Path) -> dict:
    """Load safetensors weight map from model directory."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {
                "metadata": {},
                "weight_map": {k: "model.safetensors" for k in f.keys()},
            }
    raise FileNotFoundError(f"No safetensors files found in {model_dir}")


def load_source_config(model_dir: Path) -> dict:
    with open(model_dir / "config.json") as f:
        return json.load(f)


def build_qwen3_config(moss_config: dict) -> dict:
    """Build a Qwen3ForCausalLM config.json from the MOSS-TTS config."""
    lang = dict(moss_config["language_config"])
    lang["architectures"] = ["Qwen3ForCausalLM"]
    lang["model_type"] = "qwen3"
    lang.pop("_name_or_path", None)
    lang.setdefault("torch_dtype", "bfloat16")
    lang.setdefault("transformers_version", moss_config.get("transformers_version", "4.57.1"))
    return lang


MAX_SHARD_SIZE = 5 * 1024**3  # 5 GB per output shard

# ── Main extraction ─────────────────────────────────────────────────────────

def extract(model_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    backbone_dir = output_dir / "qwen3_backbone"
    backbone_dir.mkdir(exist_ok=True)
    embed_dir = output_dir / "embeddings"
    embed_dir.mkdir(exist_ok=True)
    head_dir = output_dir / "lm_heads"
    head_dir.mkdir(exist_ok=True)

    moss_config = load_source_config(model_dir)
    index = load_source_index(model_dir)
    weight_map = index["weight_map"]

    lang_config = moss_config["language_config"]
    n_vq = moss_config.get("n_vq", 32)
    hidden_size = lang_config["hidden_size"]
    vocab_size = lang_config["vocab_size"]
    audio_vocab_size = moss_config.get("audio_vocab_size", 1024)

    log.info(
        "Model: hidden_size=%d, vocab_size=%d, n_vq=%d, audio_vocab_size=%d",
        hidden_size, vocab_size, n_vq, audio_vocab_size,
    )

    # Group source tensors by shard file
    shard_to_tensors: dict[str, list[str]] = defaultdict(list)
    for tensor_name, shard_file in weight_map.items():
        shard_to_tensors[shard_file].append(tensor_name)

    # ── Pass: iterate shards and extract all three groups ───────────────
    backbone_tensors: dict[str, torch.Tensor] = {}
    backbone_size = 0
    shard_idx = 0
    saved_shards: list[str] = []
    backbone_weight_map: dict[str, str] = {}

    def flush_backbone_shard():
        nonlocal backbone_tensors, backbone_size, shard_idx
        if not backbone_tensors:
            return
        shard_idx += 1
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        shard_path = backbone_dir / shard_name
        log.info("  Writing backbone shard %s (%d tensors, %.2f GB)",
                 shard_name, len(backbone_tensors), backbone_size / 1e9)
        save_file(backbone_tensors, str(shard_path))
        for tname in backbone_tensors:
            backbone_weight_map[tname] = shard_name
        saved_shards.append(shard_name)
        backbone_tensors = {}
        backbone_size = 0

    sorted_shards = sorted(shard_to_tensors.keys())
    for shard_file in sorted_shards:
        tensor_names = shard_to_tensors[shard_file]
        shard_path = model_dir / shard_file
        log.info("Processing shard: %s (%d tensors)", shard_file, len(tensor_names))

        with safe_open(str(shard_path), framework="pt") as sf:
            for tname in sorted(tensor_names):
                tensor = sf.get_tensor(tname)

                # ── Embeddings → npy ────────────────────────────────
                if tname == "language_model.embed_tokens.weight":
                    npy_path = embed_dir / "embed_tokens.npy"
                    np.save(str(npy_path), tensor.to(torch.float16).numpy())
                    log.info("  Saved %s → %s  shape=%s", tname, npy_path.name, list(tensor.shape))

                if tname.startswith("emb_ext.") and tname.endswith(".weight"):
                    idx = int(tname.split(".")[1])
                    npy_path = embed_dir / f"emb_ext_{idx:02d}.npy"
                    np.save(str(npy_path), tensor.to(torch.float16).numpy())
                    log.info("  Saved %s → %s  shape=%s", tname, npy_path.name, list(tensor.shape))

                # ── LM heads → npy ──────────────────────────────────
                if tname.startswith("lm_heads.") and tname.endswith(".weight"):
                    head_idx = int(tname.split(".")[1])
                    if head_idx == 0:
                        npy_path = head_dir / "lm_head_text.npy"
                    else:
                        npy_path = head_dir / f"lm_head_audio_{head_idx - 1:02d}.npy"
                    np.save(str(npy_path), tensor.to(torch.float16).numpy())
                    log.info("  Saved %s → %s  shape=%s", tname, npy_path.name, list(tensor.shape))

                # ── Backbone → Qwen3ForCausalLM safetensors ─────────
                qwen_name = remap_backbone_name(tname)
                if qwen_name is not None:
                    tensor_bytes = tensor.nelement() * tensor.element_size()
                    if backbone_size + tensor_bytes > MAX_SHARD_SIZE and backbone_tensors:
                        flush_backbone_shard()
                    backbone_tensors[qwen_name] = tensor
                    backbone_size += tensor_bytes

    flush_backbone_shard()

    # Fix shard filenames with correct total count
    total_shards = len(saved_shards)
    renamed_shards = []
    for i, old_name in enumerate(saved_shards, 1):
        new_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old_name != new_name:
            (backbone_dir / old_name).rename(backbone_dir / new_name)
        renamed_shards.append(new_name)
        for tname in list(backbone_weight_map.keys()):
            if backbone_weight_map[tname] == old_name:
                backbone_weight_map[tname] = new_name

    # Write backbone index
    total_size = 0
    for shard_name in renamed_shards:
        total_size += (backbone_dir / shard_name).stat().st_size

    backbone_index = {
        "metadata": {"total_size": total_size},
        "weight_map": backbone_weight_map,
    }
    if total_shards > 1:
        with open(backbone_dir / "model.safetensors.index.json", "w") as f:
            json.dump(backbone_index, f, indent=2, sort_keys=True)
        log.info("Wrote backbone index: %d shards, %.2f GB total", total_shards, total_size / 1e9)
    elif total_shards == 1:
        # Single shard: rename to model.safetensors (no index needed)
        single = backbone_dir / renamed_shards[0]
        target = backbone_dir / "model.safetensors"
        if single != target:
            single.rename(target)
        log.info("Wrote single backbone shard: %.2f GB", total_size / 1e9)

    # Write Qwen3 config
    qwen3_config = build_qwen3_config(moss_config)
    with open(backbone_dir / "config.json", "w") as f:
        json.dump(qwen3_config, f, indent=2)
    log.info("Wrote backbone config.json")

    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json",
        "merges.txt", "vocab.json",
    ]
    copied = 0
    for tf in tokenizer_files:
        src = model_dir / tf
        if src.exists():
            shutil.copy2(str(src), str(backbone_dir / tf))
            copied += 1
    log.info("Copied %d tokenizer files to backbone dir", copied)

    # Save extraction metadata
    meta = {
        "source_model": str(model_dir),
        "n_vq": n_vq,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size,
        "audio_vocab_size": audio_vocab_size,
        "backbone_dir": str(backbone_dir),
        "embedding_dir": str(embed_dir),
        "lm_head_dir": str(head_dir),
        "moss_config": moss_config,
    }
    with open(output_dir / "extraction_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Wrote extraction_meta.json")

    # Summary
    embed_files = sorted(embed_dir.glob("*.npy"))
    head_files = sorted(head_dir.glob("*.npy"))
    log.info("=" * 60)
    log.info("Extraction complete!")
    log.info("  Backbone:    %s (%d shards)", backbone_dir, total_shards)
    log.info("  Embeddings:  %s (%d files)", embed_dir, len(embed_files))
    log.info("  LM heads:    %s (%d files)", head_dir, len(head_files))
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MOSS-TTS-Delay weights into three groups"
    )
    parser.add_argument(
        "--model", type=str, default="OpenMOSS-Team/MOSS-TTS",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--output", type=str, default="weights/extracted",
        help="Output directory for extracted weights",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="HuggingFace cache directory for model download",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.is_dir() and (model_path / "config.json").exists():
        model_dir = model_path
        log.info("Using local model directory: %s", model_dir)
    else:
        log.info("Downloading model from HuggingFace: %s", args.model)
        model_dir = Path(snapshot_download(
            args.model,
            cache_dir=args.cache_dir,
            ignore_patterns=["*.md", "*.py", "*.jinja", "__pycache__"],
        ))
        log.info("Model downloaded to: %s", model_dir)

    extract(model_dir, Path(args.output))


if __name__ == "__main__":
    main()
