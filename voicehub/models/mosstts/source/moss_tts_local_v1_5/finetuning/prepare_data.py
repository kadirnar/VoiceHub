from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import AutoProcessor

from common import (
    dump_jsonl,
    format_timestamp,
    load_jsonl,
    normalize_audio_path_list,
    resolve_record_audio_paths,
    resolve_shard_spec,
    select_rank_shard,
    shard_output_path,
)


DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute MOSS-TTS Local v1.5 audio codes for supervised finetuning."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--codec-path", type=str, default=DEFAULT_CODEC_PATH)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Codec device. Use `auto` to follow the current Accelerate rank device.",
    )
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--n-vq",
        type=int,
        default=None,
        help="Must match the v1.5 model config. Leave unset for the checkpoint default, normally 12.",
    )
    parser.add_argument(
        "--codec-weight-dtype",
        type=str,
        default="fp32",
        choices=["fp32"],
        help="Codec weight loading dtype policy. The public v1.5 finetuning path supports fp32 only.",
    )
    parser.add_argument(
        "--codec-compute-dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Codec inference compute dtype. This does not change the persisted audio_codes.",
    )
    parser.add_argument(
        "--codec-attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument(
        "--skip-reference-audio-codes",
        dest="encode_reference_audio",
        action="store_false",
        help="Only encode target `audio`; skip reference audio fields.",
    )
    parser.add_argument("--save-shard-suffix", action="store_true")
    parser.set_defaults(encode_reference_audio=True)
    return parser.parse_args()


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name or "bf16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name!r}")


def resolve_attn_implementation(requested: str, device: str, dtype: torch.dtype) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized not in {"auto", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unsupported attention implementation: {requested!r}")
    if normalized != "auto":
        return normalized

    device_obj = torch.device(device)
    if (
        device_obj.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        try:
            major, _ = torch.cuda.get_device_capability(device_obj)
            if major >= 8:
                return "flash_attention_2"
        except Exception:
            pass
    return "sdpa" if device_obj.type == "cuda" else "eager"


def build_processor(args: argparse.Namespace, device: str):
    codec_compute_dtype = _resolve_dtype(args.codec_compute_dtype)
    codec_attn_implementation = resolve_attn_implementation(
        args.codec_attn_implementation,
        device=device,
        dtype=codec_compute_dtype,
    )
    processor = load_processor_with_codec_kwargs(
        args.model_path,
        codec_path=args.codec_path,
        codec_weight_dtype=args.codec_weight_dtype,
        codec_compute_dtype=args.codec_compute_dtype,
        codec_attention_implementation=codec_attn_implementation,
    )
    if getattr(processor, "audio_tokenizer", None) is None:
        raise RuntimeError("Loaded processor has no audio_tokenizer.")
    configure_audio_tokenizer(
        processor.audio_tokenizer,
        codec_compute_dtype=args.codec_compute_dtype,
        codec_attention_implementation=codec_attn_implementation,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)
    if hasattr(processor.audio_tokenizer, "eval"):
        processor.audio_tokenizer.eval()
    return processor, codec_attn_implementation


def load_processor_with_codec_kwargs(
    model_path: str,
    *,
    codec_path: str,
    codec_weight_dtype: str,
    codec_compute_dtype: str,
    codec_attention_implementation: str,
):
    try:
        return AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            codec_path=codec_path,
            codec_weight_dtype=codec_weight_dtype,
            codec_compute_dtype=codec_compute_dtype,
            codec_attention_implementation=codec_attention_implementation,
        )
    except TypeError as exc:
        message = str(exc)
        unsupported_codec_kwarg = any(
            key in message
            for key in ("codec_weight_dtype", "codec_compute_dtype", "codec_attention_implementation")
        )
        if not unsupported_codec_kwarg:
            raise
        return AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            codec_path=codec_path,
        )


def configure_audio_tokenizer(
    audio_tokenizer,
    *,
    codec_compute_dtype: str,
    codec_attention_implementation: str,
) -> None:
    if hasattr(audio_tokenizer, "set_compute_dtype"):
        audio_tokenizer.set_compute_dtype(codec_compute_dtype)
    if hasattr(audio_tokenizer, "set_attention_implementation"):
        audio_tokenizer.set_attention_implementation(codec_attention_implementation)


@torch.inference_mode()
def batch_encode(
    processor,
    paths: List[str],
    batch_size: int,
    n_vq: Optional[int],
    desc: str,
) -> List[torch.Tensor]:
    all_codes: List[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=desc):
        batch_paths = paths[start : start + batch_size]
        all_codes.extend(processor.encode_audios_from_path(batch_paths, n_vq=n_vq))
    return all_codes


def collect_paths(records: List[Dict[str, Any]], field_name: str) -> List[str]:
    paths: List[str] = []
    for record in records:
        values = normalize_audio_path_list(
            record.get(field_name),
            field_name,
            allow_none=(field_name == "reference"),
        )
        if values is not None:
            paths.extend(value for value in values if value is not None)
    return list(dict.fromkeys(paths))


def collect_reference_paths(records: List[Dict[str, Any]]) -> List[str]:
    unique_paths: List[str] = []
    for field_name in ("ref_audio", "reference_audio", "reference"):
        unique_paths.extend(collect_paths(records, field_name))
    return list(dict.fromkeys(unique_paths))


def attach_reference_audio_codes(
    records: List[Dict[str, Any]],
    path_to_codes: Dict[str, List[List[int]]],
) -> None:
    for record in records:
        ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if ref_audio is not None and record.get("ref_audio_codes") is None:
            if len(ref_audio) != 1:
                raise ValueError("`ref_audio` only supports a single path.")
            record["ref_audio_codes"] = path_to_codes[ref_audio[0]]

        reference_audio = normalize_audio_path_list(record.get("reference_audio"), "reference_audio")
        if reference_audio is not None and record.get("reference_audio_codes") is None:
            record["reference_audio_codes"] = [path_to_codes[path] for path in reference_audio]

        reference = normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
        if reference is not None and record.get("reference_audio_codes") is None:
            record["reference_audio_codes"] = [
                None if path is None else path_to_codes[path]
                for path in reference
            ]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("`batch_size` must be > 0.")
    if args.n_vq is not None and args.n_vq <= 0:
        raise ValueError("`n_vq` must be > 0 when provided.")

    accelerator = Accelerator()
    device = str(accelerator.device) if args.device == "auto" else args.device

    input_jsonl_path = Path(args.input_jsonl).resolve()
    all_records = [
        resolve_record_audio_paths(record, base_dir=input_jsonl_path.parent)
        for record in load_jsonl(input_jsonl_path)
    ]
    world_size, rank = resolve_shard_spec(
        args.num_shards,
        args.shard_rank,
        default_num_shards=accelerator.num_processes,
        default_shard_rank=accelerator.process_index,
    )
    records = select_rank_shard(all_records, world_size, rank)
    if not records:
        raise ValueError(
            f"No records found for shard rank={rank} / world_size={world_size} in {input_jsonl_path}."
        )

    processor, codec_attn_implementation = build_processor(args, device=device)

    target_audio_paths: List[str] = []
    target_records: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if record.get("audio_codes") is not None:
            continue
        audio_path = record.get("audio")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Record {index} is missing a valid `audio` field.")
        target_audio_paths.append(audio_path)
        target_records.append(record)

    if target_audio_paths:
        target_audio_codes = batch_encode(
            processor=processor,
            paths=target_audio_paths,
            batch_size=args.batch_size,
            n_vq=args.n_vq,
            desc="Encoding target audio",
        )
        for record, codes in zip(target_records, target_audio_codes):
            record["audio_codes"] = codes.tolist()

    if args.encode_reference_audio:
        unique_reference_paths = collect_reference_paths(records)
        if unique_reference_paths:
            reference_codes = batch_encode(
                processor=processor,
                paths=unique_reference_paths,
                batch_size=args.batch_size,
                n_vq=args.n_vq,
                desc="Encoding reference audio",
            )
            reference_code_map = {
                path: codes.tolist()
                for path, codes in zip(unique_reference_paths, reference_codes)
            }
            attach_reference_audio_codes(records, reference_code_map)

    output_path = Path(args.output_jsonl)
    if world_size > 1 or args.save_shard_suffix:
        output_path = shard_output_path(output_path, rank, world_size)
    dump_jsonl(records, output_path)
    accelerator.print(
        f"[{format_timestamp()}] [prepare_data] rank={rank}/{world_size} "
        f"input_records={len(all_records)} local_records={len(records)} "
        f"device={device} codec_attn={codec_attn_implementation} "
        f"codec_weight_dtype={args.codec_weight_dtype} codec_compute_dtype={args.codec_compute_dtype} "
        f"output={output_path}"
    )


if __name__ == "__main__":
    main()
