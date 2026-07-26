from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import math
import re
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedType, set_seed
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor, get_scheduler
from transformers.utils import cached_file

try:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        CheckpointWrapper,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    _CHECKPOINT_WRAPPER_AVAILABLE = True
except Exception:
    CheckpointImpl = None
    CheckpointWrapper = None
    apply_activation_checkpointing = None
    checkpoint_wrapper = None
    _CHECKPOINT_WRAPPER_AVAILABLE = False

FINETUNING_DIR = Path(__file__).resolve().parent
MODULE_ROOT = FINETUNING_DIR.parent
REPO_ROOT = MODULE_ROOT.parent
if str(FINETUNING_DIR) not in sys.path:
    sys.path.insert(0, str(FINETUNING_DIR))

from common import format_duration, format_timestamp, load_jsonl, normalize_audio_path_list, resolve_jsonl_paths
from dataset import MossTTSLocalV15SFTDataset


DEFAULT_MODEL_PATH = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_CODEC_PATH = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"

SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)

SUPPORT_FILES = (
    MODULE_ROOT / "__init__.py",
    MODULE_ROOT / "configuration_moss_tts.py",
    MODULE_ROOT / "gpt2_decoder.py",
    MODULE_ROOT / "modeling_moss_tts.py",
    MODULE_ROOT / "processing_moss_tts.py",
    MODULE_ROOT / "qwen3_decoder.py",
)

INFERENCE_ASSET_FILES = (
    "__init__.py",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

BASE_PROCESSOR_CONFIG = {
    "processor_class": "MossTTSLocalProcessor",
    "auto_map": {
        "AutoProcessor": "processing_moss_tts.MossTTSLocalProcessor",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised finetuning for MOSS-TTS Local Transformer v1.5."
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--codec-path", type=str, default=DEFAULT_CODEC_PATH)
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="A single JSONL, directory, glob, or comma-separated list of JSONLs produced by prepare_data.py.",
    )
    parser.add_argument("--output-dir", type=str, default="output/moss_tts_local_v1_5_sft")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument(
        "--adam-eps",
        type=float,
        default=1e-4,
        help="Adam epsilon. A larger default is used because full BF16 parameter finetuning is unstable with 1e-8.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", type=str, default="constant", choices=SCHEDULER_CHOICES)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument("--audio-tokenizer-device", type=str, default=None)
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--gradient-checkpointing-scope",
        type=str,
        default="global",
        choices=["all", "global", "local"],
        help=(
            "Which transformer stack should use activation checkpointing. "
            "`global` means the main Qwen3 decoder; `local` means the RVQ local decoder."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing-impl",
        type=str,
        default="module_wrapper",
        choices=["module_wrapper", "model_internal"],
        help=(
            "Activation checkpointing implementation. `module_wrapper` wraps whole decoder "
            "layers with torch checkpoint_wrapper, matching Nano-TTS and avoiding the "
            "model-internal Qwen3 checkpoint closure."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing-use-reentrant",
        action="store_true",
        help="Use PyTorch reentrant activation checkpointing instead of the non-reentrant variant.",
    )
    parser.add_argument(
        "--channelwise-loss-weight",
        type=str,
        default="1,32",
        help=(
            "Comma-separated loss weights. Use either n_vq+1 values "
            "(text_head,vq0,...,vqN) or two values "
            "(text_weight,total_audio_weight). When two values are given, "
            "the total audio weight is evenly distributed across all audio heads."
        ),
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
        help="Codec inference compute dtype for on-the-fly reference audio encoding.",
    )
    parser.add_argument(
        "--codec-attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--skip-nonfinite-batches",
        action="store_true",
        help="Skip batches with non-finite loss or gradient norm instead of raising an error.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.per_device_batch_size <= 0:
        raise ValueError("`per_device_batch_size` must be > 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be > 0.")
    if args.learning_rate <= 0:
        raise ValueError("`learning_rate` must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("`weight_decay` must be >= 0.")
    if args.warmup_steps < 0:
        raise ValueError("`warmup_steps` must be >= 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("`warmup_ratio` must be in [0, 1).")
    if args.num_epochs <= 0:
        raise ValueError("`num_epochs` must be > 0.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("`max_train_steps` must be > 0 when provided.")
    if args.max_grad_norm < 0:
        raise ValueError("`max_grad_norm` must be >= 0.")
    if args.logging_steps <= 0:
        raise ValueError("`logging_steps` must be > 0.")
    if args.save_every_epochs <= 0:
        raise ValueError("`save_every_epochs` must be > 0.")
    if args.num_workers < 0:
        raise ValueError("`num_workers` must be >= 0.")
    if args.n_vq is not None and args.n_vq <= 0:
        raise ValueError("`n_vq` must be > 0 when provided.")


def configure_torch_backends() -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def set_rank_cuda_device(accelerator: Accelerator) -> None:
    if not torch.cuda.is_available() or accelerator.device.type != "cuda":
        return
    device_index = accelerator.local_process_index % torch.cuda.device_count()
    torch.cuda.set_device(device_index)


def resolve_torch_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_accelerate_mixed_precision(mixed_precision: str) -> str:
    if not torch.cuda.is_available():
        return "no"
    return mixed_precision


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name or "bf16").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name!r}")


def resolve_attn_implementation(requested: str, device: torch.device, dtype: torch.dtype) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized in {"flash", "flash_attn", "flash-attn", "flash_attention"}:
        normalized = "flash_attention_2"
    if normalized not in {"auto", "flash_attention_2", "sdpa", "eager"}:
        raise ValueError(f"Unsupported attention implementation: {requested!r}")
    if normalized != "auto":
        return normalized

    if (
        device.type == "cuda"
        and importlib.util.find_spec("flash_attn") is not None
        and dtype in {torch.float16, torch.bfloat16}
    ):
        try:
            major, _ = torch.cuda.get_device_capability(device)
            if major >= 8:
                return "flash_attention_2"
        except Exception:
            pass
    return "sdpa" if device.type == "cuda" else "eager"


def resolve_warmup_steps(args: argparse.Namespace, num_training_steps: int) -> int:
    if args.warmup_steps > 0:
        return args.warmup_steps
    if args.warmup_ratio > 0:
        return math.ceil(num_training_steps * args.warmup_ratio)
    return 0


def parse_channelwise_loss_weight(spec: Optional[str], n_heads: int) -> Optional[List[float]]:
    if spec is None:
        return None
    values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    if not values:
        return None
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        per_audio_weight = total_audio_weight / max(1, n_heads - 1)
        resolved = [text_weight] + [per_audio_weight] * (n_heads - 1)
    else:
        raise ValueError(
            f"`channelwise_loss_weight` expects either {n_heads} values or 2 values, got {len(values)}."
        )
    if sum(resolved) <= 0:
        raise ValueError("`channelwise_loss_weight` must sum to a positive value.")
    return resolved


def processor_needs_audio_tokenizer(records: List[Dict[str, Any]]) -> bool:
    for record in records:
        ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if record.get("ref_audio_codes") is None and ref_audio is not None:
            return True
        if record.get("reference_audio_codes") is None:
            reference = normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
            if reference is not None and any(item is not None for item in reference):
                return True
            reference_audio = normalize_audio_path_list(record.get("reference_audio"), "reference_audio")
            if reference_audio is not None:
                return True
    return False


@contextmanager
def processor_init_context(accelerator: Accelerator):
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        yield
        return

    plugin = accelerator.state.deepspeed_plugin
    if plugin is None or not plugin.is_zero3_init_enabled():
        yield
        return

    import deepspeed

    with plugin.zero3_init_context_manager(enable=False):
        deepspeed.zero.partition_parameters.shutdown_init_context()
        try:
            yield
        finally:
            deepspeed.zero.partition_parameters.restore_init_context()


@contextmanager
def transformers_zero3_load_context(accelerator: Accelerator):
    if accelerator.distributed_type != DistributedType.DEEPSPEED:
        yield
        return

    plugin = accelerator.state.deepspeed_plugin
    if plugin is None or int(getattr(plugin, "zero_stage", 0) or 0) != 3:
        yield
        return

    import transformers.integrations.deepspeed as hf_deepspeed

    old_ref = getattr(hf_deepspeed, "_hf_deepspeed_config_weak_ref", None)
    hf_deepspeed.unset_hf_deepspeed_config()
    try:
        yield
    finally:
        old_config = old_ref() if old_ref is not None else None
        if old_config is not None:
            hf_deepspeed.set_hf_deepspeed_config(old_config)


@contextmanager
def model_init_context(accelerator: Accelerator):
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        plugin = accelerator.state.deepspeed_plugin
        if plugin is not None and plugin.is_zero3_init_enabled():
            with transformers_zero3_load_context(accelerator):
                with plugin.zero3_init_context_manager(enable=False):
                    yield
            return

    with transformers_zero3_load_context(accelerator):
        yield


def build_processor(
    args: argparse.Namespace,
    *,
    need_audio_tokenizer: bool,
    default_audio_tokenizer_device: str,
) -> Any:
    codec_compute_dtype = _resolve_dtype(args.codec_compute_dtype)
    codec_device = torch.device(args.audio_tokenizer_device or default_audio_tokenizer_device)
    codec_attn_implementation = resolve_attn_implementation(
        args.codec_attn_implementation,
        device=codec_device,
        dtype=codec_compute_dtype,
    )
    processor = load_processor_with_codec_kwargs(
        args.model_path,
        codec_path=args.codec_path,
        codec_weight_dtype=args.codec_weight_dtype,
        codec_compute_dtype=args.codec_compute_dtype,
        codec_attention_implementation=codec_attn_implementation,
        load_audio_tokenizer=need_audio_tokenizer,
    )
    if need_audio_tokenizer:
        if getattr(processor, "audio_tokenizer", None) is None:
            raise RuntimeError("Loaded processor has no audio_tokenizer.")
        configure_audio_tokenizer(
            processor.audio_tokenizer,
            codec_compute_dtype=args.codec_compute_dtype,
            codec_attention_implementation=codec_attn_implementation,
        )
        processor.audio_tokenizer = processor.audio_tokenizer.to(codec_device)
        if hasattr(processor.audio_tokenizer, "eval"):
            processor.audio_tokenizer.eval()
    else:
        processor.audio_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return processor


def load_processor_with_codec_kwargs(
    model_path: str,
    *,
    codec_path: str,
    codec_weight_dtype: str,
    codec_compute_dtype: str,
    codec_attention_implementation: str,
    load_audio_tokenizer: bool,
):
    def model_load_context():
        if load_audio_tokenizer:
            return nullcontext()
        # The processor's from_pretrained implementation always loads the codec.
        # SFT with precomputed audio codes only needs tokenizer/config. Avoid
        # constructing the codec under ZeRO-3, where its weight-norm parameters
        # can be temporarily partitioned to empty tensors during module init.
        return patch.object(AutoModel, "from_pretrained", return_value=None)

    try:
        with model_load_context():
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
        with model_load_context():
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


def load_training_model(
    *,
    accelerator: Accelerator,
    model_path: str,
    model_dtype: torch.dtype,
    attn_implementation: str,
):
    with model_init_context(accelerator):
        return AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=model_dtype,
            attn_implementation=attn_implementation,
        )


def unwrap_training_model(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def gradient_checkpointing_modules(scope: str) -> set[str]:
    normalized = str(scope or "global").strip().lower()
    if normalized == "all":
        return {"global", "local"}
    if normalized not in {"global", "local"}:
        raise ValueError(f"Unsupported gradient checkpointing scope: {scope!r}")
    return {normalized}


def set_gradient_checkpointing_flag(module: torch.nn.Module, value: bool) -> None:
    if hasattr(module, "gradient_checkpointing"):
        module.gradient_checkpointing = value
    for child in module.children():
        set_gradient_checkpointing_flag(child, value)


def apply_activation_checkpoint_wrapper_to_model(
    model: torch.nn.Module,
    *,
    scope: str,
    use_reentrant: bool,
) -> None:
    if not _CHECKPOINT_WRAPPER_AVAILABLE:
        raise ImportError("Activation checkpointing requires torch checkpoint_wrapper support.")

    modules = gradient_checkpointing_modules(scope)
    target_global = "global" in modules
    target_local = "local" in modules
    target_class_names = set()
    if target_global:
        target_class_names.update({"MossQwen3DecoderLayer", "NanoQwen3DecoderLayer"})
    if target_local:
        target_class_names.update({"MossTTSNanoGPT2Block", "NanoGPT2Block"})

    def check_fn(submodule: torch.nn.Module) -> bool:
        if CheckpointWrapper is not None and isinstance(submodule, CheckpointWrapper):
            return False
        return submodule.__class__.__name__ in target_class_names

    checkpoint_impl = (
        CheckpointImpl.REENTRANT
        if use_reentrant
        else CheckpointImpl.NO_REENTRANT
    )
    wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=checkpoint_impl,
    )
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=check_fn,
    )


def configure_gradient_checkpointing(
    model,
    *,
    scope: str,
    use_reentrant: bool,
    impl: str,
) -> None:
    base_model = unwrap_training_model(model)
    stacks = {
        "global": getattr(base_model, "transformer", None),
        "local": getattr(base_model, "local_transformer", None),
    }
    for name in gradient_checkpointing_modules(scope):
        if stacks.get(name) is None:
            raise ValueError(f"Cannot enable {name} gradient checkpointing; model stack not found.")

    set_gradient_checkpointing_flag(base_model, False)
    for stack in stacks.values():
        if stack is not None and hasattr(stack, "gradient_checkpointing_use_reentrant"):
            stack.gradient_checkpointing_use_reentrant = bool(use_reentrant)

    if impl == "module_wrapper":
        apply_activation_checkpoint_wrapper_to_model(
            base_model,
            scope=scope,
            use_reentrant=use_reentrant,
        )
        return

    if impl != "model_internal":
        raise ValueError(f"Unsupported gradient checkpointing implementation: {impl!r}")

    enabled = ("global", "local") if scope == "all" else (scope,)
    for name in enabled:
        stack = stacks.get(name)
        if not hasattr(stack, "gradient_checkpointing"):
            raise ValueError(f"Cannot enable {name} gradient checkpointing; stack has no flag.")
        stack.gradient_checkpointing = True


def compute_supervised_loss_from_hidden(
    base_model,
    *,
    global_hidden_states: torch.FloatTensor,
    labels: torch.LongTensor,
    channelwise_loss_weight: Optional[List[float]],
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = global_hidden_states.shape
    n_vq = int(base_model.config.n_vq)
    if labels.shape[-1] != n_vq + 1:
        raise ValueError(f"Expected labels with {n_vq + 1} channels, got {labels.shape[-1]}.")

    weights = channelwise_loss_weight or [1.0] * (n_vq + 1)
    if len(weights) != n_vq + 1:
        raise ValueError(f"`channelwise_loss_weight` length {len(weights)} != {n_vq + 1}.")

    flat_hidden = global_hidden_states.reshape(batch_size * seq_len, hidden_size)
    flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)
    local_dtype = base_model.local_transformer.ln_f.weight.dtype
    local_prefix_hidden = base_model._global_hidden_to_local(flat_hidden).to(dtype=local_dtype)
    local_inputs = torch.zeros(
        (batch_size * seq_len, n_vq, int(local_prefix_hidden.shape[-1])),
        dtype=local_dtype,
        device=flat_hidden.device,
    )
    local_inputs[:, 0, :] = local_prefix_hidden

    audio_targets = flat_labels[:, 1:]
    for channel_index in range(n_vq - 1):
        teacher_ids = audio_targets[:, channel_index]
        embedding = base_model.audio_embeddings[channel_index]
        valid_mask = (teacher_ids >= 0) & (teacher_ids < embedding.num_embeddings)
        safe_ids = teacher_ids.masked_fill(~valid_mask, 0)
        channel_embeds = embedding(safe_ids).to(dtype=local_dtype)
        channel_embeds = channel_embeds * valid_mask.unsqueeze(-1)
        local_inputs[:, channel_index + 1, :] = channel_embeds

    local_outputs = base_model.local_transformer(
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=local_inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cu_seqlens=None,
        num_sequences=None,
    )
    local_hidden_states = local_outputs.last_hidden_state

    total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
    total_weight = 0.0
    text_targets = flat_labels[:, 0]

    if (
        hasattr(base_model, "_use_binary_local_text_head")
        and base_model._use_binary_local_text_head()
        and getattr(base_model, "local_text_lm_head", None) is not None
    ):
        text_logits = base_model.local_text_lm_head(local_hidden_states[:, 0, :])
        binary_targets = torch.full_like(text_targets, -100)
        binary_targets = binary_targets.masked_fill(
            text_targets.eq(int(base_model.config.audio_assistant_slot_token_id)),
            0,
        )
        binary_targets = binary_targets.masked_fill(
            text_targets.eq(int(base_model.config.audio_end_token_id)),
            1,
        )
        if (binary_targets != -100).any():
            text_loss = F.cross_entropy(text_logits.float(), binary_targets, ignore_index=-100)
            total_loss = total_loss + float(weights[0]) * text_loss.float()
            total_weight += float(weights[0])
    elif (text_targets != -100).any():
        text_logits = base_model.text_lm_head(local_hidden_states[:, 0, :])
        text_loss = F.cross_entropy(text_logits.float(), text_targets, ignore_index=-100)
        total_loss = total_loss + float(weights[0]) * text_loss.float()
        total_weight += float(weights[0])

    for channel_index in range(n_vq):
        channel_targets = audio_targets[:, channel_index]
        if not (channel_targets != -100).any():
            continue
        channel_logits = base_model.audio_lm_heads[channel_index](
            local_hidden_states[:, channel_index, :]
        )
        channel_loss = F.cross_entropy(channel_logits.float(), channel_targets, ignore_index=-100)
        total_loss = total_loss + float(weights[channel_index + 1]) * channel_loss.float()
        total_weight += float(weights[channel_index + 1])

    if total_weight <= 0:
        raise RuntimeError("All labels are ignored; check dataset packing.")
    return total_loss / total_weight


def compute_supervised_loss(
    model,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    labels: torch.LongTensor,
    channelwise_loss_weight: Optional[List[float]],
) -> torch.Tensor:
    base_model = unwrap_training_model(model)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    global_hidden_states = outputs.last_hidden_state
    if global_hidden_states is None:
        raise RuntimeError("Model forward did not return last_hidden_state.")
    return compute_supervised_loss_from_hidden(
        base_model,
        global_hidden_states=global_hidden_states,
        labels=labels,
        channelwise_loss_weight=channelwise_loss_weight,
    )


def copy_support_files(output_dir: Path) -> None:
    for src in SUPPORT_FILES:
        if src.exists():
            shutil.copy2(src, output_dir / src.name)


def resolve_inference_asset(model_path: str, filename: str) -> Optional[Path]:
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        candidate = model_path_obj / filename
        return candidate if candidate.exists() else None

    try:
        resolved = cached_file(
            model_path,
            filename,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError:
        return None
    return None if resolved is None else Path(resolved)


def copy_inference_assets(model_path: str, codec_path: str, output_dir: Path) -> None:
    for filename in INFERENCE_ASSET_FILES:
        src = resolve_inference_asset(model_path, filename)
        if src is not None and src.exists():
            shutil.copy2(src, output_dir / filename)

    processor_config = dict(BASE_PROCESSOR_CONFIG)
    processor_config["audio_tokenizer_name_or_path"] = codec_path
    with open(output_dir / "processor_config.json", "w", encoding="utf-8") as handle:
        json.dump(processor_config, handle, indent=2, ensure_ascii=False)


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    model_path: str,
    codec_path: str,
    output_dir: Path,
    train_args: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = accelerator.get_state_dict(model)
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=True,
    )

    if accelerator.is_main_process:
        copy_support_files(output_dir)
        copy_inference_assets(model_path, codec_path, output_dir)
        metadata = dict(train_args)
        metadata["saved_global_step"] = int(global_step)
        metadata["saved_epoch"] = int(epoch)
        metadata["saved_at"] = format_timestamp()
        with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
    accelerator.wait_for_everyone()


def shard_paths_for_rank(paths: List[Path], world_size: int, rank: int) -> tuple[List[Path], bool]:
    if world_size <= 1:
        return paths, False

    shard_pattern = re.compile(r"\.rank(\d+)-of-(\d+)\.jsonl$")
    parsed: List[tuple[Path, int, int]] = []
    for path in paths:
        match = shard_pattern.search(path.name)
        if match is None:
            return paths, False
        shard_rank = int(match.group(1))
        shard_world_size = int(match.group(2))
        parsed.append((path, shard_rank, shard_world_size))

    shard_world_sizes = {item[2] for item in parsed}
    if len(shard_world_sizes) != 1:
        return paths, False

    selected = [path for path, shard_rank, _ in parsed if shard_rank % world_size == rank]
    if not selected:
        raise ValueError(
            f"No shard assigned for rank={rank} world_size={world_size}. "
            "Please check --train-jsonl shard files and distributed config."
        )
    return selected, True


def load_jsonl_for_rank(
    spec: str,
    world_size: int,
    rank: int,
) -> tuple[List[Path], List[Dict[str, Any]], List[Path], bool]:
    all_paths = resolve_jsonl_paths(spec)
    rank_paths, using_pre_sharded_files = shard_paths_for_rank(
        all_paths,
        world_size=world_size,
        rank=rank,
    )
    records: List[Dict[str, Any]] = []
    for path in rank_paths:
        records.extend(load_jsonl(path))
    return all_paths, records, rank_paths, using_pre_sharded_files


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def is_finite_scalar(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.detach().float()).all().item())


def first_nonfinite_gradient(model) -> Optional[str]:
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        finite_mask = torch.isfinite(grad.detach())
        if bool(finite_mask.all().item()):
            continue
        num_bad = int((~finite_mask).sum().item())
        num_total = int(grad.numel())
        return f"{name} bad={num_bad}/{num_total} dtype={grad.dtype} shape={tuple(grad.shape)}"
    return None


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_backends()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=resolve_accelerate_mixed_precision(args.mixed_precision),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    set_rank_cuda_device(accelerator)
    set_seed(args.seed, device_specific=True)

    global_micro_batch_size = args.per_device_batch_size * accelerator.num_processes
    global_batch_size = global_micro_batch_size * args.gradient_accumulation_steps
    accelerator.print(
        f"[{format_timestamp()}] [sft] global_batch_size="
        f"{args.per_device_batch_size} x {accelerator.num_processes} x "
        f"{args.gradient_accumulation_steps} = {global_batch_size}"
    )

    train_paths, records, local_train_paths, using_pre_sharded_files = load_jsonl_for_rank(
        args.train_jsonl,
        world_size=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    if not records:
        raise ValueError(f"No records found in {args.train_jsonl}.")
    accelerator.print(
        f"[{format_timestamp()}] [sft] distributed_type={accelerator.distributed_type} "
        f"num_processes={accelerator.num_processes} "
        f"using_pre_sharded_files={using_pre_sharded_files} "
        f"train_files={len(train_paths)} local_train_files={len(local_train_paths)} "
        f"local_train_records={len(records)}"
    )

    need_audio_tokenizer = processor_needs_audio_tokenizer(records)
    if need_audio_tokenizer:
        accelerator.print(
            f"[{format_timestamp()}] [sft] found records without precomputed reference audio codes; "
            "keeping the codec loaded for on-the-fly reference encoding."
        )
    with processor_init_context(accelerator):
        processor = build_processor(
            args,
            need_audio_tokenizer=need_audio_tokenizer,
            default_audio_tokenizer_device=str(accelerator.device),
        )

    dataset = MossTTSLocalV15SFTDataset(
        records=records,
        processor=processor,
        n_vq=args.n_vq,
    )

    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(
        args.attn_implementation,
        device=accelerator.device,
        dtype=model_dtype,
    )
    model = load_training_model(
        accelerator=accelerator,
        model_path=args.model_path,
        model_dtype=model_dtype,
        attn_implementation=attn_implementation,
    )
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing:
        configure_gradient_checkpointing(
            model,
            scope=args.gradient_checkpointing_scope,
            use_reentrant=args.gradient_checkpointing_use_reentrant,
            impl=args.gradient_checkpointing_impl,
        )

    resolved_channelwise_loss_weight = parse_channelwise_loss_weight(
        args.channelwise_loss_weight,
        int(model.config.n_vq) + 1,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] attn={attn_implementation} model_dtype={model_dtype} "
        f"n_vq={model.config.n_vq} "
        f"channelwise_loss_weight={resolved_channelwise_loss_weight} "
        f"gradient_checkpointing={args.gradient_checkpointing} "
        f"gradient_checkpointing_impl={args.gradient_checkpointing_impl} "
        f"gradient_checkpointing_scope={args.gradient_checkpointing_scope} "
        f"gradient_checkpointing_use_reentrant={args.gradient_checkpointing_use_reentrant}"
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        drop_last=using_pre_sharded_files,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )
    local_micro_batches_per_epoch = len(train_dataloader)
    dropped_local_micro_batches_per_epoch = 0
    if using_pre_sharded_files:
        local_micro_batches_tensor = torch.tensor(
            [local_micro_batches_per_epoch],
            device=accelerator.device,
            dtype=torch.long,
        )
        gathered_micro_batches = accelerator.gather(local_micro_batches_tensor)
        micro_batches_per_epoch = int(gathered_micro_batches.min().item())
        micro_batches_per_epoch = (
            micro_batches_per_epoch // args.gradient_accumulation_steps
        ) * args.gradient_accumulation_steps
        if micro_batches_per_epoch <= 0:
            raise ValueError(
                "Pre-sharded training has fewer local micro-batches than "
                "`gradient_accumulation_steps`; reduce gradient accumulation or use larger shards."
            )
        dropped_local_micro_batches_per_epoch = max(
            local_micro_batches_per_epoch - micro_batches_per_epoch,
            0,
        )
    else:
        micro_batches_per_epoch = math.ceil(len(records) / global_micro_batch_size)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        foreach=False,
    )

    update_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * update_steps_per_epoch)
    warmup_steps = resolve_warmup_steps(args, max_train_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    set_rank_cuda_device(accelerator)
    if using_pre_sharded_files:
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

    output_root = Path(args.output_dir)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    train_args_to_save = vars(args).copy()
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["local_micro_batches_per_epoch"] = local_micro_batches_per_epoch
    train_args_to_save["micro_batches_per_epoch"] = micro_batches_per_epoch
    train_args_to_save["dropped_local_micro_batches_per_epoch"] = dropped_local_micro_batches_per_epoch
    train_args_to_save["optimizer_steps_per_epoch"] = update_steps_per_epoch
    train_args_to_save["resolved_warmup_steps"] = warmup_steps
    train_args_to_save["resolved_attn_implementation"] = attn_implementation
    train_args_to_save["resolved_channelwise_loss_weight"] = resolved_channelwise_loss_weight
    train_args_to_save["checkpoint_wrapper_available"] = _CHECKPOINT_WRAPPER_AVAILABLE
    train_args_to_save["records_paths"] = [str(path.resolve()) for path in train_paths]

    accelerator.print(
        f"[{format_timestamp()}] [sft] scheduler={args.lr_scheduler_type} "
        f"warmup_steps={warmup_steps} local_micro_batches_per_epoch={local_micro_batches_per_epoch} "
        f"micro_batches_per_epoch={micro_batches_per_epoch} "
        f"dropped_local_micro_batches_per_epoch={dropped_local_micro_batches_per_epoch} "
        f"optimizer_steps_per_epoch={update_steps_per_epoch} max_train_steps={max_train_steps}"
    )

    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0
    accumulation_record_ids: List[str] = []

    for epoch in range(args.num_epochs):
        model.train()
        for local_micro_batch_index, batch in enumerate(train_dataloader):
            if using_pre_sharded_files and local_micro_batch_index >= micro_batches_per_epoch:
                break
            accumulation_record_ids.extend(str(item) for item in batch.get("record_ids", []))
            batch = move_batch_to_device(batch, accelerator.device)
            with accelerator.accumulate(model):
                loss = compute_supervised_loss(
                    model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    channelwise_loss_weight=resolved_channelwise_loss_weight,
                )
                if not is_finite_scalar(loss):
                    optimizer.zero_grad()
                    message = (
                        f"Non-finite loss at epoch={epoch} global_step={global_step}. "
                        "This usually indicates unstable BF16 training or an invalid batch. "
                        f"record_ids={accumulation_record_ids}"
                    )
                    if args.skip_nonfinite_batches:
                        accelerator.print(f"[{format_timestamp()}] [sft] warning: {message}; skipped")
                        accumulation_record_ids.clear()
                        continue
                    raise FloatingPointError(message)
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if torch.is_tensor(grad_norm) and not is_finite_scalar(grad_norm):
                        bad_grad = first_nonfinite_gradient(model)
                        optimizer.zero_grad()
                        message = (
                            f"Non-finite gradient norm at epoch={epoch} global_step={global_step}: "
                            f"{grad_norm.detach().float().item()}. "
                            f"first_nonfinite_grad={bad_grad}; "
                            f"record_ids={accumulation_record_ids}"
                        )
                        if args.skip_nonfinite_batches:
                            accelerator.print(f"[{format_timestamp()}] [sft] warning: {message}; skipped")
                            accumulation_record_ids.clear()
                            continue
                        raise FloatingPointError(message)

                optimizer.step()
                if accelerator.sync_gradients and not getattr(optimizer, "step_was_skipped", False):
                    lr_scheduler.step()
                optimizer.zero_grad()
            sync_gradients = accelerator.sync_gradients

            if sync_gradients:
                accumulation_record_ids.clear()
                global_step += 1
                if global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    steps_since_last_log = max(global_step - last_logged_step, 1)
                    elapsed = max(now - last_log_time, 1e-12)
                    last_log_time = now
                    last_logged_step = global_step
                    step_time = elapsed / steps_since_last_log
                    steps_per_sec = steps_since_last_log / elapsed
                    samples_per_sec = (global_batch_size * steps_since_last_log) / elapsed
                    eta_seconds = max(max_train_steps - global_step, 0) / steps_per_sec
                    logged_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    lr_val = lr_scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"[{format_timestamp()}] epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} lr={lr_val:.2e} "
                        f"step_time={step_time:.2f}s steps_per_sec={steps_per_sec:.3f} "
                        f"samples_per_sec={samples_per_sec:.2f} eta={format_duration(eta_seconds)}"
                    )

                if global_step >= max_train_steps:
                    break

        completed_epochs = epoch + 1
        if (epoch + 1) % args.save_every_epochs == 0 or global_step >= max_train_steps:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                model_path=args.model_path,
                codec_path=args.codec_path,
                output_dir=output_root / f"checkpoint-epoch-{epoch + 1}",
                train_args=train_args_to_save,
                global_step=global_step,
                epoch=epoch + 1,
            )

        if global_step >= max_train_steps:
            break

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        model_path=args.model_path,
        codec_path=args.codec_path,
        output_dir=output_root / "checkpoint-last",
        train_args=train_args_to_save,
        global_step=global_step,
        epoch=completed_epochs,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] finished "
        f"global_step={global_step} saved_epochs={completed_epochs} output_dir={output_root}"
    )


if __name__ == "__main__":
    main()
