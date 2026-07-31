#!/usr/bin/env python3
"""Run one reproducible real-checkpoint VITS generator warm-start step.

MMS-TTS metadata does not publish the original FFT, hop, window, mel, or
segment recipe. This smoke test therefore exercises VoiceHub's explicitly
preprocessed generator warm-start, not the complete adversarial VITS recipe.
It creates one short example from the base checkpoint, takes one deterministic
step, exports native Safetensors, reloads them, and runs long-form inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

TRAINING_TEXT = (
    "VoiceHub makes speech model inference easier, clearer, and more "
    "reproducible for every user."
)
VALIDATION_TEXT = (
    "VoiceHub makes speech model inference easier to understand and reproduce. "
    "This sample is intentionally long enough to produce more than ten seconds "
    "of clear spoken audio. It measures the complete text tokenizer, acoustic "
    "model, normalizing flow, and neural vocoder pipeline on one graphics "
    "processor. The same sentence and random seed are used for every benchmark "
    "so that baseline and fine-tuned runs remain directly comparable."
)
_UNSET = object()


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be finite and greater than zero.")
    return parsed


def prepare_output_directory(path: Path) -> Path:
    """Create an empty destination without overwriting prior artifacts."""
    destination = path.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise FileExistsError(f"VITS export path is not a directory: {destination}.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"VITS export path is not empty: {destination}. "
            "Choose a new path to preserve the existing checkpoint.")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def build_linear_spectrogram(
    waveform: Any,
    torch: Any,
    *,
    n_fft: int = 1_024,
    hop_length: int = 256,
    win_length: int = 1_024,
) -> Any:
    """Create the explicit linear magnitude input used by this smoke test."""
    waveform = waveform.detach().float().cpu().reshape(-1).contiguous()
    if waveform.numel() < n_fft:
        raise ValueError(
            f"Training waveform needs at least {n_fft} samples; "
            f"received {waveform.numel()}.")
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError("Training waveform contains NaN or infinite samples.")
    return torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length),
        return_complex=True,
    ).abs()


def training_reference(
    output: Any,
    torch: Any,
    *,
    minimum_seconds: float,
) -> tuple[Any, int]:
    """Return the complete transcript-aligned self-synthesis waveform."""
    sample_rate = getattr(output, "sample_rate", None)
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise RuntimeError("Training audio sample rate must be a positive integer.")
    audio = getattr(output, "audio", None)
    if audio is None:
        raise RuntimeError("Training synthesis did not return an audio waveform.")
    audio = audio.detach().float().cpu().reshape(-1).contiguous()
    if not bool(torch.isfinite(audio).all().item()):
        raise RuntimeError("Training synthesis contains NaN or infinite samples.")
    duration_seconds = audio.numel() / sample_rate
    if duration_seconds < minimum_seconds:
        raise RuntimeError(
            f"Training text produced only {duration_seconds:.3f} seconds; "
            f"required at least {minimum_seconds:.3f} seconds.")
    return audio, sample_rate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(
    model: Any,
    *,
    requested_source: str | Path,
    requested_revision: object = _UNSET,
) -> dict[str, Any]:
    """Describe the exact resolved artifact when the runtime exposes it."""
    config = getattr(model, "config", None)
    if requested_revision is _UNSET:
        requested_revision = getattr(config, "revision", None)
    artifacts = None
    for owner in (
            model,
            getattr(model, "model", None),
            getattr(model, "runtime", None),
    ):
        candidate = getattr(owner, "artifacts", None)
        if candidate is not None:
            artifacts = candidate
            break

    resolved_source = getattr(artifacts, "source", None)
    resolved_revision = getattr(artifacts, "revision", None)
    revision_was_explicit = requested_revision is not None
    if (
            requested_revision is None
            and getattr(config, "model_type", None) == "vits"
            and resolved_revision is not None
    ):
        requested_revision = "main"
    checkpoint_value = getattr(artifacts, "checkpoint", None)
    checkpoint_path = None
    if isinstance(checkpoint_value, (str, Path)):
        candidate_path = Path(checkpoint_value).expanduser()
        if candidate_path.is_file():
            checkpoint_path = candidate_path.resolve()
    weight_path = (
        checkpoint_path
        if checkpoint_path is not None
        and not checkpoint_path.name.endswith(".index.json")
        else None
    )
    return {
        "requested_source": str(requested_source),
        "requested_revision": requested_revision,
        "requested_revision_was_explicit": revision_was_explicit,
        "resolved_source": (
            None if resolved_source is None else str(resolved_source)),
        "resolved_revision": resolved_revision,
        "local_checkpoint_path": (
            None if checkpoint_path is None else str(checkpoint_path)),
        "local_weight_sha256": (
            None if weight_path is None else _sha256_file(weight_path)),
        "weight_digest_status": (
            "sha256"
            if weight_path is not None else
            "checkpoint-index-only"
            if checkpoint_path is not None else
            "not-exposed"),
    }


def state_fingerprint(state: Mapping[str, Any]) -> str:
    """Hash tensor names, dtypes, shapes, and exact bytes deterministically."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _run_seeded(adapter: Any, inputs: dict[str, Any], torch: Any, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    inputs["generator"] = torch.Generator(device=inputs["audio_values"].device).manual_seed(seed)
    if inputs["audio_values"].device.type == "cuda":
        torch.cuda.synchronize(inputs["audio_values"].device)
    started = time.perf_counter()
    output = adapter(**inputs)
    if inputs["audio_values"].device.type == "cuda":
        torch.cuda.synchronize(inputs["audio_values"].device)
    return output, time.perf_counter() - started


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from voicehub import (
        AutoModelForTextToSpeech,
    )
    from voicehub import __version__ as voicehub_version

    destination = prepare_output_directory(args.output_dir)
    config_kwargs = {
        "torch_dtype": args.dtype,
        "local_files_only": args.local_files_only,
    }
    if args.revision is not None:
        config_kwargs["revision"] = args.revision
    inference = AutoModelForTextToSpeech.from_pretrained(
        args.model,
        model_type="vits",
        device=args.device,
        lazy_load=False,
        config_kwargs=config_kwargs,
    )
    generated = inference.generate(
        args.training_text,
        seed=args.generation_seed,
    )
    base_checkpoint_identity = checkpoint_identity(
        inference,
        requested_source=args.model,
        requested_revision=args.revision,
    )
    reference, sample_rate = training_reference(
        generated,
        torch,
        minimum_seconds=args.minimum_training_seconds,
    )
    del generated, inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    spectrogram = build_linear_spectrogram(reference, torch)
    training_config = {
        **config_kwargs,
        "enable_native_generator_training": True,
    }
    model = AutoModelForTextToSpeech.from_pretrained(
        args.model,
        model_type="vits",
        device=args.device,
        lazy_load=True,
        config_kwargs=training_config,
    )
    adapter = model.get_training_adapter().build_training_graph()
    parameters = list(adapter.parameters())
    optimizer = torch.optim.SGD(parameters, lr=args.learning_rate)
    device = torch.device(model.device)
    inputs = {
        "training_phase": "generator",
        "text": args.training_text,
        "spectrogram": spectrogram.unsqueeze(0).to(device),
        "audio_values": reference.unsqueeze(0).to(device),
        "audio_lengths": torch.tensor(
            [reference.numel()],
            dtype=torch.long,
            device=device,
        ),
    }

    repeat_one, _ = _run_seeded(adapter, inputs, torch, args.training_seed)
    repeat_one_loss = float(repeat_one.loss.detach().float().cpu())
    del repeat_one
    repeat_two, _ = _run_seeded(adapter, inputs, torch, args.training_seed)
    repeat_two_loss = float(repeat_two.loss.detach().float().cpu())
    del repeat_two
    pre_step_fingerprint = state_fingerprint(model.model.state_dict())

    optimizer.zero_grad(set_to_none=True)
    before, forward_seconds = _run_seeded(
        adapter,
        inputs,
        torch,
        args.training_seed,
    )
    loss_before = float(before.loss.detach().float().cpu())
    losses_before = {
        name: float(value.detach().float().cpu())
        for name, value in before.losses.items()
    }
    before.loss.backward()
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            parameters,
            args.max_gradient_norm,
        ).detach().float().cpu())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after, evaluation_seconds = _run_seeded(
        adapter,
        inputs,
        torch,
        args.training_seed,
    )
    loss_after = float(after.loss.detach().float().cpu())
    losses_after = {
        name: float(value.detach().float().cpu())
        for name, value in after.losses.items()
    }
    trained_fingerprint = state_fingerprint(model.model.state_dict())

    export_started = time.perf_counter()
    model.export_native_pretrained(destination)
    export_seconds = time.perf_counter() - export_started
    native_export_files = sorted(
        str(path.relative_to(destination))
        for path in destination.iterdir() if path.is_file())
    del after, before, adapter, optimizer, parameters, inputs, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reload_config_kwargs = {
        key: value
        for key, value in config_kwargs.items() if key != "revision"
    }
    reloaded = AutoModelForTextToSpeech.from_pretrained(
        destination,
        model_type="vits",
        device=args.device,
        lazy_load=False,
        config_kwargs=reload_config_kwargs,
    )
    reloaded_checkpoint_identity = checkpoint_identity(
        reloaded,
        requested_source=destination,
        requested_revision=None,
    )
    reloaded_fingerprint = state_fingerprint(reloaded.model.state_dict())
    validation_path = destination / "validation.wav"
    if torch.device(reloaded.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(reloaded.device))
        torch.cuda.synchronize(torch.device(reloaded.device))
    inference_started = time.perf_counter()
    validation = reloaded.generate(
        args.validation_text,
        seed=args.generation_seed,
        output_file=validation_path,
    )
    if torch.device(reloaded.device).type == "cuda":
        torch.cuda.synchronize(torch.device(reloaded.device))
    inference_seconds = time.perf_counter() - inference_started
    duration_seconds = validation.audio.numel() / validation.sample_rate
    if duration_seconds < args.minimum_validation_seconds:
        raise RuntimeError(
            f"Reloaded validation audio is {duration_seconds:.3f} seconds, "
            f"below the required {args.minimum_validation_seconds:.3f} seconds.")
    peak_memory = (
        int(torch.cuda.max_memory_allocated(torch.device(reloaded.device)))
        if torch.device(reloaded.device).type == "cuda" else None)

    result = {
        "status":
        "ok",
        "coverage":
        "real-checkpoint-one-step-generator-warm-start",
        "voicehub_version":
        voicehub_version,
        "checkpoint":
        args.model,
        "checkpoint_identity":
        base_checkpoint_identity,
        "checkpoint_parameters":
        sum(parameter.numel() for parameter in reloaded.model.parameters()),
        "dataset_examples":
        1,
        "dataset_source":
        "base checkpoint self-synthesis",
        "training_text":
        args.training_text,
        "training_audio_samples":
        reference.numel(),
        "training_audio_duration_seconds":
        reference.numel() / sample_rate,
        "sample_rate":
        sample_rate,
        "spectrogram_shape":
        list(spectrogram.shape),
        "spectrogram_recipe": {
            "n_fft": 1_024,
            "hop_length": 256,
            "win_length": 1_024,
            "window": "hann",
            "magnitude": True,
        },
        "training_scope": (
            "preprocessed generator warm-start; not the full adversarial "
            "recipe because MMS metadata omits published acoustic settings"),
        "optimizer": {
            "name": "SGD",
            "learning_rate": args.learning_rate,
        },
        "steps":
        1,
        "comparison_seed":
        args.training_seed,
        "pre_step_repeat_exact":
        repeat_one_loss == repeat_two_loss,
        "pre_step_repeat_losses": [
            repeat_one_loss,
            repeat_two_loss,
        ],
        "training_forward_matches_precheck":
        loss_before == repeat_one_loss,
        "pre_step_state_sha256":
        pre_step_fingerprint,
        "state_changed":
        trained_fingerprint != pre_step_fingerprint,
        "loss_before":
        loss_before,
        "loss_after":
        loss_after,
        "loss_delta":
        loss_after - loss_before,
        "loss_components_before":
        losses_before,
        "loss_components_after":
        losses_after,
        "gradient_norm_before_clip":
        gradient_norm,
        "max_gradient_norm":
        args.max_gradient_norm,
        "forward_seconds":
        forward_seconds,
        "post_step_evaluation_seconds":
        evaluation_seconds,
        "export_path":
        str(destination),
        "export_seconds":
        export_seconds,
        "export_files":
        native_export_files,
        "trained_state_sha256":
        trained_fingerprint,
        "reloaded_state_sha256":
        reloaded_fingerprint,
        "reload_exact":
        trained_fingerprint == reloaded_fingerprint,
        "reloaded_checkpoint_identity":
        reloaded_checkpoint_identity,
        "reloaded_inference": {
            "audio_samples":
            validation.audio.numel(),
            "sample_rate":
            validation.sample_rate,
            "duration_seconds":
            duration_seconds,
            "latency_seconds":
            inference_seconds,
            "real_time_factor":
            inference_seconds / duration_seconds,
            "peak_allocated_bytes":
            peak_memory,
            "output_file":
            str(validation_path),
            "sha256_float32":
            hashlib.sha256(
                validation.audio.detach().float().cpu().numpy().tobytes(order="C")).hexdigest(),
        },
    }
    report_path = destination / "smoke_finetune_report.json"
    report_path.write_text(
        json.dumps(
            result,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/mms-tts-eng")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-text", default=TRAINING_TEXT)
    parser.add_argument("--validation-text", default=VALIDATION_TEXT)
    parser.add_argument(
        "--minimum-training-seconds",
        type=_positive_float,
        default=2.5,
    )
    parser.add_argument("--minimum-validation-seconds", type=_positive_float, default=10.0)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-4)
    parser.add_argument("--max-gradient-norm", type=_positive_float, default=1.0)
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = run(args)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return int(
        not result["pre_step_repeat_exact"]
        or not result["training_forward_matches_precheck"]
        or not result["state_changed"]
        or result["loss_after"] >= result["loss_before"]
        or not result["reload_exact"])


if __name__ == "__main__":
    raise SystemExit(main())
