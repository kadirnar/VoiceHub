#!/usr/bin/env python3
"""Synthetic NFE/latency benchmark for VoiceHub diffusion policies.

This checks controller plumbing and relative neural-network evaluation
counts.  It is not an audio-quality benchmark; use the same
configuration with a real checkpoint before selecting a production
preset.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from torch import nn

from voicehub.optimization import DiffusionSamplingConfig, DiffusionSamplingController, DiffusionStepContext


class SyntheticVelocity(nn.Module):

    def __init__(self, width: int, depth: int):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend((nn.Linear(width, width), nn.SiLU()))
        layers.append(nn.Linear(width, width))
        self.network = nn.Sequential(*layers)
        self.calls = 0

    def forward(self, state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.network(state) + timestep.to(state.dtype)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _config(
    method: str,
    *,
    target_steps: int,
) -> DiffusionSamplingConfig:
    common = {"target_steps": target_steps}
    if method == "native":
        return DiffusionSamplingConfig(**common)
    if method == "stork2":
        return DiffusionSamplingConfig(
            **common,
            solver="stork2",
        )
    if method in {"fora", "taylor"}:
        return DiffusionSamplingConfig(
            **common,
            prediction_cache=method,
            cache_interval=2,
            cache_warmup_steps=1,
            cache_max_consecutive_steps=1,
        )
    if method == "teacache":
        return DiffusionSamplingConfig(
            **common,
            prediction_cache="teacache",
            teacache_coefficients=(0.0, 1.0),
            cache_warmup_steps=1,
            cache_rel_l1_threshold=0.10,
            cache_error_budget=0.20,
        )
    if method == "smoothcache":
        return DiffusionSamplingConfig(
            **common,
            prediction_cache="smoothcache",
            smoothcache_compute_step_mask=tuple(index % 2 == 0 for index in range(target_steps)),
        )
    raise ValueError(f"Unknown method: {method}")


@torch.inference_mode()
def _run(
    model: SyntheticVelocity,
    initial: torch.Tensor,
    native_schedule: torch.Tensor,
    config: DiffusionSamplingConfig,
) -> torch.Tensor:
    controller = DiffusionSamplingController(config)
    schedule = controller.prepare_schedule(native_schedule)
    state = initial.clone()
    total_steps = schedule.numel() - 1
    for index in range(total_steps):
        context = DiffusionStepContext(
            index=index,
            total_steps=total_steps,
            timestep=schedule[index],
            next_timestep=schedule[index + 1],
            lane="synthetic",
            solver=config.solver.value,
        )

        def compute() -> torch.Tensor:
            return model(state, schedule[index])

        velocity = controller.evaluate(
            context,
            state,
            compute,
        )
        state = controller.advance(
            context,
            state,
            velocity,
        )
    return state


def _measure(
    model: SyntheticVelocity,
    initial: torch.Tensor,
    schedule: torch.Tensor,
    config: DiffusionSamplingConfig,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    device = initial.device
    for _ in range(warmup):
        _run(model, initial, schedule, config)
    _synchronize(device)
    calls_before = model.calls
    started = time.perf_counter()
    for _ in range(repeats):
        _run(model, initial, schedule, config)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    calls = (model.calls - calls_before) / repeats
    return elapsed * 1_000 / repeats, calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "native",
            "stork2",
            "fora",
            "teacache",
            "smoothcache",
            "taylor",
        ),
        default="stork2",
    )
    parser.add_argument("--native-steps", type=int, default=50)
    parser.add_argument("--target-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.native_steps <= 0 or args.target_steps <= 0:
        parser.error("Step counts must be positive.")
    if args.target_steps > args.native_steps:
        parser.error("--target-steps cannot exceed --native-steps.")
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup non-negative.")

    device = torch.device(args.device)
    model = SyntheticVelocity(args.width, args.depth).to(device).eval()
    initial = torch.randn(
        args.batch_size,
        args.frames,
        args.width,
        device=device,
    )
    schedule = torch.linspace(
        0,
        1,
        args.native_steps + 1,
        device=device,
    )
    baseline = DiffusionSamplingConfig()
    candidate = _config(
        args.method,
        target_steps=args.target_steps,
    )
    baseline_ms, baseline_nfe = _measure(
        model,
        initial,
        schedule,
        baseline,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    candidate_ms, candidate_nfe = _measure(
        model,
        initial,
        schedule,
        candidate,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    print(
        json.dumps(
            {
                "benchmark": "synthetic-diffusion-sampling",
                "device": str(device),
                "method": args.method,
                "native_steps": args.native_steps,
                "target_steps": args.target_steps,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "speedup": baseline_ms / candidate_ms,
                "baseline_nfe": baseline_nfe,
                "candidate_nfe": candidate_nfe,
                "quality_validated": False,
            },
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
