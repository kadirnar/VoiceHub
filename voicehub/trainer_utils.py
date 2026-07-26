"""Framework-light utility types shared by the VoiceHub trainer."""

from __future__ import annotations

import json
import math
import random
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any, NamedTuple

PREFIX_CHECKPOINT_DIR = "checkpoint"
TRAINER_STATE_NAME = "trainer_state.json"
TRAINING_ARGS_NAME = "training_args.json"
MODEL_STATE_NAME = "model_state.pt"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
RNG_STATE_NAME = "rng_state.pth"


class ExplicitEnum(str, Enum):
    """String enum that reports accepted values in invalid-value errors."""

    @classmethod
    def _missing_(cls, value):
        choices = ", ".join(repr(member.value) for member in cls)
        raise ValueError(f"{value!r} is not valid for {cls.__name__}; choose {choices}.")

    def __str__(self) -> str:
        return self.value


class IntervalStrategy(ExplicitEnum):
    """When a recurring Trainer action should run."""

    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"


class SchedulerType(ExplicitEnum):
    """Learning-rate schedules implemented without external trainer
    packages."""

    LINEAR = "linear"
    COSINE = "cosine"
    CONSTANT = "constant"


class TrainOutput(NamedTuple):
    """Return value of :meth:`voicehub.Trainer.train`."""

    global_step: int
    training_loss: float
    metrics: dict[str, float]


class PredictionOutput(NamedTuple):
    """Predictions, references, and metrics returned by ``Trainer.predict``."""

    predictions: Any
    label_ids: Any
    metrics: dict[str, float]


class EvalPrediction:
    """Container passed to a user-provided ``compute_metrics`` function."""

    def __init__(
        self,
        predictions: Any,
        label_ids: Any,
        inputs: Any | None = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is None:
            return iter((self.predictions, self.label_ids))
        return iter((self.predictions, self.label_ids, self.inputs))

    def __getitem__(self, index: int):
        return tuple(self)[index]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and installed compute backends."""
    random.seed(seed)
    try:
        numpy = import_module("numpy")
        numpy.random.seed(seed)
    except ModuleNotFoundError:
        pass

    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_last_checkpoint(folder: str | Path) -> str | None:
    """Return the checkpoint with the greatest numeric global step."""
    output_dir = Path(folder).expanduser()
    if not output_dir.is_dir():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for path in output_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def get_scheduler_lambda(
    scheduler_type: SchedulerType | str,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
):
    """Build the scalar schedule used by ``torch.optim.lr_scheduler``."""
    schedule = SchedulerType(scheduler_type)
    total_steps = max(1, num_training_steps)
    warmup_steps = max(0, num_warmup_steps)

    def warmup(current_step: int) -> float:
        if warmup_steps and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    if schedule is SchedulerType.CONSTANT:
        return warmup

    def progress(current_step: int) -> float:
        if current_step < warmup_steps:
            return warmup(current_step)
        denominator = max(1, total_steps - warmup_steps)
        return min(1.0, max(0.0, (current_step - warmup_steps) / denominator))

    if schedule is SchedulerType.COSINE:

        def cosine(current_step: int) -> float:
            if current_step < warmup_steps:
                return warmup(current_step)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress(current_step))))

        return cosine

    def linear(current_step: int) -> float:
        if current_step < warmup_steps:
            return warmup(current_step)
        return max(0.0, 1.0 - progress(current_step))

    return linear


def denumpify_detensorize(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert scalar tensors and NumPy values into JSON-compatible values."""
    normalized = {}
    for key, value in metrics.items():
        if hasattr(value, "item") and callable(value.item):
            try:
                value = value.item()
            except (TypeError, ValueError):
                pass
        normalized[key] = value
    return normalized


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a deterministic UTF-8 JSON document."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
