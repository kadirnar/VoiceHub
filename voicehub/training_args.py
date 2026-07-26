"""Transformers-style configuration for VoiceHub training loops."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from importlib import import_module
from pathlib import Path
from typing import Any

from voicehub.trainer_utils import IntervalStrategy, SchedulerType, write_json


@dataclass
class TrainingArguments:
    """Arguments controlling a single-process PyTorch training run.

    The names intentionally follow ``transformers.TrainingArguments`` so
    TTS recipes can move to VoiceHub without inventing a second
    vocabulary.
    """

    output_dir: str = "trainer_output"
    overwrite_output_dir: bool = False
    do_train: bool = False
    do_eval: bool = False
    eval_strategy: IntervalStrategy | str = IntervalStrategy.NO
    prediction_loss_only: bool = False

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int | None = None

    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lr_scheduler_type: SchedulerType | str = SchedulerType.LINEAR
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    logging_strategy: IntervalStrategy | str = IntervalStrategy.STEPS
    logging_steps: int = 500
    logging_first_step: bool = False
    eval_steps: int | None = None
    save_strategy: IntervalStrategy | str = IntervalStrategy.STEPS
    save_steps: int = 500
    save_total_limit: int | None = None

    seed: int = 42
    data_seed: int | None = None
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = True
    label_names: list[str] = field(default_factory=lambda: ["labels"])

    load_best_model_at_end: bool = False
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    use_cpu: bool = False
    disable_tqdm: bool = True
    report_to: list[str] = field(default_factory=list)
    run_name: str | None = None

    # Compatibility alias retained for recipes written against older
    # Transformers releases.
    evaluation_strategy: IntervalStrategy | str | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.evaluation_strategy is not None:
            if self.eval_strategy != IntervalStrategy.NO:
                raise ValueError("Use either `eval_strategy` or `evaluation_strategy`, not both.")
            self.eval_strategy = self.evaluation_strategy

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        self.output_dir = str(Path(self.output_dir).expanduser())

        positive_values = {
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"`{name}` must be greater than zero.")
        if self.num_train_epochs <= 0 and self.max_steps <= 0:
            raise ValueError("Set a positive `num_train_epochs` or `max_steps`.")
        if self.logging_strategy is IntervalStrategy.STEPS and self.logging_steps <= 0:
            raise ValueError("`logging_steps` must be positive for step logging.")
        if self.save_strategy is IntervalStrategy.STEPS and self.save_steps <= 0:
            raise ValueError("`save_steps` must be positive for step checkpointing.")
        if self.eval_strategy is IntervalStrategy.STEPS:
            self.eval_steps = self.eval_steps or self.logging_steps
            if self.eval_steps <= 0:
                raise ValueError("`eval_steps` must be positive for step evaluation.")
        if self.save_total_limit is not None and self.save_total_limit <= 0:
            raise ValueError("`save_total_limit` must be greater than zero.")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("`warmup_ratio` must be between zero and one.")
        if self.warmup_steps < 0:
            raise ValueError("`warmup_steps` cannot be negative.")
        if self.fp16 and self.bf16:
            raise ValueError("At most one of `fp16` and `bf16` can be enabled.")

        if self.load_best_model_at_end:
            if self.eval_strategy is IntervalStrategy.NO:
                raise ValueError("`load_best_model_at_end` requires an evaluation strategy.")
            if self.eval_strategy is not self.save_strategy:
                raise ValueError("Save and evaluation strategies must match when loading the best model.")
            if self.eval_strategy is IntervalStrategy.STEPS:
                if self.save_steps % int(self.eval_steps or 1) != 0:
                    raise ValueError(
                        "`save_steps` must be a multiple of `eval_steps` when "
                        "loading the best model.")

        if self.metric_for_best_model is None and self.load_best_model_at_end:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

    @property
    def train_batch_size(self) -> int:
        """Effective per-process training batch size."""
        return self.per_device_train_batch_size

    @property
    def eval_batch_size(self) -> int:
        """Effective per-process evaluation batch size."""
        return self.per_device_eval_batch_size

    @property
    def device(self) -> str:
        """Select CPU, CUDA, or MPS without importing PyTorch at package
        import."""
        if self.use_cpu:
            return "cpu"
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def get_warmup_steps(self, num_training_steps: int) -> int:
        """Return explicit warmup steps or derive them from
        ``warmup_ratio``."""
        if self.warmup_steps > 0:
            return self.warmup_steps
        return int(num_training_steps * self.warmup_ratio)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        values = asdict(self)
        values.pop("evaluation_strategy", None)
        for key in (
                "eval_strategy",
                "logging_strategy",
                "save_strategy",
                "lr_scheduler_type",
        ):
            value = values[key]
            values[key] = value.value if isinstance(value, Enum) else value
        return values

    def to_json_string(self) -> str:
        """Serialize arguments as stable, readable JSON."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def save_json(self, json_file: str | Path) -> Path:
        """Save arguments for checkpoint reproducibility."""
        return write_json(json_file, self.to_dict())

    @classmethod
    def from_json_file(cls, json_file: str | Path) -> TrainingArguments:
        """Restore arguments saved by :meth:`save_json`."""
        payload = json.loads(Path(json_file).expanduser().read_text(encoding="utf-8"))
        return cls(**payload)
