"""Callback, state, and flow-control primitives for VoiceHub Trainer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from voicehub.trainer_utils import IntervalStrategy, write_json


@dataclass
class TrainerState:
    """Mutable, serializable progress state shared with callbacks."""

    epoch: float | None = None
    global_step: int = 0
    max_steps: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    num_train_epochs: float = 0.0
    total_flos: float = 0.0
    log_history: list[dict[str, Any]] = field(default_factory=list)
    best_metric: float | None = None
    best_model_checkpoint: str | None = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str | None = None

    def save_to_json(self, json_path: str | Path) -> Path:
        """Persist state in a human-readable checkpoint file."""
        return write_json(json_path, asdict(self))

    @classmethod
    def load_from_json(cls, json_path: str | Path) -> TrainerState:
        """Restore state from a checkpoint."""
        payload = json.loads(Path(json_path).expanduser().read_text(encoding="utf-8"))
        return cls(**payload)


@dataclass
class TrainerControl:
    """Boolean signals returned by callbacks to control the loop."""

    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False

    def _new_training(self) -> TrainerControl:
        self.should_training_stop = False
        return self

    def _new_epoch(self) -> TrainerControl:
        self.should_epoch_stop = False
        return self

    def _new_step(self) -> TrainerControl:
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False
        return self


class TrainerCallback:
    """Base class for non-invasive Trainer customizations."""

    def on_init_end(self, args, state, control, **kwargs):
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        return control

    def on_train_end(self, args, state, control, **kwargs):
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        return control

    def on_predict(self, args, state, control, **kwargs):
        return control

    def on_save(self, args, state, control, **kwargs):
        return control

    def on_log(self, args, state, control, **kwargs):
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        return control


class DefaultFlowCallback(TrainerCallback):
    """Translate interval strategies into log/evaluate/save signals."""

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step == 1 and args.logging_first_step or
                args.logging_strategy is IntervalStrategy.STEPS and
                state.global_step % state.logging_steps == 0):
            control.should_log = True
        if (args.eval_strategy is IntervalStrategy.STEPS and state.eval_steps > 0 and
                state.global_step % state.eval_steps == 0):
            control.should_evaluate = True
        if (args.save_strategy is IntervalStrategy.STEPS and state.save_steps > 0 and
                state.global_step % state.save_steps == 0):
            control.should_save = True
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.logging_strategy is IntervalStrategy.EPOCH:
            control.should_log = True
        if args.eval_strategy is IntervalStrategy.EPOCH:
            control.should_evaluate = True
        if args.save_strategy is IntervalStrategy.EPOCH:
            control.should_save = True
        return control


class PrinterCallback(TrainerCallback):
    """Minimal logging callback suitable for scripts without integrations."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero:
            print(logs)
        return control


class EarlyStoppingCallback(TrainerCallback):
    """Stop after a metric fails to improve for a configured patience."""

    def __init__(
        self,
        early_stopping_patience: int = 1,
        early_stopping_threshold: float = 0.0,
    ):
        if early_stopping_patience < 1:
            raise ValueError("`early_stopping_patience` must be at least one.")
        if early_stopping_threshold < 0:
            raise ValueError("`early_stopping_threshold` cannot be negative.")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if not args.load_best_model_at_end:
            raise ValueError("EarlyStoppingCallback requires `load_best_model_at_end=True`.")
        if args.metric_for_best_model is None:
            raise ValueError("EarlyStoppingCallback requires `metric_for_best_model`.")
        self.early_stopping_patience_counter = 0
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        metric_name = args.metric_for_best_model
        if not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        if metrics is None or metric_name not in metrics:
            available = ", ".join(sorted(metrics or {}))
            raise KeyError(
                f"Early stopping metric {metric_name!r} is unavailable. "
                f"Available metrics: {available}.")

        metric_value = float(metrics[metric_name])
        if state.best_metric is None:
            improved = True
        elif args.greater_is_better:
            improved = (metric_value > state.best_metric + self.early_stopping_threshold)
        else:
            improved = (metric_value < state.best_metric - self.early_stopping_threshold)

        if improved:
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
            if (self.early_stopping_patience_counter >= self.early_stopping_patience):
                control.should_training_stop = True
        return control


class CallbackHandler:
    """Own callbacks and dispatch events with shared Trainer objects."""

    def __init__(
        self,
        callbacks,
        *,
        model=None,
        processing_class=None,
        optimizer=None,
        lr_scheduler=None,
    ):
        self.callbacks: list[TrainerCallback] = []
        for callback in callbacks:
            self.add_callback(callback)
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def add_callback(self, callback) -> None:
        callback_instance = callback() if isinstance(callback, type) else callback
        if not isinstance(callback_instance, TrainerCallback):
            raise TypeError("Callbacks must inherit from `TrainerCallback`.")
        if any(type(item) is type(callback_instance) for item in self.callbacks):
            return
        self.callbacks.append(callback_instance)

    def pop_callback(self, callback):
        callback_type = callback if isinstance(callback, type) else type(callback)
        for index, item in enumerate(self.callbacks):
            if isinstance(item, callback_type):
                return self.callbacks.pop(index)
        return None

    def remove_callback(self, callback) -> None:
        self.pop_callback(callback)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                **kwargs,
            )
            if result is not None:
                control = result
        return control

    def on_init_end(self, args, state, control):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args, state, control):
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args, state, control):
        return self.call_event("on_train_end", args, state, control)

    def on_epoch_begin(self, args, state, control):
        control._new_epoch()
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args, state, control):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args, state, control):
        control._new_step()
        return self.call_event("on_step_begin", args, state, control)

    def on_substep_end(self, args, state, control):
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args, state, control):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args, state, control, metrics):
        control.should_evaluate = False
        return self.call_event(
            "on_evaluate",
            args,
            state,
            control,
            metrics=metrics,
        )

    def on_predict(self, args, state, control, metrics):
        return self.call_event(
            "on_predict",
            args,
            state,
            control,
            metrics=metrics,
        )

    def on_save(self, args, state, control):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args, state, control, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args, state, control):
        return self.call_event("on_prediction_step", args, state, control)
