"""A Transformers-style Trainer specialized for source-integrated TTS."""

from __future__ import annotations

import inspect
import math
import shutil
import time
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

from voicehub.data_collator import default_data_collator
from voicehub.dependencies import import_optional
from voicehub.errors import UnknownModelError
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from voicehub.trainer_utils import (
    MODEL_STATE_NAME,
    OPTIMIZER_NAME,
    PREFIX_CHECKPOINT_DIR,
    RNG_STATE_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    get_scheduler_lambda,
    set_seed,
)
from voicehub.training.adapters import BaseTrainingAdapter
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.optimization import OptimizerBundle, SchedulerBundle
from voicehub.training_args import TrainingArguments


class Trainer:
    """Complete single-process PyTorch train/evaluate/checkpoint loop.

    Models can return an object or mapping with a ``loss`` field, or
    return loss as the first tuple value. Existing TTS architectures can
    instead supply ``compute_loss_func`` while their source training
    path is adapted.
    """

    def __init__(
        self,
        model=None,
        args: TrainingArguments | None = None,
        data_collator: Callable[[list[Any]], dict[str, Any]] | None = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init: Callable[[], Any] | None = None,
        compute_loss_func: Callable[[Any, Any, int | None], Any] | None = None,
        compute_metrics: Callable[[EvalPrediction], dict[str, float]] | None = None,
        callbacks: list[TrainerCallback | type[TrainerCallback]] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[Any, Any], Any] | None = None,
        training_adapter: BaseTrainingAdapter | None = None,
    ):
        if model is None and model_init is None:
            raise ValueError("Pass either `model` or `model_init` to Trainer.")
        if model is not None and model_init is not None:
            raise ValueError("Pass `model` or `model_init`, not both.")
        if optimizer_cls_and_kwargs is not None and optimizers[0] is not None:
            raise ValueError("`optimizer_cls_and_kwargs` cannot be combined with `optimizers`.")

        self.args = args or TrainingArguments()
        self.model_init = model_init
        self.model = model if model is not None else model_init()
        self.model_wrapped = self.model
        self.training_adapter = self._create_training_adapter(
            self.model,
            training_adapter,
        )
        self._uses_default_data_collator = data_collator is None
        self.data_collator = data_collator or (
            self.training_adapter.data_collator
            if self.training_adapter is not None else default_data_collator)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs
        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_in_train = False
        self._torch = None
        self._scaler = None
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._loss_at_last_log = 0.0
        self._current_gradient_accumulation_steps = (self.args.gradient_accumulation_steps)

        default_callbacks: list[type[TrainerCallback]] = [DefaultFlowCallback]
        if not self.args.disable_tqdm:
            default_callbacks.append(PrinterCallback)
        self.callback_handler = CallbackHandler(
            default_callbacks + list(callbacks or []),
            model=self.model,
            processing_class=self.processing_class,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
        )
        self.control = self.callback_handler.on_init_end(
            self.args,
            self.state,
            self.control,
        )

    def add_callback(self, callback) -> None:
        """Add a callback class or instance."""
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback):
        """Remove and return the first callback of the requested type."""
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback) -> None:
        """Remove the first callback of the requested type."""
        self.callback_handler.remove_callback(callback)

    def _import_torch(self):
        if self._torch is None:
            self._torch = import_optional(
                "torch",
                model_type="Trainer",
                install_extra="training",
            )
        return self._torch

    @staticmethod
    def _create_training_adapter(model, training_adapter):
        if training_adapter is not None:
            if not isinstance(training_adapter, BaseTrainingAdapter):
                raise TypeError("`training_adapter` must inherit `BaseTrainingAdapter`.")
            if training_adapter.model is not model:
                raise ValueError("The training adapter must wrap the model passed to Trainer.")
            return training_adapter
        if isinstance(model, PreTrainedTTSModel):
            try:
                return AutoTrainingAdapter.from_model(model)
            except (KeyError, UnknownModelError):
                return None
        return None

    def _ensure_model_loaded(self) -> None:
        if (isinstance(self.model, PreTrainedTTSModel) and not self.model.is_loaded):
            self.model.load()
        if self.training_adapter is not None:
            self.training_adapter.setup()
            self.model_wrapped = self.training_adapter

    def _runtime_model(self):
        if self.training_adapter is not None:
            return self.training_adapter
        if hasattr(self.model, "parameters"):
            return self.model
        runtime = getattr(self.model, "model", None)
        return runtime if runtime is not None else self.model

    def _move_model_to_device(self) -> None:
        runtime = self._runtime_model()
        if hasattr(runtime, "to"):
            runtime.to(self.args.device)
        if isinstance(self.model, PreTrainedTTSModel):
            self.model.device = self.args.device

    def _set_model_mode(self, training: bool) -> None:
        runtime = self._runtime_model()
        method = getattr(runtime, "train" if training else "eval", None)
        if callable(method):
            method()

    def get_train_dataloader(self):
        """Return a shuffled training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer requires a `train_dataset` for training.")
        torch = self._import_torch()
        generator = torch.Generator()
        generator.manual_seed(self.args.data_seed if self.args.data_seed is not None else self.args.seed)
        is_iterable = isinstance(
            self.train_dataset,
            torch.utils.data.IterableDataset,
        )
        has_length = hasattr(self.train_dataset, "__len__") and not is_iterable
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=has_length,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.args.dataloader_pin_memory and self.args.device.startswith("cuda")),
            generator=generator if has_length else None,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Return a deterministic evaluation DataLoader."""
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer requires an `eval_dataset` for evaluation.")
        torch = self._import_torch()
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.args.dataloader_pin_memory and self.args.device.startswith("cuda")),
        )

    def get_test_dataloader(self, test_dataset):
        """Return a deterministic prediction DataLoader."""
        return self.get_eval_dataloader(test_dataset)

    def create_optimizer(self):
        """Create AdamW parameter groups unless an optimizer was supplied."""
        if self.optimizer is not None:
            return self.optimizer
        torch = self._import_torch()
        runtime = self._runtime_model()
        if not hasattr(runtime, "named_parameters"):
            raise TypeError(
                "The trainable model must expose `named_parameters()`, or an "
                "optimizer must be passed to Trainer.")

        trainable = [(name, parameter) for name, parameter in runtime.named_parameters()
                     if parameter.requires_grad]
        if not trainable:
            raise ValueError("The model has no trainable parameters.")

        if (self.training_adapter is not None and self.training_adapter.spec.separate_optimizers):
            named_groups = self.training_adapter.named_parameter_groups()
            if len(named_groups) > 1:
                self.optimizer = OptimizerBundle({
                    name: self._create_single_optimizer(parameters, torch)
                    for name, parameters in named_groups
                })
            else:
                self.optimizer = self._create_single_optimizer(
                    trainable,
                    torch,
                )
        else:
            self.optimizer = self._create_single_optimizer(trainable, torch)

        self.callback_handler.optimizer = self.optimizer
        return self.optimizer

    def _create_single_optimizer(self, trainable, torch):
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            return optimizer_cls(
                [parameter for _, parameter in trainable],
                **optimizer_kwargs,
            )
        decay_parameters = []
        non_decay_parameters = []
        for name, parameter in trainable:
            normalized_name = name.lower()
            if name.endswith(".bias") or "norm" in normalized_name:
                non_decay_parameters.append(parameter)
            else:
                decay_parameters.append(parameter)
        groups = [
            {
                "params": decay_parameters,
                "weight_decay": self.args.weight_decay
            },
            {
                "params": non_decay_parameters,
                "weight_decay": 0.0
            },
        ]
        groups = [group for group in groups if group["params"]]
        return torch.optim.AdamW(
            groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create a linear, cosine, or constant learning-rate scheduler."""
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        torch = self._import_torch()
        optimizer = optimizer or self.optimizer
        if optimizer is None:
            raise ValueError("Create an optimizer before creating a scheduler.")
        schedule = get_scheduler_lambda(
            self.args.lr_scheduler_type,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        if isinstance(optimizer, OptimizerBundle):
            self.lr_scheduler = SchedulerBundle({
                name:
                torch.optim.lr_scheduler.LambdaLR(
                    named_optimizer,
                    schedule,
                )
                for name, named_optimizer in optimizer.optimizers.items()
            })
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                schedule,
            )
        self.callback_handler.lr_scheduler = self.lr_scheduler
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Set up both optimization objects."""
        self.create_optimizer()
        self.create_scheduler(num_training_steps)

    def _prepare_input(self, value):
        torch = self._import_torch()
        if isinstance(value, dict):
            return {key: self._prepare_input(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self._prepare_input(item) for item in value)
        if isinstance(value, list):
            return [self._prepare_input(item) for item in value]
        if torch.is_tensor(value):
            return value.to(self.args.device)
        return value

    def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not inputs:
            raise ValueError("The received batch was empty.")
        return self._prepare_input(inputs)

    def _model_forward(self, model, inputs):
        if isinstance(model, BaseTrainingAdapter):
            return model(**inputs)
        if isinstance(model, PreTrainedTTSModel):
            return model.forward(**inputs)
        return model(**inputs)

    def _filter_model_inputs(self, model, inputs):
        if not self.args.remove_unused_columns:
            return inputs
        forward = model.forward if hasattr(model, "forward") else model
        parameters = inspect.signature(forward).parameters
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return inputs
        accepted = set(parameters)
        return {key: value for key, value in inputs.items() if key in accepted}

    @staticmethod
    def _extract_loss(outputs):
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, (tuple, list)) and outputs:
            loss = outputs[0]
        else:
            loss = None
        if loss is None:
            raise ValueError(
                "The model did not return a loss. Return `TTSTrainingOutput(loss=...)`, "
                "a mapping with `loss`, or pass `compute_loss_func`.")
        return loss

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        """Run a forward pass and obtain its scalar differentiable loss."""
        model_inputs = dict(inputs)
        labels = None
        if self.compute_loss_func is not None:
            present_labels = {
                name: model_inputs.pop(name)
                for name in self.args.label_names if name in model_inputs
            }
            if len(present_labels) == 1:
                labels = next(iter(present_labels.values()))
            elif present_labels:
                labels = present_labels

        model_inputs = self._filter_model_inputs(model, model_inputs)
        outputs = self._model_forward(model, model_inputs)
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(
                outputs,
                labels,
                num_items_in_batch,
            )
        else:
            loss = self._extract_loss(outputs)
        return (loss, outputs) if return_outputs else loss

    def _autocast_context(self):
        torch = self._import_torch()
        device_type = self.args.device.split(":", 1)[0]
        enabled = (self.args.fp16 or self.args.bf16) and device_type in (
            "cuda",
            "cpu",
        )
        if not enabled:
            return nullcontext()
        dtype = torch.float16 if self.args.fp16 else torch.bfloat16
        return torch.autocast(device_type=device_type, dtype=dtype)

    @staticmethod
    def _find_batch_size(inputs) -> int | None:
        for value in inputs.values():
            if hasattr(value, "shape") and len(value.shape) > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return None

    def training_step(
        self,
        model,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ):
        """Compute, normalize, and backpropagate one micro-batch loss."""
        self._set_model_mode(training=True)
        inputs = self._prepare_inputs(inputs)
        with self._autocast_context():
            loss = self.compute_loss(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
            loss = loss / self._current_gradient_accumulation_steps

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.detach()

    def _optimizer_step(self) -> None:
        torch = self._import_torch()
        runtime = self._runtime_model()
        if self._scaler is not None:
            if isinstance(self.optimizer, OptimizerBundle):
                for optimizer in self.optimizer.optimizers.values():
                    self._scaler.unscale_(optimizer)
            else:
                self._scaler.unscale_(self.optimizer)
        if self.args.max_grad_norm and self.args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                runtime.parameters(),
                self.args.max_grad_norm,
            )
        if self._scaler is not None:
            if isinstance(self.optimizer, OptimizerBundle):
                for optimizer in self.optimizer.optimizers.values():
                    self._scaler.step(optimizer)
            else:
                self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()

    def _initialize_state(
        self,
        *,
        max_steps: int,
        num_train_epochs: int,
    ) -> None:
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.logging_steps = self.args.logging_steps
        self.state.eval_steps = int(self.args.eval_steps or 0)
        self.state.save_steps = self.args.save_steps

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
    ) -> TrainOutput:
        """Train the model and return final loss and runtime metrics."""
        torch = self._import_torch()
        if self.model_init is not None:
            self.model = self.model_init()
            self.model_wrapped = self.model
            self.training_adapter = self._create_training_adapter(
                self.model,
                None,
            )
            if (self._uses_default_data_collator and self.training_adapter is not None):
                self.data_collator = self.training_adapter.data_collator
            self.optimizer = None
            self.lr_scheduler = None
            self.callback_handler.model = self.model

        set_seed(self.args.seed)
        self._ensure_model_loaded()
        self._move_model_to_device()
        runtime = self._runtime_model()
        if self.args.gradient_checkpointing:
            enable_checkpointing = getattr(
                runtime,
                "gradient_checkpointing_enable",
                None,
            )
            if not callable(enable_checkpointing):
                raise ValueError("The model does not implement `gradient_checkpointing_enable()`.")
            enable_checkpointing()
        if self.args.fp16 and self.args.device.split(":", 1)[0] != "cuda":
            raise ValueError("`fp16` training requires a CUDA device.")

        train_dataloader = self.get_train_dataloader()
        try:
            steps_per_epoch = len(train_dataloader)
            dataloader_has_length = True
        except TypeError:
            steps_per_epoch = None
            dataloader_has_length = False
        if not dataloader_has_length and self.args.max_steps <= 0:
            raise ValueError("An iterable dataset without length requires positive `max_steps`.")
        if dataloader_has_length:
            if steps_per_epoch == 0:
                raise ValueError("The training dataset produced no batches.")
            updates_per_epoch = max(
                1,
                math.ceil(steps_per_epoch / self.args.gradient_accumulation_steps),
            )
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
            else:
                max_steps = math.ceil(self.args.num_train_epochs * updates_per_epoch)
            num_train_epochs = max(
                1,
                math.ceil(max_steps / updates_per_epoch),
            )
        else:
            steps_per_epoch = self.args.max_steps * self.args.gradient_accumulation_steps
            updates_per_epoch = self.args.max_steps
            max_steps = self.args.max_steps
            num_train_epochs = 1

        self.create_optimizer_and_scheduler(max_steps)
        checkpoint = self._resolve_checkpoint(resume_from_checkpoint)
        if checkpoint is None:
            self.state = TrainerState()
        self._initialize_state(
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
        )
        if self.args.fp16:
            self._scaler = torch.cuda.amp.GradScaler()

        if checkpoint is not None:
            self._load_checkpoint(checkpoint)
            self._initialize_state(
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
            )

        try:
            self.optimizer.zero_grad(set_to_none=True)
        except TypeError:
            self.optimizer.zero_grad()
        self.control._new_training()
        self.control = self.callback_handler.on_train_begin(
            self.args,
            self.state,
            self.control,
        )
        self.is_in_train = True
        start_time = time.time()
        starting_step = self.state.global_step
        tr_loss = 0.0
        epochs_trained = self.state.global_step // updates_per_epoch
        updates_in_current_epoch = self.state.global_step % updates_per_epoch
        batches_to_skip = (updates_in_current_epoch * self.args.gradient_accumulation_steps)

        for epoch in range(epochs_trained, num_train_epochs):
            self.control = self.callback_handler.on_epoch_begin(
                self.args,
                self.state,
                self.control,
            )
            for step, inputs in enumerate(train_dataloader):
                if epoch == epochs_trained and step < batches_to_skip:
                    continue
                is_accumulation_start = (step % self.args.gradient_accumulation_steps == 0)
                if is_accumulation_start:
                    self.control = self.callback_handler.on_step_begin(
                        self.args,
                        self.state,
                        self.control,
                    )

                batch_size = self._find_batch_size(inputs)
                if dataloader_has_length:
                    group_start = (
                        step // self.args.gradient_accumulation_steps * self.args.gradient_accumulation_steps)
                    self._current_gradient_accumulation_steps = min(
                        self.args.gradient_accumulation_steps,
                        steps_per_epoch - group_start,
                    )
                else:
                    self._current_gradient_accumulation_steps = (self.args.gradient_accumulation_steps)
                num_items = (
                    None if batch_size is None else batch_size * self._current_gradient_accumulation_steps)
                loss = self.training_step(
                    self.model_wrapped,
                    inputs,
                    num_items_in_batch=num_items,
                )
                tr_loss += (float(loss.item()) * self._current_gradient_accumulation_steps)

                is_last_batch = dataloader_has_length and step + 1 == steps_per_epoch
                should_update = ((step + 1) % self.args.gradient_accumulation_steps == 0 or is_last_batch)
                if not should_update:
                    self.control = self.callback_handler.on_substep_end(
                        self.args,
                        self.state,
                        self.control,
                    )
                    continue

                self._optimizer_step()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1) / max(1, steps_per_epoch)
                self.control = self.callback_handler.on_step_end(
                    self.args,
                    self.state,
                    self.control,
                )
                self._maybe_log_save_evaluate(tr_loss)

                if (self.control.should_epoch_stop or self.control.should_training_stop):
                    break

            self.control = self.callback_handler.on_epoch_end(
                self.args,
                self.state,
                self.control,
            )
            self._maybe_log_save_evaluate(tr_loss)
            batches_to_skip = 0
            if self.control.should_training_stop:
                break

        self.is_in_train = False
        if (self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None):
            self._load_model(self.state.best_model_checkpoint)
        self.control = self.callback_handler.on_train_end(
            self.args,
            self.state,
            self.control,
        )

        completed_steps = max(1, self.state.global_step - starting_step)
        training_loss = tr_loss / completed_steps
        runtime_seconds = time.time() - start_time
        metrics = {
            "train_runtime":
            runtime_seconds,
            "train_samples_per_second":
            (self._estimate_train_samples(completed_steps) / max(runtime_seconds, 1e-8)),
            "train_steps_per_second":
            completed_steps / max(runtime_seconds, 1e-8),
            "train_loss":
            training_loss,
        }
        self.log(metrics)
        return TrainOutput(self.state.global_step, training_loss, metrics)

    def _estimate_train_samples(self, completed_steps: int) -> int:
        return (completed_steps * self.args.train_batch_size * self.args.gradient_accumulation_steps)

    def _maybe_log_save_evaluate(self, tr_loss: float) -> None:
        metrics = None
        if self.control.should_log and self.state.global_step > 0:
            steps_since_log = max(
                1,
                self.state.global_step - self._globalstep_last_logged,
            )
            logs = {
                "loss": round(
                    (tr_loss - self._loss_at_last_log) / steps_since_log,
                    6,
                ),
                "learning_rate": self.get_learning_rate(),
            }
            self._loss_at_last_log = tr_loss
            self._globalstep_last_logged = self.state.global_step
            self.log(logs)

        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._update_best_metric(metrics)

        if self.control.should_save:
            self._save_checkpoint()
            self.control = self.callback_handler.on_save(
                self.args,
                self.state,
                self.control,
            )

    def get_learning_rate(self) -> float:
        """Return the first optimizer group's current learning rate."""
        if self.optimizer is None or not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0]["lr"])

    def get_learning_rates(self) -> list[float]:
        """Return the current learning rate of every optimizer group."""
        if self.optimizer is None:
            return []
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def get_num_trainable_parameters(self) -> int:
        """Count parameters whose gradients are enabled."""
        runtime = self._runtime_model()
        if not hasattr(runtime, "parameters"):
            return 0
        return sum(parameter.numel() for parameter in runtime.parameters() if parameter.requires_grad)

    def log(self, logs: dict[str, Any]) -> None:
        """Record metrics in state and dispatch ``on_log`` callbacks."""
        normalized = denumpify_detensorize(dict(logs))
        if self.state.epoch is not None:
            normalized["epoch"] = round(self.state.epoch, 4)
        output = {**normalized, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args,
            self.state,
            self.control,
            normalized,
        )

    def _update_best_metric(self, metrics: dict[str, float]) -> None:
        if not self.args.load_best_model_at_end:
            return
        metric_name = self.args.metric_for_best_model or "loss"
        if not metric_name.startswith("eval_"):
            metric_name = f"eval_{metric_name}"
        if metric_name not in metrics:
            raise KeyError(
                f"Metric {metric_name!r} was not returned by evaluation. "
                f"Available metrics: {', '.join(sorted(metrics))}.")
        metric_value = float(metrics[metric_name])
        is_better = (
            self.state.best_metric is None or
            self.args.greater_is_better and metric_value > self.state.best_metric or
            not self.args.greater_is_better and metric_value < self.state.best_metric)
        if is_better:
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = str(
                Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")

    def prediction_step(
        self,
        model,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
    ) -> tuple[Any | None, Any | None, Any | None]:
        """Run one no-gradient prediction/evaluation batch."""
        torch = self._import_torch()
        prepared = self._prepare_inputs(inputs)
        label_values = self._get_label_values(prepared)
        has_labels = bool(label_values)
        labels = (
            label_values[0] if len(label_values) == 1 else tuple(label_values) if label_values else None)

        self._set_model_mode(training=False)
        with torch.no_grad(), self._autocast_context():
            if has_labels:
                loss, outputs = self.compute_loss(
                    model,
                    prepared,
                    return_outputs=True,
                    num_items_in_batch=self._find_batch_size(prepared),
                )
                loss = loss.detach().mean()
            else:
                loss = None
                model_inputs = self._filter_model_inputs(model, prepared)
                outputs = self._model_forward(model, model_inputs)

        if prediction_loss_only:
            return loss, None, None
        logits = self._extract_predictions(
            outputs,
            ignore_loss=(has_labels and self.compute_loss_func is None),
        )
        if self.preprocess_logits_for_metrics is not None:
            logits = self.preprocess_logits_for_metrics(logits, labels)
        return (
            loss,
            self._nested_detach(logits),
            self._nested_detach(labels),
        )

    def _get_label_values(self, inputs):
        if self.training_adapter is not None:
            for name in self.training_adapter.spec.label_names:
                if name in inputs:
                    return [inputs[name]]
            return []
        if (self.args.label_names and all(name in inputs for name in self.args.label_names)):
            return [inputs[name] for name in self.args.label_names]
        return []

    @staticmethod
    def _extract_predictions(outputs, *, ignore_loss: bool):
        if isinstance(outputs, dict):
            if "logits" in outputs:
                return outputs["logits"]
            if "audio_values" in outputs:
                return outputs["audio_values"]
            values = [
                value for key, value in outputs.items() if key not in ("loss", "hidden_states", "attentions")
            ]
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            return outputs.logits
        elif hasattr(outputs, "audio_values") and outputs.audio_values is not None:
            return outputs.audio_values
        elif isinstance(outputs, (tuple, list)):
            values = list(outputs[1:] if ignore_loss else outputs)
        else:
            return outputs
        if len(values) == 1:
            return values[0]
        return tuple(values)

    def _nested_detach(self, value):
        torch = self._import_torch()
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: self._nested_detach(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._nested_detach(item) for item in value)
        return value

    def _nested_concat(self, values):
        torch = self._import_torch()
        values = [value for value in values if value is not None]
        if not values:
            return None
        first = values[0]
        if torch.is_tensor(first):
            tensors = [value.reshape(1) if value.ndim == 0 else value for value in values]
            return torch.cat(tensors, dim=0)
        if isinstance(first, dict):
            return {key: self._nested_concat([value[key] for value in values]) for key in first}
        if isinstance(first, (tuple, list)):
            return type(first)(
                self._nested_concat([value[index] for value in values]) for index in range(len(first)))
        return values

    def _nested_numpify(self, value):
        torch = self._import_torch()
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.numpy()
        if isinstance(value, dict):
            return {key: self._nested_numpify(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._nested_numpify(item) for item in value)
        try:
            numpy = import_module("numpy")
            return numpy.asarray(value)
        except (ModuleNotFoundError, TypeError, ValueError):
            return value

    def evaluation_loop(
        self,
        dataloader,
        *,
        prediction_loss_only: bool | None = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """Shared evaluation and prediction loop."""
        prediction_loss_only = (
            self.args.prediction_loss_only if prediction_loss_only is None else prediction_loss_only)
        losses = []
        prediction_batches = []
        label_batches = []
        observed_samples = 0

        for inputs in dataloader:
            batch_size = self._find_batch_size(inputs) or 1
            observed_samples += batch_size
            loss, predictions, labels = self.prediction_step(
                self.model_wrapped,
                inputs,
                prediction_loss_only,
            )
            if loss is not None:
                losses.append(float(loss.item()))
            if predictions is not None:
                prediction_batches.append(predictions)
            if labels is not None:
                label_batches.append(labels)
            self.control = self.callback_handler.on_prediction_step(
                self.args,
                self.state,
                self.control,
            )

        predictions = self._nested_numpify(self._nested_concat(prediction_batches))
        label_ids = self._nested_numpify(self._nested_concat(label_batches))
        metrics: dict[str, Any] = {}
        if losses:
            metrics[f"{metric_key_prefix}_loss"] = sum(losses) / len(losses)
        if (self.compute_metrics is not None and predictions is not None and label_ids is not None):
            computed = self.compute_metrics(EvalPrediction(
                predictions=predictions,
                label_ids=label_ids,
            ))
            for key, value in computed.items():
                normalized_key = (
                    key if key.startswith(f"{metric_key_prefix}_") else f"{metric_key_prefix}_{key}")
                metrics[normalized_key] = value
        metrics[f"{metric_key_prefix}_samples"] = observed_samples
        return PredictionOutput(
            predictions,
            label_ids,
            denumpify_detensorize(metrics),
        )

    def evaluate(
        self,
        eval_dataset=None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Run evaluation and return prefixed metrics."""
        self._ensure_model_loaded()
        self._move_model_to_device()
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(dataset, dict):
            metrics = {}
            for name, split in dataset.items():
                metrics.update(self.evaluate(
                    split,
                    metric_key_prefix=f"{metric_key_prefix}_{name}",
                ))
            return metrics
        output = self.evaluation_loop(
            self.get_eval_dataloader(dataset),
            metric_key_prefix=metric_key_prefix,
        )
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args,
            self.state,
            self.control,
            output.metrics,
        )
        return output.metrics

    def predict(
        self,
        test_dataset,
        metric_key_prefix: str = "test",
    ) -> PredictionOutput:
        """Run prediction and return model outputs, labels, and metrics."""
        self._ensure_model_loaded()
        self._move_model_to_device()
        output = self.evaluation_loop(
            self.get_test_dataloader(test_dataset),
            prediction_loss_only=False,
            metric_key_prefix=metric_key_prefix,
        )
        self.control = self.callback_handler.on_predict(
            self.args,
            self.state,
            self.control,
            output.metrics,
        )
        return output

    def save_model(self, output_dir: str | Path | None = None) -> Path:
        """Save model, processor, and arguments in a portable directory."""
        torch = self._import_torch()
        destination = Path(output_dir or self.args.output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(destination)
        runtime = self._runtime_model()
        if hasattr(runtime, "state_dict"):
            torch.save(runtime.state_dict(), destination / MODEL_STATE_NAME)
        if (self.processing_class is not None and hasattr(self.processing_class, "save_pretrained")):
            self.processing_class.save_pretrained(destination)
        self.args.save_json(destination / TRAINING_ARGS_NAME)
        return destination

    def save_state(self) -> Path:
        """Save Trainer state at the root output directory."""
        return self.state.save_to_json(Path(self.args.output_dir) / TRAINER_STATE_NAME)

    def _save_checkpoint(self) -> Path:
        torch = self._import_torch()
        checkpoint = (Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self.save_model(checkpoint)
        self.state.save_to_json(checkpoint / TRAINER_STATE_NAME)
        torch.save(self.optimizer.state_dict(), checkpoint / OPTIMIZER_NAME)
        torch.save(self.lr_scheduler.state_dict(), checkpoint / SCHEDULER_NAME)
        rng_state = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.random.get_rng_state_all()
        torch.save(rng_state, checkpoint / RNG_STATE_NAME)
        self._rotate_checkpoints()
        return checkpoint

    def _rotate_checkpoints(self) -> None:
        limit = self.args.save_total_limit
        if limit is None:
            return
        output_dir = Path(self.args.output_dir)
        checkpoints = []
        for path in output_dir.glob(f"{PREFIX_CHECKPOINT_DIR}-*"):
            try:
                step = int(path.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, path))
        checkpoints.sort()
        best = (
            Path(self.state.best_model_checkpoint).resolve() if self.state.best_model_checkpoint else None)
        latest = checkpoints[-1][1].resolve() if checkpoints else None
        effective_limit = limit
        if (limit == 1 and best is not None and latest is not None and best != latest):
            effective_limit = 2
        while len(checkpoints) > effective_limit:
            removable_index = next(
                (
                    index for index, (_, path) in enumerate(checkpoints)
                    if (best is None or path.resolve() != best) and
                    (latest is None or path.resolve() != latest)),
                None,
            )
            if removable_index is None:
                break
            _, path = checkpoints.pop(removable_index)
            shutil.rmtree(path)

    def _resolve_checkpoint(
        self,
        resume_from_checkpoint: str | bool | None,
    ) -> str | None:
        if resume_from_checkpoint is None or resume_from_checkpoint is False:
            return None
        if resume_from_checkpoint is True:
            checkpoint = get_last_checkpoint(self.args.output_dir)
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found in {self.args.output_dir!r}.")
            return checkpoint
        checkpoint = Path(resume_from_checkpoint).expanduser()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return str(checkpoint)

    def _torch_load(self, path: Path):
        torch = self._import_torch()
        try:
            return torch.load(
                path,
                map_location=self.args.device,
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location=self.args.device)

    def _load_model(self, checkpoint: str | Path) -> None:
        state_path = Path(checkpoint) / MODEL_STATE_NAME
        if not state_path.is_file():
            raise FileNotFoundError(f"Checkpoint is missing {MODEL_STATE_NAME}: {checkpoint}")
        runtime = self._runtime_model()
        if not hasattr(runtime, "load_state_dict"):
            raise TypeError("The trainable model does not implement `load_state_dict()`.")
        runtime.load_state_dict(self._torch_load(state_path))

    def _load_checkpoint(self, checkpoint: str | Path) -> None:
        torch = self._import_torch()
        checkpoint_path = Path(checkpoint)
        self._load_model(checkpoint_path)
        state_path = checkpoint_path / TRAINER_STATE_NAME
        if state_path.is_file():
            self.state = TrainerState.load_from_json(state_path)
        optimizer_path = checkpoint_path / OPTIMIZER_NAME
        scheduler_path = checkpoint_path / SCHEDULER_NAME
        if optimizer_path.is_file():
            self.optimizer.load_state_dict(self._torch_load(optimizer_path))
        if scheduler_path.is_file():
            self.lr_scheduler.load_state_dict(self._torch_load(scheduler_path))
        rng_path = checkpoint_path / RNG_STATE_NAME
        if rng_path.is_file():
            rng_state = self._torch_load(rng_path)
            torch.random.set_rng_state(rng_state["cpu"])
            if "cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(rng_state["cuda"])
