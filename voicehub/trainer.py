"""A Transformers-style Trainer specialized for source-integrated TTS."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import shutil
import time
import uuid
from collections.abc import Mapping
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
    CHECKPOINT_COMPLETE_NAME,
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_MANIFEST_NAME,
    FORMAT_V2_REQUIRED_FILES,
    LEGACY_RESUME_FILES,
    MODEL_STATE_NAME,
    NATIVE_EXPORT_DIR,
    OPTIMIZER_NAME,
    PREFIX_CHECKPOINT_DIR,
    RNG_STATE_NAME,
    SCALER_STATE_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    TRAINING_RECIPE_NAME,
    TRAINING_RUNTIME_STATE_NAME,
    EpochRandomSampler,
    EvalPrediction,
    PredictionOutput,
    TrainOutput,
    denumpify_detensorize,
    get_last_checkpoint,
    get_scheduler_lambda,
    set_seed,
    write_json,
)
from voicehub.training.adapters import BaseTrainingAdapter
from voicehub.training.auto import AutoTrainingAdapter
from voicehub.training.contracts import TrainingContext
from voicehub.training.optimization import OptimizerBundle, SchedulerBundle
from voicehub.training.strategy import TrainingStrategy, get_training_strategy
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
        optimizer_factory: Callable[[str, list[tuple[str, Any]], TrainingArguments], Any] | None = None,
        scheduler_factory: Callable[[str, Any, int, TrainingArguments], Any] | None = None,
        training_strategy: str | TrainingStrategy | None = None,
    ):
        if model is None and model_init is None:
            raise ValueError("Pass either `model` or `model_init` to Trainer.")
        if model is not None and model_init is not None:
            raise ValueError("Pass `model` or `model_init`, not both.")
        if optimizer_cls_and_kwargs is not None and optimizers[0] is not None:
            raise ValueError("`optimizer_cls_and_kwargs` cannot be combined with `optimizers`.")
        if optimizer_factory is not None and (optimizer_cls_and_kwargs is not None or
                                              optimizers[0] is not None):
            raise ValueError(
                "`optimizer_factory` cannot be combined with supplied optimizers "
                "or `optimizer_cls_and_kwargs`.")
        if scheduler_factory is not None and optimizers[1] is not None:
            raise ValueError("`scheduler_factory` cannot be combined with a supplied scheduler.")
        if model_init is not None and training_adapter is not None:
            raise ValueError(
                "A concrete `training_adapter` cannot be reused with `model_init`; "
                "register an adapter factory with AutoTrainingAdapter instead.")
        if model_init is not None and any(item is not None for item in optimizers):
            raise ValueError(
                "Supplied optimizer objects are tied to one model instance and "
                "cannot be combined with `model_init`.")

        self.args = args or TrainingArguments()
        self.training_strategy = get_training_strategy(training_strategy)
        self.model_init = model_init
        self.model = model
        self.model_wrapped = self.model
        self.training_adapter = self._create_training_adapter(
            self.model,
            training_adapter,
        )
        self._uses_default_data_collator = data_collator is None
        dataset_collator = getattr(train_dataset, "collate_fn", None)
        self._dataset_data_collator = (dataset_collator if callable(dataset_collator) else None)
        self.data_collator = (
            data_collator or self._dataset_data_collator or (
                self.training_adapter.data_collator
                if self.training_adapter is not None else default_data_collator))
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_in_train = False
        self._torch = None
        self._scaler = None
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._loss_at_last_log = 0.0
        self._current_gradient_accumulation_steps = (self.args.gradient_accumulation_steps)
        self._active_optimizer_names: set[str] = set()
        self._optimizer_microstep_counts: dict[str, int] = {}
        self._train_sampler = None
        self._train_dataloader_generator = None
        self._model_prepared = False
        self._optimization_prepared = False
        self._uses_named_optimizers = False
        self._optimizer_names: tuple[str, ...] = ("default", )
        self._deferred_rng_state: dict[str, Any] = {}
        self._resume_topology: dict[str, int] | None = None
        self._strict_runtime_resume = False

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
        if self.training_adapter is not None:
            self.training_adapter.build_training_graph()
            return
        if (isinstance(self.model, PreTrainedTTSModel) and not self.model.is_loaded):
            self.model.load()

    def _runtime_model(self):
        if self.training_adapter is not None:
            return self.training_adapter
        if hasattr(self.model, "parameters"):
            return self.model
        runtime = getattr(self.model, "model", None)
        return runtime if runtime is not None else self.model

    def _move_model_to_device(self) -> None:
        if self._model_prepared:
            return
        runtime = self._runtime_model()
        if self.training_adapter is not None:
            prepared = self.training_strategy.prepare_training_adapter(
                self.training_adapter,
                device=self.args.device,
            )
        else:
            prepared = self.training_strategy.prepare_model(
                runtime,
                device=self.args.device,
            )
        self.model_wrapped = prepared
        self._model_prepared = True
        if isinstance(self.model, PreTrainedTTSModel):
            self.model.device = self.args.device

    def _set_model_mode(self, training: bool) -> None:
        runtime = self.model_wrapped
        method = getattr(runtime, "train" if training else "eval", None)
        if callable(method):
            method()

    def get_train_dataloader(self):
        """Return a shuffled training DataLoader."""
        if self.train_dataset is None:
            raise ValueError("Trainer requires a `train_dataset` for training.")
        torch = self._import_torch()
        is_iterable = isinstance(
            self.train_dataset,
            torch.utils.data.IterableDataset,
        )
        has_length = hasattr(self.train_dataset, "__len__") and not is_iterable
        self._train_dataloader_generator = torch.Generator()
        self._train_dataloader_generator.manual_seed(
            self.args.data_seed if self.args.data_seed is not None else self.args.seed)
        self._train_sampler = (
            EpochRandomSampler(
                self.train_dataset,
                seed=(self.args.data_seed if self.args.data_seed is not None else self.args.seed),
            ) if has_length else None)
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=False,
            sampler=self._train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.args.dataloader_pin_memory and self.args.device.startswith("cuda")),
            generator=self._train_dataloader_generator,
        )
        return self.training_strategy.prepare_dataloader(
            dataloader,
            training=True,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Return a deterministic evaluation DataLoader."""
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer requires an `eval_dataset` for evaluation.")
        torch = self._import_torch()
        generator = torch.Generator()
        generator.manual_seed(self.args.data_seed if self.args.data_seed is not None else self.args.seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=(self.args.dataloader_pin_memory and self.args.device.startswith("cuda")),
            generator=generator,
        )
        return self.training_strategy.prepare_dataloader(
            dataloader,
            training=False,
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
            if not named_groups:
                raise ValueError(
                    "The training recipe requests separate optimizers but "
                    "resolved no named trainable parameter groups.")
            self.optimizer = OptimizerBundle({
                name:
                self._create_single_optimizer(
                    parameters,
                    torch,
                    name=name,
                )
                for name, parameters in named_groups
            })
        else:
            self.optimizer = self._create_single_optimizer(trainable, torch)

        self.callback_handler.optimizer = self.optimizer
        return self.optimizer

    def _create_single_optimizer(self, trainable, torch, *, name: str = "default"):
        if self.optimizer_factory is not None:
            return self.optimizer_factory(
                name,
                list(trainable),
                self.args,
            )
        if self.optimizer_cls_and_kwargs is not None:
            optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            return optimizer_cls(
                [parameter for _, parameter in trainable],
                **optimizer_kwargs,
            )
        if self.training_adapter is not None:
            recipe_optimizer = self.training_adapter.create_optimizer(
                name,
                list(trainable),
                self.args,
            )
            if recipe_optimizer is not None:
                return recipe_optimizer
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
        if self.scheduler_factory is not None:
            if isinstance(optimizer, OptimizerBundle):
                self.lr_scheduler = SchedulerBundle({
                    name:
                    self.scheduler_factory(
                        name,
                        named_optimizer,
                        self._optimizer_training_steps(
                            name,
                            num_training_steps,
                        ),
                        self.args,
                    )
                    for name, named_optimizer in optimizer.optimizers.items()
                })
            else:
                self.lr_scheduler = self.scheduler_factory(
                    "default",
                    optimizer,
                    num_training_steps,
                    self.args,
                )
            self.callback_handler.lr_scheduler = self.lr_scheduler
            return self.lr_scheduler
        if isinstance(optimizer, OptimizerBundle):
            schedulers = {}
            for name, named_optimizer in optimizer.optimizers.items():
                optimizer_steps = self._optimizer_training_steps(
                    name,
                    num_training_steps,
                )
                recipe_scheduler = (
                    self.training_adapter.create_scheduler(
                        name,
                        named_optimizer,
                        optimizer_steps,
                        self.args,
                    ) if self.training_adapter is not None else None)
                if recipe_scheduler is not None:
                    schedulers[name] = recipe_scheduler
                    continue
                named_schedule = get_scheduler_lambda(
                    self.args.lr_scheduler_type,
                    num_warmup_steps=self.args.get_warmup_steps(optimizer_steps),
                    num_training_steps=optimizer_steps,
                )
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                    named_optimizer,
                    named_schedule,
                )
            self.lr_scheduler = SchedulerBundle(schedulers)
        else:
            recipe_scheduler = (
                self.training_adapter.create_scheduler(
                    "default",
                    optimizer,
                    num_training_steps,
                    self.args,
                ) if self.training_adapter is not None else None)
            if recipe_scheduler is not None:
                self.lr_scheduler = recipe_scheduler
            else:
                schedule = get_scheduler_lambda(
                    self.args.lr_scheduler_type,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
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
        if not self._optimization_prepared:
            named_optimizers = self._named_optimizers(self.optimizer)
            self._uses_named_optimizers = named_optimizers is not None
            self._optimizer_names = (
                tuple(named_optimizers) if named_optimizers is not None else ("default", ))
            prepared_model, prepared_optimizer, prepared_scheduler = (
                self.training_strategy.prepare_optimization(
                    self.model_wrapped,
                    self.optimizer,
                    self.lr_scheduler,
                ))
            self.model_wrapped = prepared_model
            self.optimizer = prepared_optimizer
            self.lr_scheduler = prepared_scheduler
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self._optimization_prepared = True

    @staticmethod
    def _named_optimizers(optimizer) -> Mapping[str, Any] | None:
        """Return a routed optimizer mapping, including strategy proxies."""
        optimizers = getattr(optimizer, "optimizers", None)
        return optimizers if isinstance(optimizers, Mapping) else None

    def _optimizer_training_steps(
        self,
        optimizer_name: str,
        num_training_steps: int,
    ) -> int:
        """Count scheduled global steps for one named optimizer."""
        if self.training_adapter is None:
            return num_training_steps
        phases = tuple(
            phase for phase in self.training_adapter.spec.phases if optimizer_name in phase.optimizer_names)
        if not phases:
            return num_training_steps
        return max(
            1,
            sum(any(phase.is_scheduled(step) for phase in phases) for step in range(num_training_steps)),
        )

    def _prepare_input(self, value):
        return self.training_strategy.prepare_input(
            value,
            device=self.args.device,
        )

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
    def _get_loss(outputs):
        if isinstance(outputs, dict):
            return outputs.get("loss")
        elif hasattr(outputs, "loss"):
            return outputs.loss
        elif isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        return None

    @classmethod
    def _extract_loss(cls, outputs):
        loss = cls._get_loss(outputs)
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
        return self.training_strategy.autocast_context(self.args)

    @staticmethod
    def _find_batch_size(inputs) -> int | None:
        if isinstance(inputs, Mapping):
            for value in inputs.values():
                batch_size = Trainer._find_batch_size(value)
                if batch_size is not None:
                    return batch_size
            return None
        if hasattr(inputs, "shape") and len(inputs.shape) > 0:
            return int(inputs.shape[0])
        if isinstance(inputs, (list, tuple)):
            return len(inputs)
        return None

    def training_step(
        self,
        model,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
        *,
        sync_gradients: bool = True,
    ):
        """Compute, normalize, and backpropagate one micro-batch loss."""
        self._set_model_mode(training=True)
        inputs = self._prepare_inputs(inputs)
        with self.training_strategy.no_sync(
                model,
                enabled=not sync_gradients,
        ):
            if (self.training_adapter is not None and self.compute_loss_func is None and
                    not self._has_explicit_training_phase(inputs)):
                return self._training_plan_step(
                    model,
                    self.training_adapter,
                    inputs,
                    num_items_in_batch=num_items_in_batch,
                )
            with self._autocast_context():
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )

            optimizer_names = self._get_output_optimizer_names(outputs)
            self._record_optimizer_activity(optimizer_names)
            self.training_strategy.backward(
                loss,
                scaler=self._scaler,
            )
        return loss.detach()

    @staticmethod
    def _has_explicit_training_phase(inputs) -> bool:
        if "training_phase" in inputs or "training_context" in inputs:
            return True
        model_inputs = inputs.get("model_inputs")
        return isinstance(model_inputs,
                          dict) and ("training_phase" in model_inputs or "training_context" in model_inputs)

    def _training_plan_step(
        self,
        model,
        adapter: BaseTrainingAdapter,
        inputs: dict[str, Any],
        *,
        num_items_in_batch: int | None,
    ):
        phases = adapter.plan_training_phases(self.state.global_step)
        if not phases:
            raise RuntimeError(
                f"Training recipe for {adapter.model_type!r} scheduled no "
                f"phase at step {self.state.global_step}.")

        detached_loss = None
        micro_optimizer_names: set[str] = set()
        flattened_inputs = adapter.flatten_model_inputs(inputs)
        for phase in phases:
            context = adapter.create_training_context(
                flattened_inputs,
                training_phase=phase,
                step=self.state.global_step,
                epoch=self.state.epoch,
                metadata={"num_items_in_batch": num_items_in_batch},
            )
            with self._autocast_context():
                output = self.training_strategy.execute_training_phase(
                    model,
                    adapter,
                    context,
                )
            loss = self._extract_loss(output)
            optimizer_names = self._get_output_optimizer_names(output)
            micro_optimizer_names.update(optimizer_names)
            self.training_strategy.backward(
                loss,
                scaler=self._scaler,
            )
            current = loss.detach()
            detached_loss = current if detached_loss is None else detached_loss + current
        self._record_optimizer_activity(tuple(micro_optimizer_names), )
        return detached_loss

    def _record_optimizer_activity(
        self,
        optimizer_names: tuple[str, ...],
    ) -> None:
        if self._uses_named_optimizers:
            names = optimizer_names or self._optimizer_names
        else:
            names = ("default", )
        names = tuple(dict.fromkeys(names))
        self._active_optimizer_names.update(name for name in names if name != "default")
        for name in names:
            self._optimizer_microstep_counts[name] = (self._optimizer_microstep_counts.get(name, 0) + 1)

    @staticmethod
    def _get_output_optimizer_names(outputs) -> tuple[str, ...]:
        if isinstance(outputs, dict):
            names = outputs.get("optimizer_names")
            metadata = outputs.get("metadata")
        else:
            names = getattr(outputs, "optimizer_names", None)
            metadata = getattr(outputs, "metadata", None)
        if names is None and isinstance(metadata, dict):
            names = metadata.get("optimizer_names")
        if names is None:
            return ()
        if isinstance(names, str):
            return (names, )
        return tuple(str(name) for name in names)

    def _optimizer_step(self) -> bool:
        runtime = self._runtime_model()
        optimizer_names = (
            tuple(sorted(self._active_optimizer_names))
            if self._uses_named_optimizers and self._active_optimizer_names else None)
        next_step = self.state.global_step + 1
        self.training_strategy.normalize_gradients(
            self.optimizer,
            self._optimizer_microstep_counts,
        )
        if self.args.max_grad_norm and self.args.max_grad_norm > 0:
            self.training_strategy.clip_grad_norm(
                runtime.parameters(),
                self.args.max_grad_norm,
                optimizer=self.optimizer,
                scaler=self._scaler,
                optimizer_names=optimizer_names,
            )
        if self.training_adapter is not None:
            self.training_adapter.on_before_optimizer_step(
                optimizer_names=optimizer_names,
                step=next_step,
            )
        did_step = self.training_strategy.optimizer_step(
            self.optimizer,
            scaler=self._scaler,
            optimizer_names=optimizer_names,
        )
        did_step = True if did_step is None else bool(did_step)
        if did_step:
            self.training_strategy.scheduler_step(
                self.lr_scheduler,
                optimizer_names=optimizer_names,
            )
            if self.training_adapter is not None:
                self.training_adapter.on_optimizer_step(
                    optimizer_names=optimizer_names,
                    step=next_step,
                )
        elif self.training_adapter is not None:
            self.training_adapter.on_optimizer_step_skipped(
                optimizer_names=optimizer_names,
                step=next_step,
            )
        self.training_strategy.zero_grad(
            self.optimizer,
            optimizer_names=None,
        )
        self._active_optimizer_names.clear()
        self._optimizer_microstep_counts.clear()
        return did_step

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
        set_seed(self.args.seed)
        if self.model_init is not None:
            self.model = self.model_init()
            self.model_wrapped = self.model
            self.training_adapter = self._create_training_adapter(
                self.model,
                None,
            )
            if (self._uses_default_data_collator and self.training_adapter is not None):
                self.data_collator = (self._dataset_data_collator or self.training_adapter.data_collator)
            self.optimizer = None
            self.lr_scheduler = None
            self._model_prepared = False
            self._optimization_prepared = False
            self._uses_named_optimizers = False
            self._optimizer_names = ("default", )
            self.callback_handler.model = self.model

        self._validate_output_dir(resume_from_checkpoint)
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
            # A finite, re-iterable IterableDataset may exhaust before
            # max_steps. Each pass can contribute at least one optimizer
            # update; max_steps is therefore a safe upper bound.
            num_train_epochs = self.args.max_steps

        self.create_optimizer_and_scheduler(max_steps)
        self._resume_topology = {
            "dataloader_length": int(steps_per_epoch),
            "updates_per_epoch": int(updates_per_epoch),
            "max_steps": int(max_steps),
        }
        checkpoint = self._resolve_checkpoint(resume_from_checkpoint)
        if checkpoint is None:
            self.state = TrainerState()
            self.state.train_epoch = 0
            self.state.train_batch_cursor = 0
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = 0
            self._loss_at_last_log = 0.0
        self._initialize_state(
            max_steps=max_steps,
            num_train_epochs=num_train_epochs,
        )
        self._scaler = self.training_strategy.create_grad_scaler(self.args)

        if checkpoint is not None:
            if not dataloader_has_length:
                raise ValueError(
                    "Exact generic resume requires a dataloader with a stable "
                    "length and cursor. Use a stateful strategy/dataloader for "
                    "iterable datasets.")
            if self.args.dataloader_num_workers != 0:
                raise ValueError(
                    "Exact checkpoint resume requires "
                    "dataloader_num_workers=0. Worker prefetch/RNG state is "
                    "not recoverable by a generic DataLoader.")
            self._load_checkpoint(checkpoint)
            self._initialize_state(
                max_steps=max_steps,
                num_train_epochs=num_train_epochs,
            )

        self.training_strategy.zero_grad(self.optimizer)
        self._active_optimizer_names.clear()
        self._optimizer_microstep_counts.clear()
        self.control._new_training()
        self.control = self.callback_handler.on_train_begin(
            self.args,
            self.state,
            self.control,
        )
        self.is_in_train = True
        start_time = time.time()
        starting_step = self.state.global_step
        pending_loss = 0.0
        pending_microsteps = 0
        if (self.state.train_epoch is None or self.state.train_batch_cursor is None):
            epochs_trained = self.state.global_step // updates_per_epoch
            updates_in_current_epoch = (self.state.global_step % updates_per_epoch)
            batches_to_skip = (updates_in_current_epoch * self.args.gradient_accumulation_steps)
        else:
            epochs_trained = int(self.state.train_epoch)
            batches_to_skip = int(self.state.train_batch_cursor)
        if epochs_trained < 0 or batches_to_skip < 0:
            raise ValueError("Checkpoint dataloader cursor cannot be negative.")
        if dataloader_has_length and batches_to_skip > steps_per_epoch:
            raise ValueError(
                "Checkpoint dataloader cursor exceeds the current epoch "
                "length. The dataset or batching configuration changed.")
        if dataloader_has_length and batches_to_skip == steps_per_epoch:
            epochs_trained += 1
            batches_to_skip = 0
        self.state.train_epoch = epochs_trained
        self.state.train_batch_cursor = batches_to_skip

        epoch_iterator_range = (() if self.state.global_step >= max_steps else range(
            epochs_trained, num_train_epochs))
        for epoch in epoch_iterator_range:
            if self._train_sampler is not None:
                self._train_sampler.set_epoch(epoch)
            if self._train_dataloader_generator is not None:
                data_seed = (self.args.data_seed if self.args.data_seed is not None else self.args.seed)
                self._train_dataloader_generator.manual_seed(data_seed + epoch)
            self.control = self.callback_handler.on_epoch_begin(
                self.args,
                self.state,
                self.control,
            )
            epoch_iterator = iter(train_dataloader)
            skipped_batches = (batches_to_skip if epoch == epochs_trained else 0)
            for _ in range(skipped_batches):
                try:
                    next(epoch_iterator)
                except StopIteration as error:
                    raise RuntimeError(
                        "The resumed dataloader ended before the checkpoint "
                        "cursor. The dataset or batching configuration changed.") from error
            if self._deferred_rng_state:
                self._restore_deferred_rng_state()
            epoch_batch_count = 0
            for step, inputs in enumerate(
                    epoch_iterator,
                    start=skipped_batches,
            ):
                epoch_batch_count += 1
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
                is_last_batch = dataloader_has_length and step + 1 == steps_per_epoch
                should_update = ((step + 1) % self.args.gradient_accumulation_steps == 0 or is_last_batch)
                loss = self.training_step(
                    self.model_wrapped,
                    inputs,
                    num_items_in_batch=num_items,
                    sync_gradients=should_update,
                )
                next_batch_cursor = step + 1
                if (dataloader_has_length and next_batch_cursor >= steps_per_epoch):
                    self.state.train_epoch = epoch + 1
                    self.state.train_batch_cursor = 0
                else:
                    self.state.train_epoch = epoch
                    self.state.train_batch_cursor = next_batch_cursor
                self.state.epoch = (epoch + next_batch_cursor / max(1, steps_per_epoch))
                pending_loss += float(loss.item())
                pending_microsteps += 1

                if not should_update:
                    self.control = self.callback_handler.on_substep_end(
                        self.args,
                        self.state,
                        self.control,
                    )
                    continue

                did_step = self._optimizer_step()
                if not did_step:
                    pending_loss = 0.0
                    pending_microsteps = 0
                    self.control = self.callback_handler.on_substep_end(
                        self.args,
                        self.state,
                        self.control,
                    )
                    continue
                self._total_loss_scalar += (pending_loss / max(1, pending_microsteps))
                pending_loss = 0.0
                pending_microsteps = 0
                self.state.global_step += 1
                self.control = self.callback_handler.on_step_end(
                    self.args,
                    self.state,
                    self.control,
                )
                # At a sized epoch boundary, publish interval actions only
                # after on_epoch_end so callback state and the next-epoch
                # dataloader cursor are part of the same checkpoint.
                if not is_last_batch:
                    self._maybe_log_save_evaluate(self._total_loss_scalar)

                if (self.control.should_epoch_stop or self.control.should_training_stop):
                    break

            if (not dataloader_has_length and pending_microsteps > 0 and
                    not self.control.should_training_stop):
                did_step = self._optimizer_step()
                if did_step:
                    self._total_loss_scalar += (pending_loss / max(1, pending_microsteps))
                    self.state.global_step += 1
                    self.state.epoch = float(epoch + 1)
                    self.control = self.callback_handler.on_step_end(
                        self.args,
                        self.state,
                        self.control,
                    )
                    self._maybe_log_save_evaluate(self._total_loss_scalar)
                else:
                    self.control = self.callback_handler.on_substep_end(
                        self.args,
                        self.state,
                        self.control,
                    )
                pending_loss = 0.0
                pending_microsteps = 0
            if (not dataloader_has_length and epoch_batch_count == 0 and self.state.global_step < max_steps):
                raise RuntimeError(
                    "The iterable training dataset produced no batches before "
                    "max_steps was reached.")

            self.control = self.callback_handler.on_epoch_end(
                self.args,
                self.state,
                self.control,
            )
            self._maybe_log_save_evaluate(self._total_loss_scalar)
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

        completed_steps = max(0, self.state.global_step - starting_step)
        training_loss = self._total_loss_scalar / max(
            1,
            self.state.global_step,
        )
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

    def _validate_output_dir(
        self,
        resume_from_checkpoint: str | bool | None,
    ) -> None:
        output_dir = Path(self.args.output_dir)
        if resume_from_checkpoint:
            return
        last_checkpoint = get_last_checkpoint(output_dir)
        if last_checkpoint is not None and not self.args.overwrite_output_dir:
            raise FileExistsError(
                f"Output directory {str(output_dir)!r} already contains "
                f"{Path(last_checkpoint).name}. Pass `resume_from_checkpoint=True` "
                "or set `overwrite_output_dir=True`.")

    def _estimate_train_samples(self, completed_steps: int) -> int:
        return (completed_steps * self.args.train_batch_size * self.args.gradient_accumulation_steps)

    def _maybe_log_save_evaluate(self, tr_loss: float) -> None:
        metrics = None
        metric_improved = False
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
            metric_improved = self._update_best_metric(
                metrics,
                record_checkpoint=False,
            )
            if metric_improved and self.args.load_best_model_at_end:
                self.control.should_save = True

        if self.control.should_save:
            if metric_improved:
                self.state.best_model_checkpoint = str(
                    Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            self.control = self.callback_handler.on_save(
                self.args,
                self.state,
                self.control,
            )
            self._save_checkpoint()

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

    def _update_best_metric(
        self,
        metrics: dict[str, float],
        *,
        record_checkpoint: bool = True,
    ) -> bool:
        if not self.args.load_best_model_at_end:
            return False
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
            if record_checkpoint:
                self.state.best_model_checkpoint = str(
                    Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        return bool(is_better)

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
            if (self.training_adapter is not None and self.compute_loss_func is None):
                outputs = self._adapter_evaluation_forward(
                    model,
                    self.training_adapter,
                    prepared,
                    label_free=not has_labels,
                )
                output_loss = self._get_loss(outputs)
                loss = (output_loss.detach().mean() if output_loss is not None else None)
            elif has_labels:
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
            self._nested_detach(logits, cpu=False),
            self._nested_detach(labels, cpu=False),
        )

    def _adapter_evaluation_forward(
        self,
        model,
        adapter: BaseTrainingAdapter,
        inputs: Mapping[str, Any],
        *,
        label_free: bool = False,
    ):
        flattened = adapter.flatten_model_inputs(inputs)
        training_phase = flattened.pop("training_phase", None)
        supplied_context = flattened.pop("training_context", None)
        if supplied_context is not None:
            if not isinstance(supplied_context, TrainingContext):
                raise TypeError("training_context must be a TrainingContext.")
            if training_phase is not None:
                raise ValueError("Pass either training_phase or training_context, not both.")
            merged = dict(supplied_context.inputs)
            merged.update(flattened)
            flattened = merged
            phase = adapter.select_evaluation_phase(supplied_context.phase, )
        else:
            phase = adapter.select_evaluation_phase(training_phase)
        context = adapter.create_training_context(
            flattened,
            training_phase=phase,
            step=self.state.global_step,
            epoch=self.state.epoch,
            is_training=False,
            metadata={"evaluation": True},
        )
        if label_free:
            return self.training_strategy.execute_prediction_phase(
                model,
                adapter,
                context,
            )
        return self.training_strategy.execute_training_phase(
            model,
            adapter,
            context,
        )

    def _get_label_values(self, inputs):
        if self.training_adapter is not None:
            flattened = self.training_adapter.flatten_model_inputs(inputs)
            phase_control = flattened.pop("training_phase", None)
            supplied_context = flattened.pop("training_context", None)
            if supplied_context is not None:
                if not isinstance(supplied_context, TrainingContext):
                    raise TypeError("training_context must be a TrainingContext.")
                if phase_control is not None:
                    raise ValueError("Pass either training_phase or training_context, "
                                     "not both.")
                phase = self.training_adapter.select_evaluation_phase(supplied_context.phase, )
                merged = dict(supplied_context.inputs)
                merged.update(flattened)
                flattened = merged
            else:
                phase = self.training_adapter.select_evaluation_phase(phase_control, )
            names = tuple(dict.fromkeys(phase.label_names + self.training_adapter.spec.label_names))
            for name in names:
                if name in flattened:
                    return [flattened[name]]
            return []
        if (self.args.label_names and all(name in inputs for name in self.args.label_names)):
            return [inputs[name] for name in self.args.label_names]
        return []

    @staticmethod
    def _extract_predictions(outputs, *, ignore_loss: bool):
        if isinstance(outputs, dict):
            for key in (
                    "logits",
                    "audio_values",
                    "predictions",
                    "waveform",
            ):
                if outputs.get(key) is not None:
                    return outputs[key]
            values = [
                value for key, value in outputs.items() if key not in (
                    "loss",
                    "losses",
                    "hidden_states",
                    "attentions",
                    "metadata",
                    "optimizer_names",
                    "training_phase",
                )
            ]
        elif any(getattr(outputs, key, None) is not None for key in (
                "logits",
                "audio_values",
                "predictions",
                "waveform",
        )):
            return next(
                getattr(outputs, key) for key in (
                    "logits",
                    "audio_values",
                    "predictions",
                    "waveform",
                ) if getattr(outputs, key, None) is not None)
        elif isinstance(outputs, (tuple, list)):
            values = list(outputs[1:] if ignore_loss else outputs)
        elif hasattr(outputs, "loss"):
            return None
        else:
            return outputs
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return tuple(values)

    def _nested_detach(self, value, *, cpu: bool = True):
        torch = self._import_torch()
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach()
            return value.cpu() if cpu else value
        if isinstance(value, dict):
            return {key: self._nested_detach(item, cpu=cpu) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(self._nested_detach(item, cpu=cpu) for item in value)
        return value

    def _nested_concat(self, values, *, padding_value: float | int = 0):
        torch = self._import_torch()
        values = [value for value in values if value is not None]
        if not values:
            return None
        first = values[0]
        if torch.is_tensor(first):
            tensors = [value.reshape(1) if value.ndim == 0 else value for value in values]
            ranks = {tensor.ndim for tensor in tensors}
            if len(ranks) != 1:
                return tensors
            rank = tensors[0].ndim
            if rank > 1:
                target_shape = [
                    max(tensor.shape[dimension] for tensor in tensors) for dimension in range(1, rank)
                ]
                padded = []
                for tensor in tensors:
                    padding = []
                    for dimension in range(rank - 1, 0, -1):
                        padding.extend((
                            0,
                            target_shape[dimension - 1] - tensor.shape[dimension],
                        ))
                    if any(padding):
                        tensor = torch.nn.functional.pad(
                            tensor,
                            tuple(padding),
                            value=padding_value,
                        )
                    padded.append(tensor)
                tensors = padded
            return torch.cat(tensors, dim=0)
        if isinstance(first, dict):
            return {
                key: self._nested_concat(
                    [value[key] for value in values],
                    padding_value=padding_value,
                )
                for key in first
            }
        if isinstance(first, (tuple, list)):
            return type(first)(
                self._nested_concat(
                    [value[index] for value in values],
                    padding_value=padding_value,
                ) for index in range(len(first)))
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
        losses: list[tuple[float, int]] = []
        prediction_batches = []
        label_batches = []
        observed_samples = 0

        for inputs in dataloader:
            batch_size = self._find_batch_size(inputs) or 1
            loss, predictions, labels = self.prediction_step(
                self.model_wrapped,
                inputs,
                prediction_loss_only,
            )
            gathered = self.training_strategy.gather_for_metrics({
                "loss": loss,
                "predictions": predictions,
                "labels": labels,
                "batch_size": batch_size,
            })
            if not isinstance(gathered, dict):
                raise TypeError("gather_for_metrics() must preserve the evaluation "
                                "payload mapping.")
            loss = gathered.get("loss")
            predictions = self._nested_detach(gathered.get("predictions"))
            labels = self._nested_detach(gathered.get("labels"))
            gathered_batch_size = self._sum_batch_sizes(gathered.get("batch_size", batch_size))
            observed_samples += gathered_batch_size
            if loss is not None:
                losses.extend(self._loss_items(loss, gathered.get(
                    "batch_size",
                    batch_size,
                )))
            if predictions is not None:
                prediction_batches.append(predictions)
            if labels is not None:
                label_batches.append(labels)
            self.control = self.callback_handler.on_prediction_step(
                self.args,
                self.state,
                self.control,
            )

        predictions = self._nested_numpify(self._nested_concat(
            prediction_batches,
            padding_value=0,
        ))
        label_ids = self._nested_numpify(self._nested_concat(
            label_batches,
            padding_value=-100,
        ))
        metrics: dict[str, Any] = {}
        if losses:
            loss_items = sum(batch_size for _, batch_size in losses)
            metrics[f"{metric_key_prefix}_loss"] = (
                sum(loss * batch_size for loss, batch_size in losses) / max(1, loss_items))
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

    def _sum_batch_sizes(self, value) -> int:
        torch = self._import_torch()
        if torch.is_tensor(value):
            return int(value.detach().sum().item())
        if isinstance(value, (tuple, list)):
            return sum(self._sum_batch_sizes(item) for item in value)
        return int(value)

    def _loss_items(self, loss, batch_sizes) -> list[tuple[float, int]]:
        torch = self._import_torch()
        if torch.is_tensor(loss):
            values = loss.detach().reshape(-1).cpu().tolist()
        elif isinstance(loss, (tuple, list)):
            values = [float(value) for value in loss]
        else:
            values = [float(loss)]
        if torch.is_tensor(batch_sizes):
            sizes = batch_sizes.detach().reshape(-1).cpu().tolist()
        elif isinstance(batch_sizes, (tuple, list)):
            sizes = [self._sum_batch_sizes(value) for value in batch_sizes]
        else:
            sizes = [int(batch_sizes)]
        if len(values) == 1 and len(sizes) > 1:
            sizes = [sum(sizes)]
        if len(values) != len(sizes):
            raise ValueError("gather_for_metrics() returned incompatible loss and "
                             "batch-size shapes.")
        return [(float(value), int(size)) for value, size in zip(values, sizes)]

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

    def save_model(
        self,
        output_dir: str | Path | None = None,
        *,
        include_native_export: bool = True,
    ) -> Path:
        """Save a portable VoiceHub artifact and optional native export."""
        torch = self._import_torch()
        destination = Path(output_dir or self.args.output_dir).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        if isinstance(self.model, PreTrainedTTSModel):
            self.model.save_pretrained(
                destination,
                include_native_export=False,
            )
        elif hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(destination)
        runtime = self.training_strategy.unwrap_model(self.model_wrapped)
        if hasattr(runtime, "state_dict"):
            torch.save(runtime.state_dict(), destination / MODEL_STATE_NAME)
        if self.training_adapter is not None:
            recipe_manifest = self.training_adapter.artifact_manifest()
            if include_native_export:
                native_destination = destination / NATIVE_EXPORT_DIR
                native_destination.mkdir(parents=True, exist_ok=True)
                self.training_adapter.save_pretrained(native_destination)
                if not any(native_destination.iterdir()):
                    native_destination.rmdir()
                if native_destination.is_dir():
                    recipe_manifest["native_export_path"] = (NATIVE_EXPORT_DIR)
            write_json(
                destination / TRAINING_RECIPE_NAME,
                recipe_manifest,
            )
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
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = (checkpoint.parent / f".{checkpoint.name}.incomplete-{uuid.uuid4().hex}")
        temporary.mkdir()
        try:
            self.save_model(
                temporary,
                include_native_export=False,
            )
            self.state.save_to_json(temporary / TRAINER_STATE_NAME)
            torch.save(
                self.optimizer.state_dict(),
                temporary / OPTIMIZER_NAME,
            )
            torch.save(
                self.lr_scheduler.state_dict(),
                temporary / SCHEDULER_NAME,
            )
            rng_state = {"cpu": torch.random.get_rng_state()}
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.random.get_rng_state_all()
            mps = getattr(torch, "mps", None)
            if mps is not None and hasattr(mps, "get_rng_state"):
                try:
                    rng_state["mps"] = mps.get_rng_state()
                except RuntimeError:
                    pass
            torch.save(rng_state, temporary / RNG_STATE_NAME)
            if self._scaler is not None:
                torch.save(
                    self._scaler.state_dict(),
                    temporary / SCALER_STATE_NAME,
                )
            torch.save(
                self._runtime_checkpoint_state(),
                temporary / TRAINING_RUNTIME_STATE_NAME,
            )
            write_json(
                temporary / CHECKPOINT_MANIFEST_NAME,
                self._checkpoint_manifest(temporary),
            )
            (temporary / CHECKPOINT_COMPLETE_NAME).write_text(
                "complete\n",
                encoding="utf-8",
            )
            if checkpoint.exists():
                raise FileExistsError(
                    f"Checkpoint already exists: {checkpoint}. Refusing to "
                    "delete a valid checkpoint during atomic save.")
            temporary.replace(checkpoint)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
        self._rotate_checkpoints()
        return checkpoint

    def _checkpoint_manifest(self, checkpoint: Path) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        model_type = getattr(config, "model_type", None)
        optimizer_names = self._optimizer_names
        adapter = self.training_adapter
        adapter_version = getattr(adapter, "ADAPTER_STATE_VERSION", None)
        required_files = list(FORMAT_V2_REQUIRED_FILES)
        if self._scaler is not None:
            required_files.append(SCALER_STATE_NAME)
        if adapter is not None:
            required_files.append(TRAINING_RECIPE_NAME)
        return {
            "format_version":
            CHECKPOINT_FORMAT_VERSION,
            "global_step":
            self.state.global_step,
            "model_type":
            model_type,
            "adapter_class":
            (f"{type(adapter).__module__}.{type(adapter).__qualname__}" if adapter is not None else None),
            "adapter_state_version":
            adapter_version,
            "recipe_id": (adapter.recipe_id if adapter is not None else None),
            "recipe_version": (adapter.RECIPE_VERSION if adapter is not None else None),
            "optimizer_names":
            list(optimizer_names),
            "training_strategy":
            self.training_strategy.name,
            "resume_signature":
            self._build_resume_signature(),
            "required_files":
            required_files,
            "file_integrity": {
                name: {
                    "size": (checkpoint / name).stat().st_size,
                    "sha256": self._sha256(checkpoint / name),
                }
                for name in required_files
            },
        }

    @staticmethod
    def _qualified_class_name(value: Any) -> str:
        value_type = type(value)
        return f"{value_type.__module__}.{value_type.__qualname__}"

    @classmethod
    def _runtime_object_signature(
        cls,
        value: Any,
        *,
        collection_attribute: str,
    ) -> dict[str, dict[str, Any]]:
        collection = getattr(value, collection_attribute, None)
        objects = (collection if isinstance(collection, Mapping) else {
            "default": value,
        })
        signature = {}
        for name, item in objects.items():
            record = {
                "class": cls._qualified_class_name(item),
                "fingerprint": cls._resume_fingerprint(
                    item,
                    owner=f"{collection_attribute}[{name!r}]",
                ),
            }
            groups = getattr(item, "param_groups", None)
            if isinstance(groups, list):
                record["parameter_groups"] = len(groups)
            signature[str(name)] = record
        return signature

    @staticmethod
    def _resume_fingerprint(value: Any, *, owner: str) -> Any:
        fingerprint = getattr(value, "resume_fingerprint", None)
        if fingerprint is None:
            bound_owner = getattr(value, "__self__", None)
            fingerprint = getattr(
                bound_owner,
                "resume_fingerprint",
                None,
            )
        if fingerprint is None:
            return None
        fingerprint = fingerprint() if callable(fingerprint) else fingerprint
        normalized = BaseTrainingAdapter._manifest_value(fingerprint)
        try:
            json.dumps(
                normalized,
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{owner}.resume_fingerprint must return JSON-compatible "
                "deterministic data.") from exc
        return normalized

    def _build_resume_signature(self) -> dict[str, Any]:
        """Build the resolved topology required for exact continuation."""
        if self._resume_topology is None:
            raise RuntimeError(
                "Cannot create an exact-resume signature before the training "
                "dataloader and schedule have been resolved.")
        optimizer_steps = {}
        for name in self._optimizer_names:
            total_steps = self._optimizer_training_steps(
                name,
                self._resume_topology["max_steps"],
            )
            optimizer_steps[name] = {
                "total_steps": total_steps,
                "warmup_steps": self.args.get_warmup_steps(total_steps),
            }
        try:
            dataset_length = len(self.train_dataset)
        except TypeError:
            dataset_length = None
        stateful_callbacks = []
        for callback in self.callback_handler.callbacks:
            if type(callback).state_dict is TrainerCallback.state_dict:
                continue
            callback_class = self._qualified_class_name(callback)
            stateful_callbacks.append({
                "class":
                callback_class,
                "fingerprint":
                self._resume_fingerprint(
                    callback,
                    owner=f"callback {callback_class!r}",
                ),
            })
        strategy_signature = self.training_strategy.resume_signature()
        if not isinstance(strategy_signature, Mapping):
            raise TypeError("TrainingStrategy.resume_signature() must return a mapping.")
        adapter_signature = None
        if self.training_adapter is not None:
            adapter_signature = self.training_adapter.resume_signature()
            if not isinstance(adapter_signature, Mapping):
                raise TypeError("BaseTrainingAdapter.resume_signature() must return a "
                                "mapping.")
            adapter_signature = BaseTrainingAdapter._manifest_value(adapter_signature, )
        signature = {
            "data": {
                "train_batch_size":
                self.args.train_batch_size,
                "gradient_accumulation_steps": (self.args.gradient_accumulation_steps),
                "dataloader_drop_last":
                self.args.dataloader_drop_last,
                "dataloader_num_workers":
                self.args.dataloader_num_workers,
                "effective_data_seed":
                (self.args.data_seed if self.args.data_seed is not None else self.args.seed),
                "dataset_length":
                dataset_length,
                "dataloader_length": (self._resume_topology["dataloader_length"]),
                "updates_per_epoch": (self._resume_topology["updates_per_epoch"]),
                "dataset": {
                    "class": self._qualified_class_name(self.train_dataset),
                    "fingerprint": self._resume_fingerprint(
                        self.train_dataset,
                        owner="train_dataset",
                    ),
                },
                "collator": {
                    "class": self._qualified_class_name(self.data_collator),
                    "fingerprint": self._resume_fingerprint(
                        self.data_collator,
                        owner="data_collator",
                    ),
                },
            },
            "schedule": {
                "max_steps": self._resume_topology["max_steps"],
                "lr_scheduler_type": self.args.lr_scheduler_type.value,
                "optimizer_steps": optimizer_steps,
            },
            "optimization": {
                "optimizer": self._runtime_object_signature(
                    self.optimizer,
                    collection_attribute="optimizers",
                ),
                "scheduler": self._runtime_object_signature(
                    self.lr_scheduler,
                    collection_attribute="schedulers",
                ),
                "max_grad_norm": self.args.max_grad_norm,
                "gradient_checkpointing": self.args.gradient_checkpointing,
            },
            "precision": {
                "fp16": self.args.fp16,
                "bf16": self.args.bf16,
                "device_type": self.args.device.split(":", 1)[0],
                "scaler": (self._qualified_class_name(self._scaler) if self._scaler is not None else None),
            },
            "input_contract": {
                "remove_unused_columns": self.args.remove_unused_columns,
                "label_names": list(self.args.label_names),
            },
            "strategy": BaseTrainingAdapter._manifest_value(strategy_signature, ),
            "adapter": adapter_signature,
            "stateful_callbacks": stateful_callbacks,
        }
        try:
            json.dumps(
                signature,
                allow_nan=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "The exact-resume signature must contain deterministic "
                "JSON-compatible values.") from exc
        return signature

    @classmethod
    def _signature_differences(
        cls,
        checkpoint_value: Any,
        current_value: Any,
        *,
        path: str = "resume_signature",
    ) -> list[str]:
        if isinstance(checkpoint_value, Mapping) and isinstance(
                current_value,
                Mapping,
        ):
            differences = []
            keys = sorted(
                set(checkpoint_value) | set(current_value),
                key=str,
            )
            for key in keys:
                child_path = f"{path}.{key}"
                if key not in checkpoint_value:
                    differences.append(f"{child_path}: missing in checkpoint")
                elif key not in current_value:
                    differences.append(f"{child_path}: missing in current run")
                else:
                    differences.extend(
                        cls._signature_differences(
                            checkpoint_value[key],
                            current_value[key],
                            path=child_path,
                        ))
            return differences
        if isinstance(checkpoint_value, list) and isinstance(
                current_value,
                list,
        ):
            if checkpoint_value == current_value:
                return []
            return [f"{path}: checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}"]
        if checkpoint_value != current_value:
            return [f"{path}: checkpoint={checkpoint_value!r}, "
                    f"current={current_value!r}"]
        return []

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _runtime_checkpoint_state(self) -> dict[str, Any]:
        runtime_state: dict[str, Any] = {
            "python_rng": random.getstate(),
            "callbacks": self.callback_handler.state_dict(),
            "training_strategy": self.training_strategy.state_dict(),
            "logging": {
                "total_loss_scalar": self._total_loss_scalar,
                "globalstep_last_logged": self._globalstep_last_logged,
                "loss_at_last_log": self._loss_at_last_log,
            },
        }
        try:
            numpy = import_module("numpy")
            runtime_state["numpy_rng"] = numpy.random.get_state()
        except ModuleNotFoundError:
            pass
        if self._train_sampler is not None and hasattr(self._train_sampler, "state_dict"):
            runtime_state["sampler"] = self._train_sampler.state_dict()
        return runtime_state

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
        self._validate_checkpoint_directory(checkpoint)
        return str(checkpoint)

    @staticmethod
    def _validate_checkpoint_directory(checkpoint: Path) -> None:
        manifest = checkpoint / CHECKPOINT_MANIFEST_NAME
        if manifest.is_file():
            if not (checkpoint / CHECKPOINT_COMPLETE_NAME).is_file():
                raise RuntimeError(f"Checkpoint is incomplete: {checkpoint}")
            return
        missing = [name for name in LEGACY_RESUME_FILES if not (checkpoint / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Legacy resume checkpoint is missing required state: " + ", ".join(missing))

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
        runtime = self.training_strategy.unwrap_model(self.model_wrapped)
        if not hasattr(runtime, "load_state_dict"):
            raise TypeError("The trainable model does not implement `load_state_dict()`.")
        runtime.load_state_dict(self._torch_load(state_path))

    def _load_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint_path = Path(checkpoint)
        self._validate_checkpoint_directory(checkpoint_path)
        self._validate_checkpoint_manifest(checkpoint_path)
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
        scaler_path = checkpoint_path / SCALER_STATE_NAME
        if scaler_path.is_file() and self._scaler is not None:
            self._scaler.load_state_dict(self._torch_load(scaler_path))
        runtime_path = checkpoint_path / TRAINING_RUNTIME_STATE_NAME
        if runtime_path.is_file():
            self._load_runtime_checkpoint_state(self._torch_load(runtime_path), )
        rng_path = checkpoint_path / RNG_STATE_NAME
        if rng_path.is_file():
            self._deferred_rng_state["torch"] = self._torch_load(rng_path)

    def _validate_checkpoint_manifest(self, checkpoint: Path) -> None:
        manifest_path = checkpoint / CHECKPOINT_MANIFEST_NAME
        if not manifest_path.is_file():
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        format_version = manifest.get("format_version")
        if (isinstance(format_version, bool) or not isinstance(format_version, int) or format_version <= 0):
            raise ValueError("Checkpoint manifest has no valid format_version.")
        if format_version > CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"Checkpoint format {format_version} is newer than supported "
                f"format {CHECKPOINT_FORMAT_VERSION}.")
        required_files = manifest.get("required_files", ())
        if (not isinstance(required_files, list) or
                any(not isinstance(name, str) or Path(name).name != name or name in ("", ".", "..")
                    for name in required_files)):
            raise ValueError("Checkpoint manifest has an invalid required_files field.")
        if (format_version >= 2 and not set(FORMAT_V2_REQUIRED_FILES).issubset(required_files)):
            raise ValueError("Checkpoint manifest omits required Trainer state files.")
        missing_files = [name for name in required_files if not (checkpoint / name).is_file()]
        if missing_files:
            raise FileNotFoundError(
                "Checkpoint is incomplete; missing required files: " + ", ".join(sorted(missing_files)))
        integrity = manifest.get("file_integrity")
        if format_version >= 2 and not isinstance(integrity, dict):
            raise ValueError("Checkpoint manifest is missing file_integrity metadata.")
        if isinstance(integrity, dict):
            for name in required_files:
                record = integrity.get(name)
                if not isinstance(record, dict):
                    raise ValueError(f"Checkpoint integrity metadata is missing {name!r}.")
                path = checkpoint / name
                if record.get("size") != path.stat().st_size:
                    raise ValueError(f"Checkpoint file size does not match for {name!r}.")
                expected_digest = record.get("sha256")
                if (not isinstance(expected_digest, str) or self._sha256(path) != expected_digest):
                    raise ValueError(f"Checkpoint checksum does not match for {name!r}.")
        checkpoint_step = manifest.get("global_step")
        if (isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int) or checkpoint_step < 0):
            raise ValueError("Checkpoint manifest has no valid global_step.")
        try:
            directory_step = int(checkpoint.name.rsplit("-", 1)[1])
        except (IndexError, ValueError):
            directory_step = checkpoint_step
        if directory_step != checkpoint_step:
            raise ValueError(
                "Checkpoint directory step does not match its manifest "
                f"({directory_step} != {checkpoint_step}).")
        trainer_state_path = checkpoint / TRAINER_STATE_NAME
        if trainer_state_path.is_file():
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"), )
            if trainer_state.get("global_step") != checkpoint_step:
                raise ValueError("Checkpoint Trainer state does not match its manifest "
                                 "global_step.")
        checkpoint_uses_scaler = SCALER_STATE_NAME in required_files
        current_uses_scaler = self._scaler is not None
        if checkpoint_uses_scaler != current_uses_scaler:
            raise ValueError(
                "Checkpoint gradient-scaler topology does not match the "
                "current precision configuration.")
        current_type = getattr(
            getattr(self.model, "config", None),
            "model_type",
            None,
        )
        checkpoint_type = manifest.get("model_type")
        if (current_type is not None and checkpoint_type is not None and current_type != checkpoint_type):
            raise ValueError(
                f"Checkpoint model type {checkpoint_type!r} does not match "
                f"current model type {current_type!r}.")
        checkpoint_strategy = manifest.get("training_strategy")
        if checkpoint_strategy != self.training_strategy.name:
            raise ValueError(
                f"Checkpoint uses training strategy {checkpoint_strategy!r}, "
                f"not {self.training_strategy.name!r}.")
        adapter = self.training_adapter
        checkpoint_adapter = manifest.get("adapter_class")
        current_adapter = (
            f"{type(adapter).__module__}.{type(adapter).__qualname__}" if adapter is not None else None)
        if checkpoint_adapter != current_adapter:
            raise ValueError(
                f"Checkpoint adapter {checkpoint_adapter!r} does not match "
                f"the current adapter {current_adapter!r}.")
        checkpoint_adapter_version = manifest.get("adapter_state_version")
        current_adapter_version = getattr(adapter, "ADAPTER_STATE_VERSION", None)
        if checkpoint_adapter_version != current_adapter_version:
            raise ValueError(
                "Checkpoint adapter state version does not match the current "
                f"adapter ({checkpoint_adapter_version!r} != "
                f"{current_adapter_version!r}).")
        checkpoint_recipe = manifest.get("recipe_id")
        current_recipe = adapter.recipe_id if adapter is not None else None
        checkpoint_recipe_version = manifest.get("recipe_version")
        current_recipe_version = (adapter.RECIPE_VERSION if adapter is not None else None)
        has_recipe_metadata = ("recipe_id" in manifest or "recipe_version" in manifest)
        recipe_mismatch = (
            checkpoint_recipe != current_recipe or checkpoint_recipe_version != current_recipe_version)
        if has_recipe_metadata and recipe_mismatch:
            raise ValueError(
                "Checkpoint recipe identity/version does not match the "
                f"current recipe ({checkpoint_recipe!r} "
                f"v{checkpoint_recipe_version!r} != {current_recipe!r} "
                f"v{current_recipe_version!r}).")
        expected_names = set(self._optimizer_names)
        optimizer_names = manifest.get("optimizer_names")
        if (not isinstance(optimizer_names, list) or not optimizer_names or
                any(not isinstance(name, str) or not name for name in optimizer_names)):
            raise ValueError("Checkpoint manifest has invalid optimizer_names.")
        received_names = set(optimizer_names)
        if expected_names != received_names:
            raise ValueError(
                "Checkpoint optimizer topology does not match the current "
                f"recipe ({sorted(received_names)} != {sorted(expected_names)}).")
        if format_version >= 3:
            checkpoint_signature = manifest.get("resume_signature")
            if not isinstance(checkpoint_signature, Mapping):
                raise ValueError("Checkpoint manifest is missing its exact-resume "
                                 "signature.")
            current_signature = self._build_resume_signature()
            differences = self._signature_differences(
                checkpoint_signature,
                current_signature,
            )
            if differences:
                preview = "; ".join(differences[:8])
                if len(differences) > 8:
                    preview += f"; and {len(differences) - 8} more"
                raise ValueError(
                    "Checkpoint exact-resume signature does not match the "
                    f"current training plan: {preview}. Load the saved "
                    "VoiceHub artifact as a weight-only warm start when "
                    "starting a different plan.")
            self._strict_runtime_resume = True

    def _load_runtime_checkpoint_state(self, runtime_state) -> None:
        if "python_rng" in runtime_state:
            self._deferred_rng_state["python"] = runtime_state["python_rng"]
        if "numpy_rng" in runtime_state:
            self._deferred_rng_state["numpy"] = runtime_state["numpy_rng"]
        if "callbacks" in runtime_state:
            self.callback_handler.load_state_dict(
                runtime_state["callbacks"],
                strict=self._strict_runtime_resume,
            )
        if "training_strategy" in runtime_state:
            self.training_strategy.load_state_dict(runtime_state["training_strategy"], )
        if ("sampler" in runtime_state and self._train_sampler is not None and
                hasattr(self._train_sampler, "load_state_dict")):
            self._train_sampler.load_state_dict(runtime_state["sampler"])

        logging_state = runtime_state.get("logging")
        if isinstance(logging_state, dict):
            self._total_loss_scalar = float(logging_state.get("total_loss_scalar", 0.0))
            self._globalstep_last_logged = int(logging_state.get("globalstep_last_logged", 0))
            self._loss_at_last_log = float(logging_state.get("loss_at_last_log", 0.0))

    def _restore_deferred_rng_state(self) -> None:
        """Restore post-checkpoint RNG immediately before the next batch."""
        torch = self._import_torch()
        state = self._deferred_rng_state
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            try:
                numpy = import_module("numpy")
                numpy.random.set_state(state["numpy"])
            except ModuleNotFoundError:
                pass
        rng_state = state.get("torch")
        if rng_state is not None:
            torch.random.set_rng_state(rng_state["cpu"])
            if "cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(rng_state["cuda"])
            mps = getattr(torch, "mps", None)
            if ("mps" in rng_state and mps is not None and hasattr(mps, "set_rng_state")):
                mps.set_rng_state(rng_state["mps"])
        self._deferred_rng_state = {}
