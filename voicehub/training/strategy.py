"""Execution strategies for optimization, precision, and device handling.

The Trainer owns orchestration while a strategy owns framework-specific
execution.  Keeping that boundary explicit lets Accelerate, DeepSpeed,
FSDP, quantized optimizers, or another training runtime integrate
without changing model recipes.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

from voicehub.dependencies import import_optional


class TrainingStrategy:
    """Framework-facing operations used by :class:`voicehub.Trainer`.

    Custom strategies may wrap models and optimizers, implement
    distributed gradient synchronization, or persist runtime-specific
    state.  The default implementation is deliberately small and single-
    process.
    """

    name = "base"

    def prepare_model(self, model, *, device: str):
        """Move or wrap a trainable model and return the runtime object."""
        if hasattr(model, "to"):
            model.to(device)
        return model

    def prepare_training_adapter(self, adapter, *, device: str):
        """Prepare a phase adapter and return its strategy execution handle.

        Distributed strategies can wrap individual adapter components
        and return a proxy handle here. They must then implement
        :meth:`execute_training_phase` and :meth:`unwrap_model`.
        """
        adapter.to(device)
        return adapter

    def prepare_optimization(self, model, optimizer, scheduler):
        """Prepare optimizer/scheduler state, optionally together with a
        model."""
        return model, optimizer, scheduler

    def prepare_dataloader(self, dataloader, *, training: bool):
        """Wrap a dataloader for the execution runtime."""
        return dataloader

    def prepare_input(self, value, *, device: str):
        """Move one nested input value to the execution device."""
        return value

    def autocast_context(self, args):
        """Return the forward-pass precision context."""
        return nullcontext()

    def create_grad_scaler(self, args):
        """Create optional mixed-precision scaler state."""
        return None

    def backward(self, loss, *, scaler=None) -> None:
        """Backpropagate one differentiable loss."""
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def clip_grad_norm(
        self,
        parameters,
        max_norm: float,
        *,
        optimizer=None,
        scaler=None,
        optimizer_names: tuple[str, ...] | None = None,
    ):
        """Clip gradients and return the total norm when available."""
        return None

    def optimizer_step(
        self,
        optimizer,
        *,
        scaler=None,
        optimizer_names: tuple[str, ...] | None = None,
    ) -> bool:
        """Advance the selected optimizer components."""
        optimizers = getattr(optimizer, "optimizers", None)
        if scaler is not None:
            selected = (
                tuple(optimizers.values()) if optimizers is not None and optimizer_names is None else tuple(
                    optimizers[name] for name in optimizer_names) if optimizers is not None else
                (optimizer, ))
            previous_scale = scaler.get_scale()
            for selected_optimizer in selected:
                scaler.step(selected_optimizer)
            scaler.update()
            return scaler.get_scale() >= previous_scale
        if optimizers is not None:
            optimizer.step(names=optimizer_names)
        else:
            optimizer.step()
        return True

    def scheduler_step(
        self,
        scheduler,
        *,
        optimizer_names: tuple[str, ...] | None = None,
        metric: float | None = None,
    ) -> None:
        """Advance schedulers associated with the selected optimizers."""
        schedulers = getattr(scheduler, "schedulers", None)
        if schedulers is not None:
            scheduler.step(names=optimizer_names, metric=metric)
        elif metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def zero_grad(
        self,
        optimizer,
        *,
        optimizer_names: tuple[str, ...] | None = None,
    ) -> None:
        """Clear selected optimizer gradients."""
        try:
            optimizers = getattr(optimizer, "optimizers", None)
            if optimizers is not None:
                optimizer.zero_grad(
                    names=optimizer_names,
                    set_to_none=True,
                )
            else:
                optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

    def normalize_gradients(
        self,
        optimizer,
        microstep_counts: dict[str, int],
    ) -> None:
        """Average gradients by each routed optimizer's active micro-
        batches."""
        optimizers = getattr(optimizer, "optimizers", None)
        selected = (optimizers.items() if optimizers is not None else (("default", optimizer), ))
        seen = set()
        for name, selected_optimizer in selected:
            divisor = max(1, int(microstep_counts.get(name, 1)))
            for group in selected_optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) in seen or parameter.grad is None:
                        continue
                    seen.add(id(parameter))
                    parameter.grad.div_(divisor)

    def no_sync(self, model, *, enabled: bool):
        """Return a context that suppresses distributed gradient sync."""
        if enabled:
            no_sync = getattr(model, "no_sync", None)
            if callable(no_sync):
                return no_sync()
        return nullcontext()

    def gather_for_metrics(self, value):
        """Gather a metric value across workers."""
        return value

    def execute_training_phase(self, model, adapter, context):
        """Execute one adapter phase through the strategy-owned handle."""
        if model is adapter:
            return adapter.compute_step(context)
        if callable(model):
            return model(training_context=context)
        raise TypeError(
            "A strategy that wraps a training adapter must return a callable "
            "execution handle or override execute_training_phase().")

    def execute_prediction_phase(self, model, adapter, context):
        """Execute a label-free adapter forward through the strategy handle."""
        if model is adapter:
            return adapter.execute_prediction_phase(context)
        raise TypeError(
            "A strategy that wraps a training adapter must override "
            "execute_prediction_phase() for label-free prediction.")

    def state_dict(self) -> dict[str, Any]:
        """Return runtime-specific checkpoint state."""
        return {}

    def resume_signature(self) -> dict[str, Any]:
        """Return stable topology required for exact checkpoint resume.

        Distributed strategies should override this with their world
        size, sharding layout, accumulation semantics, and other
        topology controls.
        """
        return {
            "name": self.name,
            "world_size": 1,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore runtime-specific checkpoint state."""

    def unwrap_model(self, model):
        """Return the underlying model used for serialization."""
        return model


class TorchTrainingStrategy(TrainingStrategy):
    """Default lazy PyTorch execution strategy."""

    name = "torch"

    def __init__(self):
        self._torch = None
        self._unscaled_optimizer_ids: set[int] = set()

    def _import_torch(self):
        if self._torch is None:
            self._torch = import_optional(
                "torch",
                model_type="Trainer",
                install_extra="training",
            )
        return self._torch

    def prepare_input(self, value, *, device: str):
        torch = self._import_torch()
        if isinstance(value, dict):
            return {key: self.prepare_input(item, device=device) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.prepare_input(item, device=device) for item in value)
        if isinstance(value, list):
            return [self.prepare_input(item, device=device) for item in value]
        if torch.is_tensor(value):
            return value.to(device)
        return value

    def autocast_context(self, args):
        torch = self._import_torch()
        device_type = args.device.split(":", 1)[0]
        enabled = (args.fp16 or args.bf16) and device_type in (
            "cuda",
            "cpu",
        )
        if not enabled:
            return nullcontext()
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        return torch.autocast(device_type=device_type, dtype=dtype)

    def create_grad_scaler(self, args):
        if not args.fp16:
            return None
        torch = self._import_torch()
        amp = getattr(torch, "amp", None)
        if amp is not None and hasattr(amp, "GradScaler"):
            try:
                return amp.GradScaler("cuda")
            except TypeError:
                return amp.GradScaler()
        return torch.cuda.amp.GradScaler()

    @staticmethod
    def _named_optimizers(optimizer, optimizer_names):
        optimizers = getattr(optimizer, "optimizers", None)
        if optimizers is None:
            return (optimizer, )
        if optimizer_names is None:
            return tuple(optimizers.values())
        unknown = set(optimizer_names) - set(optimizers)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"Unknown optimizer component(s): {names}.")
        return tuple(optimizers[name] for name in optimizer_names)

    @classmethod
    def _selected_parameters(cls, optimizer, optimizer_names):
        seen = set()
        parameters = []
        for selected_optimizer in cls._named_optimizers(
                optimizer,
                optimizer_names,
        ):
            for group in selected_optimizer.param_groups:
                for parameter in group["params"]:
                    if id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return tuple(parameters)

    def _unscale_optimizers(self, scaler, optimizers) -> None:
        for optimizer in optimizers:
            identity = id(optimizer)
            if identity in self._unscaled_optimizer_ids:
                continue
            scaler.unscale_(optimizer)
            self._unscaled_optimizer_ids.add(identity)

    def _gradients_are_finite(self, optimizers) -> bool:
        torch = self._import_torch()
        seen = set()
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    if getattr(gradient, "is_sparse", False):
                        gradient = gradient.coalesce().values()
                    if not bool(torch.isfinite(gradient).all().item()):
                        return False
        return True

    def clip_grad_norm(
        self,
        parameters,
        max_norm: float,
        *,
        optimizer=None,
        scaler=None,
        optimizer_names: tuple[str, ...] | None = None,
    ):
        torch = self._import_torch()
        if scaler is not None and optimizer is not None:
            self._unscale_optimizers(
                scaler,
                self._named_optimizers(
                    optimizer,
                    optimizer_names,
                ),
            )
        if optimizer is not None:
            parameters = self._selected_parameters(
                optimizer,
                optimizer_names,
            )
        return torch.nn.utils.clip_grad_norm_(
            tuple(parameters),
            max_norm,
        )

    def optimizer_step(
        self,
        optimizer,
        *,
        scaler=None,
        optimizer_names: tuple[str, ...] | None = None,
    ) -> bool:
        selected = self._named_optimizers(optimizer, optimizer_names)
        if scaler is not None:
            previous_scale = scaler.get_scale()
            try:
                self._unscale_optimizers(scaler, selected)
                gradients_are_finite = self._gradients_are_finite(selected)
                if gradients_are_finite:
                    for selected_optimizer in selected:
                        scaler.step(selected_optimizer)
                scaler.update()
                return (gradients_are_finite and scaler.get_scale() >= previous_scale)
            finally:
                self._unscaled_optimizer_ids.clear()
        optimizers = getattr(optimizer, "optimizers", None)
        if optimizers is not None:
            optimizer.step(names=optimizer_names)
        else:
            optimizer.step()
        return True

    def scheduler_step(
        self,
        scheduler,
        *,
        optimizer_names: tuple[str, ...] | None = None,
        metric: float | None = None,
    ) -> None:
        schedulers = getattr(scheduler, "schedulers", None)
        if schedulers is not None:
            scheduler.step(names=optimizer_names, metric=metric)
            return
        if metric is None:
            scheduler.step()
        else:
            try:
                scheduler.step(metric)
            except TypeError:
                scheduler.step()

    def zero_grad(
        self,
        optimizer,
        *,
        optimizer_names: tuple[str, ...] | None = None,
    ) -> None:
        optimizers = getattr(optimizer, "optimizers", None)
        if optimizers is not None:
            optimizer.zero_grad(
                names=optimizer_names,
                set_to_none=True,
            )
            return
        super().zero_grad(optimizer)


_TRAINING_STRATEGIES: dict[str, Callable[[], TrainingStrategy]] = {
    TorchTrainingStrategy.name: TorchTrainingStrategy,
}


def register_training_strategy(
    name: str,
    factory: Callable[[], TrainingStrategy] | type[TrainingStrategy],
    *,
    exist_ok: bool = False,
) -> None:
    """Register a lazily constructed execution strategy."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Training strategy names cannot be empty.")
    if normalized in _TRAINING_STRATEGIES and not exist_ok:
        raise ValueError(f"A training strategy named {normalized!r} is already registered.")
    _TRAINING_STRATEGIES[normalized] = factory


def unregister_training_strategy(name: str) -> None:
    """Remove a custom strategy registration.

    The built-in PyTorch strategy cannot be removed because it is the
    Trainer default.
    """
    normalized = name.strip().lower()
    if normalized == TorchTrainingStrategy.name:
        raise ValueError("The built-in 'torch' strategy cannot be unregistered.")
    try:
        del _TRAINING_STRATEGIES[normalized]
    except KeyError as error:
        raise KeyError(f"No training strategy named {normalized!r} is registered.") from error


def get_training_strategy(strategy: str | TrainingStrategy | None = None, ) -> TrainingStrategy:
    """Resolve a strategy name or validate an existing strategy instance."""
    if strategy is None:
        strategy = TorchTrainingStrategy.name
    if isinstance(strategy, TrainingStrategy):
        return strategy
    if not isinstance(strategy, str):
        raise TypeError("`training_strategy` must be a name or TrainingStrategy instance.")
    normalized = strategy.strip().lower()
    try:
        factory = _TRAINING_STRATEGIES[normalized]
    except KeyError as error:
        available = ", ".join(sorted(_TRAINING_STRATEGIES))
        raise KeyError(
            f"Unknown training strategy {normalized!r}. Available strategies: {available}.") from error
    resolved = factory()
    if not isinstance(resolved, TrainingStrategy):
        raise TypeError("Training strategy factories must return a TrainingStrategy instance.")
    return resolved


def list_training_strategies() -> tuple[str, ...]:
    """Return registered execution strategy names."""
    return tuple(_TRAINING_STRATEGIES)
