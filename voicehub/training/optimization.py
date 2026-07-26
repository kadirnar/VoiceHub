"""Small optimizer/scheduler bundles for composite TTS architectures."""

from __future__ import annotations

from collections.abc import Mapping


class OptimizerBundle:
    """Present multiple named optimizers through the standard optimizer API."""

    def __init__(self, optimizers: Mapping[str, object]):
        if not optimizers:
            raise ValueError("OptimizerBundle requires at least one optimizer.")
        self.optimizers = dict(optimizers)

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers.values():
            try:
                optimizer.zero_grad(set_to_none=set_to_none)
            except TypeError:
                optimizer.zero_grad()

    def step(self) -> None:
        for optimizer in self.optimizers.values():
            optimizer.step()

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict):
        for name, optimizer_state in state_dict.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(optimizer_state)


class SchedulerBundle:
    """Keep one scheduler aligned with each named optimizer."""

    def __init__(self, schedulers: Mapping[str, object]):
        if not schedulers:
            raise ValueError("SchedulerBundle requires at least one scheduler.")
        self.schedulers = dict(schedulers)

    def step(self) -> None:
        for scheduler in self.schedulers.values():
            scheduler.step()

    def state_dict(self):
        return {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()}

    def load_state_dict(self, state_dict):
        for name, scheduler_state in state_dict.items():
            if name in self.schedulers:
                self.schedulers[name].load_state_dict(scheduler_state)
