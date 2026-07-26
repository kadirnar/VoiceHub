"""Small optimizer/scheduler bundles for composite TTS architectures."""

from __future__ import annotations

from collections.abc import Mapping


class OptimizerBundle:
    """Present multiple named optimizers through a routed optimizer API.

    ``names`` lets a training phase update only its own components.
    Calling methods without ``names`` preserves the standard optimizer-
    like behavior and operates on the complete bundle.
    """

    def __init__(self, optimizers: Mapping[str, object]):
        if not optimizers:
            raise ValueError("OptimizerBundle requires at least one optimizer.")
        self.optimizers = dict(optimizers)

    @property
    def param_groups(self):
        return [group for optimizer in self.optimizers.values() for group in optimizer.param_groups]

    def _select(self, names=None):
        if names is None:
            return tuple(self.optimizers.items())
        normalized = tuple(dict.fromkeys(names))
        unknown = set(normalized) - set(self.optimizers)
        if unknown:
            missing = ", ".join(sorted(unknown))
            available = ", ".join(self.optimizers)
            raise KeyError(
                f"Unknown optimizer component(s): {missing}. "
                f"Available components: {available}.")
        return tuple((name, self.optimizers[name]) for name in normalized)

    def zero_grad(
        self,
        set_to_none: bool = True,
        *,
        names=None,
    ) -> None:
        for _, optimizer in self._select(names):
            try:
                optimizer.zero_grad(set_to_none=set_to_none)
            except TypeError:
                optimizer.zero_grad()

    def step(self, *, names=None) -> None:
        for _, optimizer in self._select(names):
            optimizer.step()

    def state_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def load_state_dict(self, state_dict, *, strict: bool = True):
        expected = set(self.optimizers)
        received = set(state_dict)
        if strict and expected != received:
            missing = ", ".join(sorted(expected - received)) or "none"
            unexpected = ", ".join(sorted(received - expected)) or "none"
            raise ValueError(
                "Optimizer checkpoint topology does not match the current "
                f"recipe (missing: {missing}; unexpected: {unexpected}).")
        for name in expected & received:
            self.optimizers[name].load_state_dict(state_dict[name])


class SchedulerBundle:
    """Keep one scheduler aligned with each named optimizer."""

    def __init__(self, schedulers: Mapping[str, object]):
        if not schedulers:
            raise ValueError("SchedulerBundle requires at least one scheduler.")
        self.schedulers = dict(schedulers)

    def _select(self, names=None):
        if names is None:
            return tuple(self.schedulers.items())
        normalized = tuple(dict.fromkeys(names))
        unknown = set(normalized) - set(self.schedulers)
        if unknown:
            missing = ", ".join(sorted(unknown))
            available = ", ".join(self.schedulers)
            raise KeyError(
                f"Unknown scheduler component(s): {missing}. "
                f"Available components: {available}.")
        return tuple((name, self.schedulers[name]) for name in normalized)

    def step(self, *, names=None, metric=None) -> None:
        for _, scheduler in self._select(names):
            if metric is None:
                scheduler.step()
            else:
                try:
                    scheduler.step(metric)
                except TypeError:
                    scheduler.step()

    def state_dict(self):
        return {name: scheduler.state_dict() for name, scheduler in self.schedulers.items()}

    def load_state_dict(self, state_dict, *, strict: bool = True):
        expected = set(self.schedulers)
        received = set(state_dict)
        if strict and expected != received:
            missing = ", ".join(sorted(expected - received)) or "none"
            unexpected = ", ".join(sorted(received - expected)) or "none"
            raise ValueError(
                "Scheduler checkpoint topology does not match the current "
                f"recipe (missing: {missing}; unexpected: {unexpected}).")
        for name in expected & received:
            self.schedulers[name].load_state_dict(state_dict[name])
