"""Framework-light exponential moving averages for TTS training recipes."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any


class ExponentialMovingAverage:
    """Track a module's trainable parameters after successful updates.

    The implementation intentionally relies only on the tensor protocol
    used by PyTorch objects and does not import PyTorch at module import
    time.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        module,
        *,
        decay: float = 0.999,
        update_after_step: int = 0,
        update_every: int = 1,
    ):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        if isinstance(update_after_step, bool) or int(update_after_step) < 0:
            raise ValueError("EMA update_after_step must be a non-negative integer.")
        if isinstance(update_every, bool) or int(update_every) <= 0:
            raise ValueError("EMA update_every must be a positive integer.")
        if not hasattr(module, "named_parameters"):
            raise TypeError("EMA requires a module with named_parameters().")

        self.module = module
        self.decay = float(decay)
        self.update_after_step = int(update_after_step)
        self.update_every = int(update_every)
        self.num_updates = 0
        self.last_step = 0
        self._shadow: dict[str, Any] = {}

    @staticmethod
    def _named_parameters(module):
        try:
            parameters = module.named_parameters(remove_duplicate=True)
        except TypeError:
            parameters = module.named_parameters()
        return tuple(
            (name, parameter) for name, parameter in parameters if getattr(parameter, "requires_grad", False))

    def _initialize(self) -> None:
        if self._shadow:
            return
        self._shadow = {
            name: parameter.detach().clone()
            for name, parameter in self._named_parameters(self.module)
        }
        if not self._shadow:
            raise ValueError("EMA cannot track a module with no trainable parameters.")

    def update(self, *, step: int) -> bool:
        """Update the shadow after a successful optimizer step."""
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("EMA updates require a positive integer step.")
        self.last_step = step
        if step <= self.update_after_step:
            return False
        if (step - self.update_after_step - 1) % self.update_every:
            return False

        self._initialize()
        current = dict(self._named_parameters(self.module))
        if set(current) != set(self._shadow):
            raise RuntimeError("EMA parameter topology changed after initialization.")
        for name, parameter in current.items():
            shadow = self._shadow[name]
            value = parameter.detach().to(
                device=shadow.device,
                dtype=shadow.dtype,
            )
            shadow.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
        self.num_updates += 1
        return True

    def copy_to(self, module=None) -> None:
        """Copy averaged parameters into ``module`` or the tracked module."""
        self._initialize()
        target = module or self.module
        parameters = dict(self._named_parameters(target))
        if set(parameters) != set(self._shadow):
            raise ValueError("EMA target parameter topology does not match the shadow.")
        for name, parameter in parameters.items():
            parameter.data.copy_(self._shadow[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            ))

    @contextmanager
    def average_parameters(self, module=None):
        """Temporarily evaluate a module with averaged parameters."""
        target = module or self.module
        original = {name: parameter.detach().clone() for name, parameter in self._named_parameters(target)}
        self.copy_to(target)
        try:
            yield target
        finally:
            parameters = dict(self._named_parameters(target))
            for name, value in original.items():
                parameters[name].data.copy_(value)

    def state_dict(self) -> dict[str, Any]:
        self._initialize()
        return {
            "version": self.STATE_VERSION,
            "decay": self.decay,
            "update_after_step": self.update_after_step,
            "update_every": self.update_every,
            "num_updates": self.num_updates,
            "last_step": self.last_step,
            "shadow": {
                name: value.detach().clone()
                for name, value in self._shadow.items()
            },
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        if not isinstance(state_dict, Mapping):
            raise TypeError("EMA state must be a mapping.")
        if state_dict.get("version") != self.STATE_VERSION:
            raise ValueError(f"Unsupported EMA state version {state_dict.get('version')!r}.")
        shadow = state_dict.get("shadow")
        if not isinstance(shadow, Mapping):
            raise TypeError("EMA state must contain a shadow mapping.")

        current_names = {name for name, _ in self._named_parameters(self.module)}
        received_names = set(shadow)
        if strict and received_names != current_names:
            missing = sorted(current_names - received_names)
            unexpected = sorted(received_names - current_names)
            raise ValueError(
                "EMA parameter topology mismatch "
                f"(missing={missing}, unexpected={unexpected}).")

        self.decay = float(state_dict["decay"])
        self.update_after_step = int(state_dict["update_after_step"])
        self.update_every = int(state_dict["update_every"])
        self.num_updates = int(state_dict.get("num_updates", 0))
        self.last_step = int(state_dict.get("last_step", 0))
        self._shadow = {
            name: value.detach().clone()
            for name, value in shadow.items() if name in current_names
        }
