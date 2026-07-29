"""VoiceHub-native low-rank adaptation for PyTorch linear layers."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class LoRAConfig:
    """Serializable low-rank adapter configuration."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("*.q_proj", "*.v_proj")
    freeze_base: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("LoRA `rank` must be a positive integer.")
        if (isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or
                not math.isfinite(float(self.alpha)) or self.alpha <= 0):
            raise ValueError("LoRA `alpha` must be finite and positive.")
        object.__setattr__(self, "alpha", float(self.alpha))
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)) or
                not 0.0 <= float(self.dropout) < 1.0):
            raise ValueError("LoRA `dropout` must be in [0, 1).")
        object.__setattr__(self, "dropout", float(self.dropout))
        patterns = tuple(self.target_modules)
        if not patterns or any(not isinstance(pattern, str) or not pattern.strip() for pattern in patterns):
            raise ValueError("LoRA `target_modules` must contain non-empty glob patterns.")
        if len(patterns) != len(set(patterns)):
            raise ValueError("LoRA target patterns cannot contain duplicates.")
        object.__setattr__(
            self,
            "target_modules",
            tuple(pattern.strip() for pattern in patterns),
        )
        if not isinstance(self.freeze_base, bool):
            raise TypeError("LoRA `freeze_base` must be a boolean.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("LoRA `seed` must be an integer.")

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Linear layer wrapper with mergeable low-rank residual weights."""

    def __init__(
        self,
        base: nn.Linear,
        config: LoRAConfig,
        *,
        seed_offset: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can wrap only torch.nn.Linear.")
        if config.rank > min(base.in_features, base.out_features):
            raise ValueError(
                f"LoRA rank {config.rank} exceeds the smaller dimension of "
                f"Linear({base.in_features}, {base.out_features}).")
        self.base = base
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.scaling
        self.dropout = nn.Dropout(config.dropout)
        self.lora_a = nn.Parameter(
            torch.empty(
                config.rank,
                base.in_features,
                dtype=base.weight.dtype,
                device=base.weight.device,
            ))
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                config.rank,
                dtype=base.weight.dtype,
                device=base.weight.device,
            ))
        generator_device = (base.weight.device if base.weight.device.type == "cuda" else torch.device("cpu"))
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(config.seed + seed_offset)
        bound = 1.0 / math.sqrt(base.in_features)
        with torch.no_grad():
            self.lora_a.uniform_(-bound, bound, generator=generator)
        self._merged = False
        self._base_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in base.named_parameters()
        }
        if config.freeze_base:
            for parameter in base.parameters():
                parameter.requires_grad_(False)

    @property
    def merged(self) -> bool:
        return self._merged

    def adapter_delta(self) -> Tensor:
        return torch.matmul(self.lora_b, self.lora_a) * self.scaling

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.base(inputs)
        if self._merged:
            return output
        adapter_input = self.dropout(inputs).to(dtype=self.lora_a.dtype)
        residual = torch.nn.functional.linear(adapter_input, self.lora_a)
        residual = torch.nn.functional.linear(residual, self.lora_b)
        return output + residual.to(dtype=output.dtype) * self.scaling

    def merge(self) -> None:
        """Add adapter weights to the base exactly once."""
        if self._merged:
            return
        with torch.no_grad():
            self.base.weight.add_(
                self.adapter_delta().to(
                    device=self.base.weight.device,
                    dtype=self.base.weight.dtype,
                ))
        self._merged = True

    def unmerge(self) -> None:
        """Remove previously merged adapter weights."""
        if not self._merged:
            return
        with torch.no_grad():
            self.base.weight.sub_(
                self.adapter_delta().to(
                    device=self.base.weight.device,
                    dtype=self.base.weight.dtype,
                ))
        self._merged = False

    def restore_base_trainability(self) -> None:
        for name, parameter in self.base.named_parameters():
            parameter.requires_grad_(self._base_requires_grad[name])


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    leaf = name.rpartition(".")[2]
    return any(
        fnmatchcase(name, pattern) or ("*" not in pattern and "?" not in pattern and leaf == pattern)
        for pattern in patterns)


def _parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, attribute = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, attribute


class LoRAInjection:
    """Handle for adapter state, merging, and exact graph restoration."""

    def __init__(
        self,
        model: nn.Module,
        modules: Mapping[str, LoRALinear],
        config: LoRAConfig,
    ) -> None:
        self.model = model
        self.modules = dict(modules)
        self.config = config
        self._restored = False

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.modules))

    def parameters(self) -> Iterator[nn.Parameter]:
        for name in self.module_names:
            module = self.modules[name]
            yield module.lora_a
            yield module.lora_b

    def merge(self) -> None:
        self._ensure_active()
        for module in self.modules.values():
            module.merge()

    def unmerge(self) -> None:
        self._ensure_active()
        for module in self.modules.values():
            module.unmerge()

    def adapter_state_dict(self) -> dict[str, Tensor]:
        self._ensure_active()
        return {
            key: tensor.detach().clone()
            for name in self.module_names
            for key, tensor in (
                (f"{name}.lora_a", self.modules[name].lora_a),
                (f"{name}.lora_b", self.modules[name].lora_b),
            )
        }

    def load_adapter_state_dict(
        self,
        state_dict: Mapping[str, Tensor],
        *,
        strict: bool = True,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        self._ensure_active()
        if any(module.merged for module in self.modules.values()):
            raise RuntimeError("Unmerge LoRA modules before loading adapter state.")
        expected = set(self.adapter_state_dict())
        received = set(state_dict)
        missing = tuple(sorted(expected - received))
        unexpected = tuple(sorted(received - expected))
        if strict and (missing or unexpected):
            raise ValueError(f"LoRA state mismatch: missing={missing!r}, "
                             f"unexpected={unexpected!r}.")
        destinations = {
            key: parameter
            for name, module in self.modules.items()
            for key, parameter in (
                (f"{name}.lora_a", module.lora_a),
                (f"{name}.lora_b", module.lora_b),
            )
        }
        for key in sorted(expected & received):
            source = state_dict[key]
            destination = destinations[key]
            if not isinstance(source, Tensor):
                raise TypeError(f"LoRA state {key!r} must be a tensor.")
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(
                    f"LoRA state {key!r} has shape {tuple(source.shape)!r}; "
                    f"expected {tuple(destination.shape)!r}.")
        with torch.no_grad():
            for key in sorted(expected & received):
                destinations[key].copy_(
                    state_dict[key].to(
                        device=destinations[key].device,
                        dtype=destinations[key].dtype,
                    ))
        return missing, unexpected

    def restore(self) -> nn.Module:
        """Remove wrappers and restore original ``requires_grad`` flags."""
        self._ensure_active()
        self.unmerge()
        # Deeper module names are replaced first if future module types allow
        # nested injection.
        for name in sorted(self.modules, key=lambda value: value.count("."), reverse=True):
            wrapper = self.modules[name]
            wrapper.restore_base_trainability()
            parent, attribute = _parent_module(self.model, name)
            setattr(parent, attribute, wrapper.base)
        self._restored = True
        return self.model

    def _ensure_active(self) -> None:
        if self._restored:
            raise RuntimeError("This LoRA injection has already been restored.")


def inject_lora(model: nn.Module, config: LoRAConfig) -> LoRAInjection:
    """Atomically replace selected linear layers with native LoRA wrappers."""
    if not isinstance(model, nn.Module):
        raise TypeError("LoRA injection requires a torch.nn.Module.")
    if not isinstance(config, LoRAConfig):
        raise TypeError("`config` must be a LoRAConfig.")
    selected = [(name, module) for name, module in model.named_modules()
                if name and isinstance(module, nn.Linear) and _matches(name, config.target_modules)]
    if not selected:
        available = tuple(name for name, module in model.named_modules() if isinstance(module, nn.Linear))
        raise ValueError(
            f"LoRA target patterns {config.target_modules!r} matched no "
            f"linear modules. Available: {available!r}.")
    for name, module in selected:
        if config.rank > min(module.in_features, module.out_features):
            raise ValueError(
                f"LoRA rank {config.rank} is incompatible with target "
                f"{name!r} shaped ({module.out_features}, "
                f"{module.in_features}).")

    wrappers: dict[str, LoRALinear] = {}
    try:
        for index, (name, module) in enumerate(selected):
            wrapper = LoRALinear(
                module,
                config,
                seed_offset=index,
            )
            parent, attribute = _parent_module(model, name)
            setattr(parent, attribute, wrapper)
            wrappers[name] = wrapper
    except BaseException:
        for name, wrapper in reversed(tuple(wrappers.items())):
            wrapper.restore_base_trainability()
            parent, attribute = _parent_module(model, name)
            setattr(parent, attribute, wrapper.base)
        raise
    return LoRAInjection(model, wrappers, config)


__all__ = [
    "LoRAConfig",
    "LoRAInjection",
    "LoRALinear",
    "inject_lora",
]
