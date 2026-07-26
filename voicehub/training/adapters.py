"""Trainable-module discovery and objective adapters for TTS model families."""

from __future__ import annotations

import inspect
from collections import deque
from collections.abc import Mapping
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.collators import DataCollatorForTTSTraining
from voicehub.training.specs import ModelTrainingSpec


class BaseTrainingAdapter:
    """Expose an inference wrapper's source modules through a training API."""

    def __init__(self, model, spec: ModelTrainingSpec):
        self.model = model
        self.spec = spec
        self.primary_model = None
        self._components: list[tuple[str, Any]] = []
        self.data_collator = DataCollatorForTTSTraining()

    @property
    def model_type(self) -> str:
        return self.spec.model_type

    @property
    def is_ready(self) -> bool:
        return self.primary_model is not None

    def setup(self):
        """Load the wrapper and resolve all trainable source components."""
        if self.is_ready:
            return self
        if hasattr(self.model, "load"):
            self.model.load()

        components = []
        for path in self.spec.component_paths:
            candidate = self._resolve_path(path)
            if self._is_trainable(candidate):
                components.append((path, candidate))

        for path in self.spec.module_paths:
            candidate = self._resolve_path(path)
            if self._is_trainable(candidate):
                self.primary_model = candidate
                components.insert(0, (path, candidate))
                break

        if self.primary_model is None:
            discovered = self._discover_trainable_modules(self.model)
            if discovered:
                self.primary_model = discovered[0][1]
                components.insert(0, discovered[0])

        if self.primary_model is None:
            checked = ", ".join(self.spec.module_paths)
            raise TypeError(
                f"{self.model_type!r} loaded successfully but no trainable "
                f"PyTorch module was found. Checked: {checked}.")

        self._components = self._deduplicate_components(components)
        return self

    def _resolve_path(self, path: str):
        current = self.model
        for part in path.split("."):
            if isinstance(current, Mapping):
                if part not in current:
                    return None
                current = current[part]
            else:
                current = getattr(current, part, None)
            if current is None:
                return None
        return current

    @staticmethod
    def _is_trainable(candidate) -> bool:
        if candidate is None or not hasattr(candidate, "parameters"):
            return False
        if not callable(candidate) and not callable(getattr(candidate, "forward", None)):
            return False
        try:
            return any(getattr(parameter, "requires_grad", False) for parameter in candidate.parameters())
        except (AttributeError, TypeError):
            return False

    @classmethod
    def _discover_trainable_modules(
        cls,
        root,
        *,
        max_depth: int = 4,
        max_nodes: int = 512,
    ) -> list[tuple[str, Any]]:
        queue = deque([("model", root, 0)])
        visited = set()
        discovered = []
        while queue and len(visited) < max_nodes:
            path, value, depth = queue.popleft()
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            if cls._is_trainable(value):
                discovered.append((path, value))
                continue
            if depth >= max_depth:
                continue
            for name, child in cls._iter_children(value):
                queue.append((f"{path}.{name}", child, depth + 1))
        discovered.sort(
            key=lambda item: cls._parameter_count(item[1]),
            reverse=True,
        )
        return discovered

    @staticmethod
    def _iter_children(value):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key), child
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield str(index), child
            return
        values = getattr(value, "__dict__", None)
        if not isinstance(values, dict):
            return
        for name, child in values.items():
            if name.startswith("__"):
                continue
            if isinstance(child, (str, bytes, int, float, bool, type(None))):
                continue
            yield name, child

    @staticmethod
    def _parameter_count(module) -> int:
        try:
            return sum(
                parameter.numel() for parameter in module.parameters()
                if getattr(parameter, "requires_grad", False))
        except (AttributeError, TypeError):
            return 0

    @staticmethod
    def _deduplicate_components(components):
        output = []
        seen = set()
        for name, component in components:
            if id(component) in seen:
                continue
            seen.add(id(component))
            output.append((name, component))
        return output

    def named_parameters(self):
        """Yield every trainable parameter exactly once across components."""
        self.setup()
        seen = set()
        for component_name, component in self._components:
            safe_name = component_name.replace(".", "_")
            for name, parameter in component.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                yield f"{safe_name}.{name}", parameter

    def parameters(self):
        for _, parameter in self.named_parameters():
            yield parameter

    def named_parameter_groups(self):
        """Partition composite components without assigning a parameter
        twice."""
        self.setup()
        seen = set()
        groups = []
        for component_name, component in reversed(self._components):
            parameters = []
            for name, parameter in component.named_parameters():
                if (id(parameter) in seen or not getattr(parameter, "requires_grad", False)):
                    continue
                seen.add(id(parameter))
                parameters.append((name, parameter))
            if parameters:
                groups.append((
                    component_name.replace(".", "_"),
                    parameters,
                ))
        groups.reverse()
        return groups

    def to(self, device):
        self.setup()
        for _, component in self._components:
            if hasattr(component, "to"):
                component.to(device)
        return self

    def train(self):
        self.setup()
        for _, component in self._components:
            component.train()
        return self

    def eval(self):
        self.setup()
        for _, component in self._components:
            component.eval()
        return self

    def state_dict(self):
        """Serialize all resolved components without duplicate parameters."""
        self.setup()
        return {
            "__voicehub_training_adapter__": self.model_type,
            "components": {
                name: component.state_dict()
                for name, component in self._components
            },
        }

    def load_state_dict(self, state_dict):
        self.setup()
        if "components" not in state_dict:
            return self.primary_model.load_state_dict(state_dict)
        available = dict(self._components)
        results = {}
        for name, component_state in state_dict["components"].items():
            if name in available:
                results[name] = available[name].load_state_dict(component_state)
        return results

    def __call__(self, **inputs) -> TTSTrainingOutput:
        self.setup()
        forward_inputs = dict(inputs.pop("model_inputs", {}) or {})
        forward_inputs.update(inputs)
        labels = self._find_labels(forward_inputs)
        self._map_labels_to_signature(
            self.primary_model,
            forward_inputs,
            labels,
        )
        prepared = self._filter_forward_inputs(
            self.primary_model,
            forward_inputs,
        )
        outputs = self.primary_model(**prepared)
        losses = self._extract_losses(outputs)
        predictions = self._extract_predictions(outputs)
        loss = self._aggregate_losses(losses)
        if loss is None:
            loss = self.compute_objective(predictions, labels)
            losses = {"loss": loss}
        return TTSTrainingOutput(
            loss=loss,
            logits=predictions,
            audio_values=self._get_value(outputs, "audio_values"),
            losses=losses,
            metadata={
                "model_type": self.model_type,
                "training_family": self.spec.family.value,
            },
        )

    def _find_labels(self, inputs):
        for name in self.spec.label_names:
            if name in inputs:
                return inputs[name]
        return None

    def _map_labels_to_signature(self, module, inputs, labels) -> None:
        if labels is None:
            return
        forward = module.forward if hasattr(module, "forward") else module
        parameters = inspect.signature(forward).parameters
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return
        if any(name in inputs and name in parameters for name in self.spec.label_names):
            return
        for name in self.spec.label_names:
            if name in parameters:
                inputs[name] = labels
                return

    @staticmethod
    def _filter_forward_inputs(module, inputs):
        forward = module.forward if hasattr(module, "forward") else module
        parameters = inspect.signature(forward).parameters
        if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return inputs
        accepted = set(parameters)
        return {key: value for key, value in inputs.items() if key in accepted}

    @staticmethod
    def _get_value(outputs, key):
        if isinstance(outputs, Mapping):
            return outputs.get(key)
        return getattr(outputs, key, None)

    def _extract_predictions(self, outputs):
        for key in self.spec.prediction_keys:
            value = self._get_value(outputs, key)
            if value is not None:
                return value
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                if hasattr(value, "shape") and getattr(value, "ndim", 0) > 0:
                    return value
        if hasattr(outputs, "shape") and getattr(outputs, "ndim", 0) > 0:
            return outputs
        return None

    def _extract_losses(self, outputs) -> dict[str, Any]:
        losses = {}
        nested = self._get_value(outputs, "losses")
        if nested is None:
            nested = self._get_value(outputs, "loss_dict")
        if isinstance(nested, Mapping):
            losses.update({str(key): value for key, value in nested.items() if self._is_scalar(value)})

        for key in self.spec.loss_keys:
            value = self._get_value(outputs, key)
            if self._is_scalar(value):
                losses.setdefault(key, value)
        if isinstance(outputs, Mapping):
            for key, value in outputs.items():
                if ((key.endswith("_loss") or key.endswith("loss")) and self._is_scalar(value)):
                    losses.setdefault(str(key), value)
        if isinstance(outputs, (tuple, list)) and outputs:
            if self._is_scalar(outputs[0]):
                losses.setdefault("loss", outputs[0])
        if self._is_scalar(outputs):
            losses.setdefault("loss", outputs)
        return losses

    @staticmethod
    def _is_scalar(value) -> bool:
        return value is not None and (
            isinstance(value, (int, float)) or hasattr(value, "ndim") and value.ndim == 0)

    def _aggregate_losses(self, losses):
        if not losses:
            return None
        if self.spec.loss_weights:
            weighted = [losses[name] * weight for name, weight in self.spec.loss_weights if name in losses]
            if weighted:
                return sum(weighted)
        if "loss" in losses:
            return losses["loss"]
        return sum(losses.values())

    def compute_objective(self, predictions, labels):
        """Compute a family-specific fallback when source output has no
        loss."""
        raise NotImplementedError

    @staticmethod
    def _require_predictions_and_labels(predictions, labels):
        if predictions is None:
            raise ValueError(
                "The source model returned neither a loss nor predictions. "
                "Return a loss from forward() or override the training adapter.")
        if labels is None:
            raise ValueError(
                "The batch has no target field. Provide one of the profile's "
                "label_names or return a native loss from the source model.")

    @staticmethod
    def _cross_entropy(predictions, labels, *, shift: bool):
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        if predictions.ndim < 2:
            raise ValueError("Cross-entropy training requires rank-2+ logits.")
        logits = predictions
        targets = labels
        if logits.ndim >= 3 and targets.ndim >= 2:
            common_length = min(logits.shape[-2], targets.shape[-1])
            if shift:
                if common_length < 2:
                    raise ValueError("Causal token training requires at least two aligned tokens.")
                logits = logits[..., :common_length - 1, :].contiguous()
                targets = targets[..., 1:common_length].contiguous()
            else:
                logits = logits[..., :common_length, :].contiguous()
                targets = targets[..., :common_length].contiguous()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1).long(),
            ignore_index=-100,
        )


class CausalLMTrainingAdapter(BaseTrainingAdapter):
    """Autoregressive codec-token objective with shifted cross entropy."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        return self._cross_entropy(predictions, labels, shift=True)


class Seq2SeqTrainingAdapter(BaseTrainingAdapter):
    """Teacher-forced sequence objective without causal label shifting."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        return self._cross_entropy(predictions, labels, shift=False)


class FlowMatchingTrainingAdapter(BaseTrainingAdapter):
    """Continuous flow/diffusion objective with source-loss preference."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        return torch.nn.functional.mse_loss(predictions, labels)


class AcousticTrainingAdapter(BaseTrainingAdapter):
    """Mel, codec, or waveform reconstruction objective."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        if self.spec.regression_loss == "l1":
            return torch.nn.functional.l1_loss(predictions, labels)
        return torch.nn.functional.mse_loss(predictions, labels)


class CompositeTrainingAdapter(BaseTrainingAdapter):
    """Multi-component objective that sums native named losses safely."""

    def compute_objective(self, predictions, labels):
        self._require_predictions_and_labels(predictions, labels)
        if (predictions.ndim >= 2 and not getattr(labels, "is_floating_point", lambda: True)()):
            return self._cross_entropy(predictions, labels, shift=False)
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        return torch.nn.functional.mse_loss(predictions, labels)
