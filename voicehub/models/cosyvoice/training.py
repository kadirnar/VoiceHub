"""Component-native fine-tuning adapter for CosyVoice."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSTrainingOutput
from voicehub.training.adapters import CompositeTrainingAdapter
from voicehub.training.contracts import TrainingContext

_MODEL_CONFIG_NAMES = (
    "cosyvoice3.yaml",
    "cosyvoice2.yaml",
    "cosyvoice.yaml",
)
_TRAINABLE_COMPONENTS = {
    "llm": "llm",
    "language_model": "llm",
    "flow": "flow",
}
_CHECKPOINT_METADATA_KEYS = frozenset({"epoch", "step"})
_VENDORED_YAML_IMPORTS = {
    "cosyvoice": "voicehub.models.cosyvoice.source.cosyvoice",
    "matcha": "voicehub.models.cosyvoice.source.matcha",
}
_YAML_IMPORT_PATTERN = re.compile(r"(?P<tag>!(?:apply|module|name|new):)"
                                  r"(?P<prefix>cosyvoice|matcha)\.")


@dataclass(frozen=True)
class CosyVoiceTrainingArtifacts:
    """Files and recipe controls used to construct a training component."""

    model_directory: Path
    config_path: Path
    checkpoint_path: Path
    component_name: str
    train_conf: Mapping[str, Any]
    sample_rate: int


class CosyVoiceTrainingBackend:
    """Minimal source-compatible runtime containing one trainable graph.

    The public ``model.<component>`` shape intentionally matches the
    source inference wrapper. It lets the shared training adapter use
    stable component paths without constructing the unselected LLM,
    flow, frontend, or HiFT graphs.
    """

    def __init__(
        self,
        component,
        artifacts: CosyVoiceTrainingArtifacts,
    ) -> None:
        self.artifacts = artifacts
        self.model = SimpleNamespace()
        setattr(self.model, artifacts.component_name, component)

    @property
    def component_name(self) -> str:
        return self.artifacts.component_name

    @property
    def selected_component(self):
        return getattr(self.model, self.component_name)

    @property
    def train_conf(self) -> Mapping[str, Any]:
        return self.artifacts.train_conf

    @property
    def sample_rate(self) -> int:
        return self.artifacts.sample_rate


def _canonical_training_component(training_component: str) -> str:
    selected = str(training_component).strip().lower().replace("-", "_")
    try:
        return _TRAINABLE_COMPONENTS[selected]
    except KeyError as exc:
        choices = ", ".join(sorted(_TRAINABLE_COMPONENTS))
        raise ValueError(
            f"CosyVoice selected-component loading supports: {choices}; "
            f"received {selected!r}.") from exc


def _resolve_model_directory(model_name_or_path: str | Path) -> Path:
    model_directory = Path(model_name_or_path).expanduser()
    if model_directory.is_dir():
        return model_directory.resolve()

    modelscope = import_optional(
        "modelscope",
        model_type="cosyvoice",
        install_extra="cosyvoice",
    )
    resolved = Path(modelscope.snapshot_download(str(model_name_or_path)))
    if not resolved.is_dir():
        raise FileNotFoundError(
            "ModelScope did not return a CosyVoice checkpoint directory for "
            f"{model_name_or_path!r}.")
    return resolved.resolve()


def _find_model_config(model_directory: Path) -> Path:
    for config_name in _MODEL_CONFIG_NAMES:
        candidate = model_directory / config_name
        if candidate.is_file():
            return candidate
    expected = ", ".join(_MODEL_CONFIG_NAMES)
    raise ValueError(f"{model_directory} does not contain a CosyVoice model config ({expected}).")


def _load_vendored_hyperpyyaml(
    loader,
    yaml_stream,
    *args,
    **kwargs,
):
    """Resolve upstream YAML tags against VoiceHub's vendored namespaces."""
    if hasattr(yaml_stream, "read"):
        contents = yaml_stream.read()
        stream_name = getattr(yaml_stream, "name", None)
    else:
        contents = str(yaml_stream)
        stream_name = None
    if isinstance(contents, bytes):
        contents = contents.decode("utf-8")

    def replace_import(match: re.Match[str]) -> str:
        namespace = _VENDORED_YAML_IMPORTS[match.group("prefix")]
        return f"{match.group('tag')}{namespace}."

    rewritten_stream = StringIO(_YAML_IMPORT_PATTERN.sub(replace_import, contents))
    if stream_name is not None:
        rewritten_stream.name = stream_name
    return loader(rewritten_stream, *args, **kwargs)


def _load_component_checkpoint(component, checkpoint_path: Path) -> None:
    if checkpoint_path.suffix == ".safetensors":
        safetensors = import_optional(
            "safetensors.torch",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        state_dict = safetensors.load_file(
            str(checkpoint_path),
            device="cpu",
        )
    else:
        torch = import_optional(
            "torch",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        try:
            state_dict = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"CosyVoice checkpoint {checkpoint_path} must contain a state dictionary.")
    model_state = {name: value for name, value in state_dict.items() if name not in _CHECKPOINT_METADATA_KEYS}
    try:
        incompatible = component.load_state_dict(model_state, strict=True)
    except TypeError:
        incompatible = component.load_state_dict(model_state)
    missing = tuple(getattr(incompatible, "missing_keys", ()))
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"CosyVoice checkpoint {checkpoint_path} is incompatible with "
            f"the selected component (missing={missing}, "
            f"unexpected={unexpected}).")


def load_cosyvoice_training_backend(
    model_name_or_path: str | Path,
    training_component: str,
) -> CosyVoiceTrainingBackend:
    """Build only the source component selected for this fine-tuning job."""
    component_name = _canonical_training_component(training_component)
    model_directory = _resolve_model_directory(model_name_or_path)
    config_path = _find_model_config(model_directory)
    hyperpyyaml = import_optional(
        "hyperpyyaml",
        model_type="cosyvoice",
        install_extra="cosyvoice",
    )

    overrides = {name: None for name in ("llm", "flow", "hift", "hifigan") if name != component_name}
    if config_path.name in {"cosyvoice2.yaml", "cosyvoice3.yaml"}:
        overrides["qwen_pretrain_path"] = str(model_directory / "CosyVoice-BlankEN")
    with config_path.open("r", encoding="utf-8") as config_file:
        configs = _load_vendored_hyperpyyaml(
            hyperpyyaml.load_hyperpyyaml,
            config_file,
            overrides=overrides,
        )
    if not isinstance(configs, Mapping):
        raise TypeError(f"CosyVoice config {config_path} must resolve to a mapping.")

    component = configs.get(component_name)
    if component is None or not hasattr(component, "load_state_dict"):
        raise TypeError(
            f"CosyVoice config {config_path} did not construct the selected "
            f"{component_name!r} component.")
    train_conf = configs.get("train_conf")
    if not isinstance(train_conf, Mapping):
        raise TypeError(
            f"CosyVoice config {config_path} must define a 'train_conf' "
            "mapping for source-native optimization.")

    checkpoint_candidates = (
        model_directory / f"{component_name}.safetensors",
        model_directory / f"{component_name}.pt",
    )
    checkpoint_path = next(
        (path for path in checkpoint_candidates if path.is_file()),
        None,
    )
    if checkpoint_path is None:
        expected = ", ".join(path.name for path in checkpoint_candidates)
        raise FileNotFoundError(
            f"{model_directory} has no checkpoint for the selected "
            f"{component_name!r} component; expected one of: {expected}.")
    _load_component_checkpoint(component, checkpoint_path)

    artifacts = CosyVoiceTrainingArtifacts(
        model_directory=model_directory,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        component_name=component_name,
        train_conf=dict(train_conf),
        sample_rate=int(configs.get("sample_rate", 24_000)),
    )
    return CosyVoiceTrainingBackend(component, artifacts)


class CosyVoiceTrainingAdapter(CompositeTrainingAdapter):
    """Run one upstream CosyVoice component recipe per training job.

    CosyVoice's author trainer deliberately selects exactly one of
    ``llm``, ``flow``, or ``hifigan``. VoiceHub mirrors that topology
    instead of pretending these heterogeneous objectives form one model
    forward.
    """

    supports_custom_recipe = True

    _PHASES: ClassVar[dict[str, str]] = {
        "llm": "language_model",
        "language_model": "language_model",
        "flow": "flow",
        "hifigan_generator": "vocoder_generator",
        "hifigan_discriminator": "vocoder_discriminator",
    }

    @property
    def selected_phase_name(self) -> str:
        selected = (
            str(getattr(self.model.config, "training_component", "llm")).strip().lower().replace("-", "_"))
        try:
            return self._PHASES[selected]
        except KeyError as exc:
            choices = ", ".join(sorted(self._PHASES))
            raise ValueError(
                f"Unknown CosyVoice training_component {selected!r}; choose "
                f"one of: {choices}.") from exc

    def setup(self):
        phase_name = self.selected_phase_name
        if phase_name.startswith("vocoder_"):
            raise ValueError(
                "CosyVoice HiFi-GAN fine-tuning requires the training-only "
                "HiFiGan graph (generator, discriminator, and mel transform). "
                "Released inference directories contain only the bare HiFT "
                "generator. Use the upstream training YAML or attach a custom "
                "training graph adapter.")
        backend = getattr(
            self.model,
            "_cosyvoice_training_backend",
            None,
        )
        if backend is not None and getattr(self.model, "model", None) is not backend:
            self.model._prepare_for_training()
        super().setup()
        return self

    def plan_training_phases(self, step: int):
        del step
        return (self.spec.get_phase(self.selected_phase_name), )

    def recipe_resume_configuration(self):
        configuration = dict(super().recipe_resume_configuration())
        configuration["selected_phase"] = self.selected_phase_name
        backend = getattr(
            self.model,
            "_cosyvoice_training_backend",
            None,
        )
        if backend is not None:
            train_conf = backend.train_conf
            configuration["source_optimization"] = {
                key: self._manifest_value(train_conf[key])
                for key in (
                    "optim",
                    "optim_conf",
                    "scheduler",
                    "scheduler_conf",
                ) if key in train_conf
            }
        return configuration

    def _source_train_conf(self) -> Mapping[str, Any] | None:
        backend = getattr(
            self.model,
            "_cosyvoice_training_backend",
            None,
        )
        if backend is None:
            return None
        train_conf = getattr(backend, "train_conf", None)
        return train_conf if isinstance(train_conf, Mapping) else None

    def create_optimizer(self, name, parameters, training_args):
        del training_args
        if name != self.selected_phase_name:
            raise ValueError(
                f"CosyVoice optimizer {name!r} does not match selected phase "
                f"{self.selected_phase_name!r}.")
        train_conf = self._source_train_conf()
        if train_conf is None or "optim" not in train_conf:
            return None

        optimizer_name = str(train_conf["optim"]).strip().lower()
        optimizer_types = {
            "adam": "Adam",
            "adamw": "AdamW",
        }
        try:
            optimizer_type_name = optimizer_types[optimizer_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported CosyVoice source optimizer {train_conf['optim']!r}.") from exc
        optimizer_conf = train_conf.get("optim_conf", {})
        if not isinstance(optimizer_conf, Mapping):
            raise TypeError("CosyVoice train_conf.optim_conf must be a mapping.")
        torch = import_optional(
            "torch",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        optimizer_type = getattr(torch.optim, optimizer_type_name)
        return optimizer_type(
            [parameter for _, parameter in parameters],
            **dict(optimizer_conf),
        )

    def create_scheduler(
        self,
        name,
        optimizer,
        num_training_steps,
        training_args,
    ):
        del training_args
        if name != self.selected_phase_name:
            raise ValueError(
                f"CosyVoice scheduler {name!r} does not match selected phase "
                f"{self.selected_phase_name!r}.")
        train_conf = self._source_train_conf()
        if train_conf is None or "scheduler" not in train_conf:
            return None

        scheduler_name = str(train_conf["scheduler"]).strip()
        normalized_name = scheduler_name.lower()
        scheduler_conf = train_conf.get("scheduler_conf", {})
        if not isinstance(scheduler_conf, Mapping):
            raise TypeError("CosyVoice train_conf.scheduler_conf must be a mapping.")
        scheduler_conf = dict(scheduler_conf)
        scheduler_module = import_optional(
            "voicehub.models.cosyvoice.source.cosyvoice.utils.scheduler",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        if normalized_name == "warmuplr":
            scheduler_type = scheduler_module.WarmupLR
        elif normalized_name == "noamholdannealing":
            scheduler_type = scheduler_module.NoamHoldAnnealing
            scheduler_conf.setdefault("max_steps", num_training_steps)
        elif normalized_name == "constantlr":
            scheduler_type = scheduler_module.ConstantLR
            scheduler_conf = {}
        else:
            raise ValueError(f"Unsupported CosyVoice source scheduler {scheduler_name!r}.")
        return scheduler_type(optimizer, **scheduler_conf)

    def select_evaluation_phase(self, training_phase=None):
        if training_phase is None:
            training_phase = self.selected_phase_name
        return super().select_evaluation_phase(training_phase)

    def named_parameter_groups(self, training_phase=None):
        selected = (self.selected_phase_name if training_phase is None else training_phase)
        return super().named_parameter_groups(training_phase=selected)

    def execute_training_phase(
        self,
        context: TrainingContext,
    ) -> TTSTrainingOutput:
        self.setup()
        if context.phase.name != self.selected_phase_name:
            raise ValueError(
                f"CosyVoice is configured for {self.selected_phase_name!r}, "
                f"not {context.phase.name!r}.")
        component_name = "llm" if context.phase.name == "language_model" else "flow"
        component = getattr(self.model.model.model, component_name)
        source_batch = context.inputs.get("batch")
        if source_batch is None:
            source_batch = dict(context.inputs)
        elif not isinstance(source_batch, Mapping):
            raise TypeError("CosyVoice 'batch' must be a mapping.")
        try:
            device = next(component.parameters()).device
        except StopIteration as exc:
            raise ValueError(f"CosyVoice {component_name} has no trainable parameters.") from exc
        output = component(dict(source_batch), device)
        if not isinstance(output, Mapping) or "loss" not in output:
            raise TypeError(f"CosyVoice {component_name} must return a mapping containing 'loss'.")
        losses = {name: value for name, value in output.items() if name == "loss" or name.startswith("loss_")}
        return TTSTrainingOutput(
            loss=output["loss"],
            logits=output.get("logits"),
            losses=losses,
            metadata={
                "model_type": self.model_type,
                "training_family": self.spec.family_name,
                "training_support": self.spec.support.value,
                "training_phase": context.phase.name,
                "optimizer_names": context.phase.optimizer_names,
                "source_native_recipe": True,
                "metrics": {
                    name: value
                    for name, value in output.items() if name not in losses and name != "logits"
                },
            },
            training_phase=context.phase.name,
            optimizer_names=context.phase.optimizer_names,
        )

    def save_pretrained(self, save_directory) -> None:
        """Export the selected component as safetensors."""
        self.setup()
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        safetensors = import_optional(
            "safetensors.torch",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        component_name = ("llm" if self.selected_phase_name == "language_model" else "flow")
        component = getattr(self.model.model.model, component_name)
        state = {name: value.detach().cpu().contiguous() for name, value in component.state_dict().items()}
        safetensors.save_file(
            state,
            str(destination / f"{component_name}.safetensors"),
        )


__all__ = [
    "CosyVoiceTrainingAdapter",
    "CosyVoiceTrainingArtifacts",
    "CosyVoiceTrainingBackend",
    "load_cosyvoice_training_backend",
]
