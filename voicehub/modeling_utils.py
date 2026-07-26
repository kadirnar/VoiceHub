"""Transformers-style pretrained model lifecycle for text-to-speech."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.base_model import BaseTTSModel
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.generation_configuration import TTSGenerationConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.processing_utils import VoiceHubProcessor
from voicehub.trainer_utils import NATIVE_EXPORT_DIR


class PreTrainedTTSModel(BaseTTSModel, ABC):
    """Base lifecycle shared by source-integrated VoiceHub architectures."""

    config_class = VoiceHubConfig
    generation_config_class = TTSGenerationConfig
    processor_class = VoiceHubProcessor
    default_model_name_or_path = ""
    main_input_name = "text"
    base_model_prefix = "tts_model"
    supports_gradient_checkpointing = False

    def __init__(
        self,
        config: VoiceHubConfig,
        *,
        device: str = "auto",
        lazy_load: bool = True,
    ):
        super().__init__(model_path=config.name_or_path, device=device)
        self.config = config
        if not self.config.architectures:
            self.config.architectures = [self.__class__.__name__]
        self.generation_config = self.generation_config_class.from_model_config(config)
        self.model = None
        self.processor = self.processor_class()
        self._loading_for_training = False
        self._pending_model_state_path: Path | None = None
        self._pending_training_recipe_state: dict[str, Any] | None = None
        if not lazy_load:
            self.load()

    @classmethod
    def _coerce_config(
        cls,
        config: VoiceHubConfig | str | None = None,
        *,
        model_path: str | None = None,
        **overrides,
    ) -> VoiceHubConfig:
        """Normalize legacy constructor arguments into a typed
        configuration."""
        if isinstance(config, VoiceHubConfig):
            normalized = config
            if model_path is not None:
                normalized.name_or_path = model_path
            normalized.update(overrides)
            return normalized

        if isinstance(config, str):
            if model_path is not None:
                raise TypeError("Pass a path either as `config` or `model_path`, not both.")
            model_path = config
        return cls.config_class(
            name_or_path=(cls.default_model_name_or_path if model_path is None else model_path),
            **overrides,
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def is_loaded(self) -> bool:
        """Whether checkpoint-backed runtime objects have been created."""
        return self.model is not None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "",
        *,
        config: VoiceHubConfig | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Construct a model from local files or a Hub repository
        identifier."""
        if config is None:
            config_values = dict(config_kwargs or {})
            if pretrained_model_name_or_path:
                try:
                    config = cls.config_class.from_pretrained(
                        pretrained_model_name_or_path,
                        **config_values,
                    )
                except FileNotFoundError:
                    config = cls.config_class(
                        name_or_path=pretrained_model_name_or_path,
                        **config_values,
                    )
            else:
                config = cls.config_class(**config_values)
        elif pretrained_model_name_or_path:
            config.name_or_path = pretrained_model_name_or_path

        source = Path(pretrained_model_name_or_path).expanduser()
        model_state_path = source / "model_state.pt"
        has_voicehub_state = source.is_dir() and model_state_path.is_file()
        if has_voicehub_state:
            saved_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            base_model = saved_config.get("name_or_path")
            if isinstance(base_model, str) and base_model.strip():
                config.name_or_path = base_model
        model = cls(
            config,
            device=device,
            # A Trainer artifact must be attached before the first runtime load
            # so inference-only compilation/quantization is never applied to
            # weights that are about to resume training.
            lazy_load=(True if has_voicehub_state else lazy_load),
            **kwargs,
        )
        if has_voicehub_state:
            model._pending_model_state_path = model_state_path
            if not lazy_load:
                model.load()
        if source.is_dir():
            generation_path = source / "generation_config.json"
            processor_path = source / "processor_config.json"
            if generation_path.is_file():
                model.generation_config = (cls.generation_config_class.from_pretrained(source))
            if processor_path.is_file():
                model.processor = cls.processor_class.from_pretrained(source)
        return model

    def load(self):
        """Load model weights and processors once."""
        requested_training = self._loading_for_training
        if self.model is None:
            self.device = self._resolve_device(self.device)
            restore_training_state = self._pending_model_state_path is not None
            previous_mode = self._loading_for_training
            if restore_training_state:
                self._validate_training_runtime()
                self._loading_for_training = True
            try:
                self._load_pretrained_model()
            finally:
                self._loading_for_training = previous_mode
        if self._pending_model_state_path is not None:
            self._restore_voicehub_model_state(training=requested_training, )
        if not requested_training:
            self._prepare_for_inference()
        return self

    def _restore_voicehub_model_state(self, *, training: bool) -> None:
        """Restore a portable state written by :class:`voicehub.Trainer`."""
        state_path = self._pending_model_state_path
        if state_path is None:
            return
        self._pending_model_state_path = None
        try:
            torch = import_module("torch")
            try:
                state = torch.load(
                    state_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except TypeError:
                state = torch.load(state_path, map_location=self.device)

            if (isinstance(state, Mapping) and "__voicehub_training_adapter__" in state):
                adapter = self.get_training_adapter()
                adapter.setup()
                adapter.load_state_dict(
                    state,
                    strict=True,
                    load_recipe_state=False,
                )
                recipe_state = state.get("recipe_state", {})
                if recipe_state:
                    self._pending_training_recipe_state = {
                        "model_type": state["__voicehub_training_adapter__"],
                        "recipe_id": adapter.recipe_id,
                        "state": recipe_state,
                    }
                if not training:
                    adapter.eval()
                return
            if not hasattr(self.model, "load_state_dict"):
                raise TypeError("The restored VoiceHub runtime does not implement "
                                "load_state_dict().")
            self.model.load_state_dict(state)
            if not training and hasattr(self.model, "eval"):
                self.model.eval()
        except BaseException:
            self._pending_model_state_path = state_path
            raise

    @property
    def is_training_load(self) -> bool:
        """Whether the current load is constructing a training runtime."""
        return self._loading_for_training

    def load_for_training(self):
        """Load a differentiable runtime without inference-only pruning.

        Architectures that compile, quantize, fuse, or delete modules
        during inference can inspect :attr:`is_training_load` in their
        loader and restore any additional state in
        :meth:`_prepare_for_training`.
        """
        self._validate_training_runtime()
        if self.model is None:
            self._loading_for_training = True
            try:
                self.load()
            finally:
                self._loading_for_training = False
        self._prepare_for_training()
        # A specialized adapter can construct its differentiable graph before
        # delegating here (Fish Speech does this for its semantic-only graph).
        # In that case ``model`` is already populated and ``load()`` did not
        # get an opportunity to consume a lazy portable state.
        if self._pending_model_state_path is not None:
            self._restore_voicehub_model_state(training=True)
        return self

    def validate_training_support(self):
        """Validate this exact backend/checkpoint configuration without
        loading.

        The returned profile describes the family contract. Unsupported
        quantized, fused, GGUF, ONNX, or custom-recipe variants raise
        before allocating model weights.
        """
        adapter = self.get_training_adapter()
        adapter.validate_support()
        return adapter.spec

    @property
    def training_default_model_name_or_path(self) -> str:
        """Return the recommended differentiable checkpoint for this family."""
        spec = self.get_training_adapter().spec
        return (spec.training_default_model_name_or_path or self.config.name_or_path)

    def _validate_training_runtime(self) -> None:
        """Optional pre-load validation for inference-only variants."""

    def _prepare_for_training(self) -> None:
        """Optional hook that validates or restores a trainable runtime."""

    def _prepare_for_inference(self) -> None:
        """Optional hook that restores an inference-capable runtime.

        Portable Trainer artifacts are initially loaded through the
        differentiable graph so their component topology cannot be
        pruned or fused before state restoration. Architectures whose
        training graph omits serving-only tokenizers, codecs, caches, or
        processors can rebuild those objects here after the trained
        state has been applied.
        """

    def _set_training_device(self, device: str) -> None:
        """Synchronize wrapper/runtime device metadata after strategy moves."""
        self.device = str(device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve ``auto`` only when loading, keeping package imports
        cheap."""
        if device != "auto":
            return device
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @abstractmethod
    def _load_pretrained_model(self) -> None:
        """Build the local source architecture and load its checkpoint."""
        ...

    @abstractmethod
    def _generate(self, text: str, **kwargs) -> TTSOutput:
        """Backend generation hook called by the uniform public API."""
        ...

    def prepare_inputs_for_generation(
        self,
        text: str,
        **kwargs,
    ) -> dict[str, Any]:
        """Normalize raw inputs through the configured processor."""
        return dict(self.processor(text, **kwargs))

    def forward(self, text: str, **kwargs) -> TTSOutput:
        """Run text-to-speech using the backend implementation hook."""
        model_inputs = self.prepare_inputs_for_generation(text, **kwargs)
        self._validate_model_kwargs(model_inputs)
        return self._generate(**model_inputs)

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        """Reject misspelled generation options with an actionable error."""
        parameters = signature(self._generate).parameters
        if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return
        unknown = sorted(set(model_kwargs) - set(parameters))
        if unknown:
            supported = ", ".join(parameters)
            invalid = ", ".join(unknown)
            raise ValueError(
                f"Unsupported generation option(s): {invalid}. "
                f"{self.__class__.__name__} accepts: {supported}.")

    def generate(
        self,
        text: str,
        *,
        generation_config: TTSGenerationConfig | None = None,
        **kwargs,
    ) -> TTSOutput:
        """Generate speech with one signature shared by every architecture."""
        defaults = self.generation_config.to_dict()
        if generation_config is not None:
            if not isinstance(generation_config, TTSGenerationConfig):
                raise TypeError("`generation_config` must be a TTSGenerationConfig.")
            defaults.update(generation_config.to_dict())
        defaults.update(kwargs)
        return self.forward(text, **defaults)

    def __call__(
        self,
        text: str,
        *,
        generation_config: TTSGenerationConfig | None = None,
        **kwargs,
    ) -> TTSOutput:
        return self.generate(
            text,
            generation_config=generation_config,
            **kwargs,
        )

    @classmethod
    def can_generate(cls) -> bool:
        """Return whether this architecture implements generation."""
        return cls._generate is not PreTrainedTTSModel._generate

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        include_native_export: bool = True,
    ) -> Path:
        """Save VoiceHub metadata and optionally namespaced native
        artifacts."""
        output_directory = Path(save_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(output_directory)
        self.generation_config.save_pretrained(output_directory)
        self.processor.save_pretrained(output_directory)
        if include_native_export:
            self._save_pretrained(output_directory / NATIVE_EXPORT_DIR)
        return output_directory

    def get_training_adapter(self):
        """Return the unloaded model-family adapter paired with this model."""
        from voicehub.training.auto import AutoTrainingAdapter

        return AutoTrainingAdapter.from_model(self)

    def create_training_dataset(self, records, **kwargs):
        """Build this architecture's source-native fine-tuning dataset.

        The returned dataset exposes ``collate_fn`` when its token or
        acoustic layout needs model-specific batching.
        """
        return self.get_training_adapter().create_dataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Map a canonical TTS batch to one native training phase.

        Architectures with tokenizer, codec, or acoustic feature
        preparation can override this hook.  Preprocessed datasets pass
        through unchanged.
        """
        return dict(inputs)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Optionally serialize native artifacts in their own namespace."""
