"""Transformers-style pretrained model lifecycle for speech models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.base_model import BaseTTSModel
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.generation_configuration import TTSGenerationConfig
from voicehub.inference_strategy import InferenceStrategy, get_inference_strategy
from voicehub.modeling_outputs import TTSOutput
from voicehub.path_utils import normalize_model_source
from voicehub.processing_utils import VoiceHubProcessor
from voicehub.trainer_utils import NATIVE_EXPORT_DIR


class PreTrainedSpeechModel(ABC):
    """Marker base shared by task-specific pretrained speech wrappers."""


class PreTrainedTTSModel(BaseTTSModel, PreTrainedSpeechModel, ABC):
    """Base lifecycle shared by source-integrated VoiceHub architectures."""

    config_class = VoiceHubConfig
    generation_config_class = TTSGenerationConfig
    processor_class = VoiceHubProcessor
    default_model_name_or_path = ""
    main_input_name = "text"
    base_model_prefix = "tts_model"
    supports_gradient_checkpointing = False
    passthrough_generation_options: frozenset[str] | None = None

    def __init__(
        self,
        config: VoiceHubConfig,
        *,
        device: str = "auto",
        lazy_load: bool = True,
    ):
        config.name_or_path = normalize_model_source(config.name_or_path)
        super().__init__(model_path=config.name_or_path, device=device)
        self.config = config
        if not self.config.architectures:
            self.config.architectures = [self.__class__.__name__]
        self.generation_config = self.generation_config_class.from_model_config(config)
        self.model = None
        self.processor = self.processor_class()
        # Most TTS runtimes own mutable KV caches, vocoders, and temporary
        # conditioning state. Serialize lifecycle transitions and synthesis
        # on one wrapper so concurrent first use cannot load twice or mutate a
        # serving runtime while another request is using it.
        self._lifecycle_lock = RLock()
        self._inference_strategy = get_inference_strategy()
        self._inference_strategy_validated = False
        self._inference_strategy_applied = False
        self._inference_ready = False
        self._training_ready = False
        self._loading_for_training = False
        self._pending_model_state_path: Path | None = None
        self._pending_training_recipe_state: dict[str, Any] | None = None
        if not lazy_load:
            self.load()

    @classmethod
    def _coerce_config(
        cls,
        config: VoiceHubConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        **overrides,
    ) -> VoiceHubConfig:
        """Normalize legacy constructor arguments into a typed
        configuration."""
        if isinstance(config, VoiceHubConfig):
            normalized = config
            if model_path is not None:
                normalized.name_or_path = normalize_model_source(model_path)
            else:
                normalized.name_or_path = normalize_model_source(normalized.name_or_path)
            normalized.update(overrides)
            normalized.name_or_path = normalize_model_source(normalized.name_or_path)
            return normalized

        if isinstance(config, (str, Path)):
            if model_path is not None:
                raise TypeError("Pass a path either as `config` or `model_path`, not both.")
            model_path = normalize_model_source(config)
        if model_path is not None:
            model_path = normalize_model_source(model_path)
        return cls.config_class(
            name_or_path=(cls.default_model_name_or_path if model_path is None else str(model_path)),
            **overrides,
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def is_loaded(self) -> bool:
        """Whether checkpoint-backed runtime objects have been created."""
        return self.model is not None

    @property
    def inference_strategy(self) -> InferenceStrategy:
        """Return the runtime policy used for inference preparation."""
        return self._inference_strategy

    def set_inference_strategy(
        self,
        strategy: str | InferenceStrategy | None,
    ):
        """Select an inference runtime policy before serving begins.

        A model loaded only for training may still select a strategy; it
        is applied lazily on the next transition to inference. An active
        serving runtime must first transition through
        :meth:`load_for_training` so its current strategy can undo
        inference-only transformations safely.
        """
        resolved = get_inference_strategy(strategy)
        with self._lifecycle_lock:
            if self._inference_ready or self._inference_strategy_applied:
                raise RuntimeError(
                    "Cannot replace the inference strategy on an active "
                    "serving runtime. Call load_for_training() first.")
            self._inference_strategy = resolved
            self._inference_strategy_validated = False
        return self

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        config: VoiceHubConfig | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        inference_strategy: str | InferenceStrategy | None = None,
        config_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Construct a model from local files or a Hub repository
        identifier."""
        pretrained_model_name_or_path = normalize_model_source(pretrained_model_name_or_path)
        source = Path(pretrained_model_name_or_path).expanduser()
        is_direct_checkpoint_file = (source.is_file() and source.suffix.lower() != ".json")
        if config is None:
            config_values = dict(config_kwargs or {})
            if is_direct_checkpoint_file:
                config = cls.config_class(
                    name_or_path=pretrained_model_name_or_path,
                    **config_values,
                )
            elif pretrained_model_name_or_path:
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

        model_state_path = source / "model_state.pt"
        has_voicehub_state = source.is_dir() and model_state_path.is_file()
        if has_voicehub_state:
            saved_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            base_model = saved_config.get("name_or_path")
            if isinstance(base_model, str) and base_model.strip():
                config.name_or_path = base_model
        defer_initial_load = has_voicehub_state or inference_strategy is not None
        model = cls(
            config,
            device=device,
            # A Trainer artifact must be attached before the first runtime load
            # and an inference strategy must be selected before the runtime is
            # prepared. Both require deferring an eager constructor load.
            lazy_load=(True if defer_initial_load else lazy_load),
            **kwargs,
        )
        if inference_strategy is not None:
            model.set_inference_strategy(inference_strategy)
        if has_voicehub_state:
            model._pending_model_state_path = model_state_path
        if not lazy_load and defer_initial_load:
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
        """Load weights and prepare the runtime for inference exactly once.

        Loading and mode transitions are guarded because source TTS
        runtimes commonly allocate mutable caches. Failed loads remain
        retryable.
        """
        with self._lifecycle_lock:
            requested_training = self._loading_for_training
            if not requested_training:
                self._validate_inference_strategy()
            if self.model is None:
                self._inference_ready = False
                self._training_ready = False
                self.device = self._resolve_device(self.device)
                restore_training_state = self._pending_model_state_path is not None
                previous_mode = self._loading_for_training
                if restore_training_state:
                    self._validate_training_runtime()
                    self._loading_for_training = True
                try:
                    from voicehub.models._shared import preserve_inference_state

                    with preserve_inference_state(
                            device=self.device,
                            model_type=self.config.model_type,
                    ):
                        self._load_pretrained_model()
                except BaseException:
                    self.model = None
                    self._inference_ready = False
                    self._training_ready = False
                    raise
                finally:
                    self._loading_for_training = previous_mode
                if self.model is None:
                    raise RuntimeError(
                        f"{self.__class__.__name__}._load_pretrained_model() "
                        "completed without assigning `self.model`.")
            if self._pending_model_state_path is not None:
                self._restore_voicehub_model_state(training=requested_training, )
            if not requested_training and not self._inference_ready:
                self._training_ready = False
                self._prepare_for_inference()
                if not self._inference_strategy_applied:
                    prepared_model = self._inference_strategy.prepare(
                        self.model,
                        wrapper=self,
                    )
                    if prepared_model is None:
                        raise TypeError(
                            "InferenceStrategy.prepare() must return the "
                            "prepared model runtime.")
                    self.model = prepared_model
                    self._inference_strategy_applied = True
                self._inference_ready = True
        return self

    def _validate_inference_strategy(self) -> None:
        """Validate the selected runtime policy once, before allocation."""
        if self._inference_strategy_validated:
            return
        self._inference_strategy.validate(self)
        self._inference_strategy_validated = True

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
        with self._lifecycle_lock:
            self._validate_training_runtime()
            if self.model is None:
                self._loading_for_training = True
                try:
                    self.load()
                finally:
                    self._loading_for_training = False
            # Invalidate the serving state before the transition starts. A
            # backend hook may clear caches or detach serving auxiliaries
            # before it discovers an error; the next inference call must then
            # rebuild those resources instead of trusting a partially changed
            # runtime.
            self._inference_ready = False
            if not self._training_ready:
                if self._inference_strategy_applied:
                    restored_model = self._inference_strategy.restore_for_training(
                        self.model,
                        wrapper=self,
                    )
                    if restored_model is None:
                        raise TypeError(
                            "InferenceStrategy.restore_for_training() must "
                            "return the trainable model runtime.")
                    self.model = restored_model
                    self._inference_strategy_applied = False
                self._prepare_for_training()
                self._training_ready = True
            # A specialized adapter can construct its differentiable graph
            # before delegating here (Fish Speech does this for its
            # semantic-only graph). In that case ``model`` is already
            # populated and ``load()`` did not get an opportunity to consume
            # a lazy portable state.
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
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()

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

    def _validate_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        """Validate one request before allocating the model runtime.

        Backends should override this hook for dependency-free checks
        such as required reference audio, mutually exclusive options,
        and supported modes. Tensor- or model-dependent validation
        remains in :meth:`_generate`.
        """

    def _validate_common_generation_inputs(
        self,
        model_inputs: dict[str, Any],
    ) -> None:
        """Validate common options even when callers invoke ``forward``."""
        common_options = {
            name: model_inputs[name]
            for name in self.generation_config_class._COMMON_FIELDS if name in model_inputs
        }
        self.generation_config_class(**common_options)

    def forward(self, text: str, **kwargs) -> TTSOutput:
        """Run text-to-speech using the backend implementation hook."""
        model_inputs = self.prepare_inputs_for_generation(text, **kwargs)
        self._validate_model_kwargs(model_inputs)
        self._validate_common_generation_inputs(model_inputs)
        self._validate_generation_inputs(model_inputs)
        with self._lifecycle_lock:
            self.load()
            output = self._generate(**model_inputs)
        if not isinstance(output, TTSOutput):
            raise TypeError(
                f"{self.__class__.__name__}._generate() must return a "
                f"TTSOutput, received {type(output).__name__}.")
        return output

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        """Reject misspelled generation options with an actionable error."""
        parameters = signature(self._generate).parameters
        has_passthrough = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
        if has_passthrough and self.passthrough_generation_options is None:
            return
        supported_options = {
            name
            for name, parameter in parameters.items() if name != "self" and parameter.kind not in {
                Parameter.VAR_KEYWORD,
                Parameter.VAR_POSITIONAL,
            }
        }
        if self.passthrough_generation_options is not None:
            supported_options.update(self.passthrough_generation_options)
        unknown = sorted(set(model_kwargs) - supported_options)
        if unknown:
            supported = ", ".join(sorted(supported_options))
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
        generation_options = self.generation_config_class.from_dict(defaults)
        return self.forward(text, **generation_options.to_dict())

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
