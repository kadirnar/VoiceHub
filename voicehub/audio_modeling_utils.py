"""Task-neutral pretrained lifecycle for audio-input speech models."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from threading import RLock
from typing import Any

from voicehub.base_model import BaseSpeechModel
from voicehub.configuration_utils import VoiceHubConfig, reject_serialized_secrets
from voicehub.inference_configuration import ASRInferenceConfig, SpeechInferenceConfig, VADInferenceConfig
from voicehub.inference_strategy import InferenceStrategy, get_inference_strategy
from voicehub.modeling_outputs import ASROutput, VADOutput
from voicehub.modeling_utils import PreTrainedSpeechModel
from voicehub.path_utils import normalize_model_source
from voicehub.processing_utils import AudioProcessor
from voicehub.trainer_utils import MODEL_STATE_NAME, NATIVE_EXPORT_DIR


class PreTrainedAudioModel(
        BaseSpeechModel,
        PreTrainedSpeechModel,
        ABC,
):
    """Shared lazy lifecycle for ASR and VAD wrappers.

    The lifecycle intentionally mirrors :class:`PreTrainedTTSModel`:
    model allocation is lazy, inference strategy transitions are
    reversible for training, and portable trainer state is attached
    before first inference. Task subclasses own the actual input/output
    semantics.
    """

    config_class = VoiceHubConfig
    processor_class = AudioProcessor
    inference_config_class: type[SpeechInferenceConfig] = SpeechInferenceConfig
    output_type: type = object
    default_model_name_or_path = ""
    main_input_name = "audio"
    base_model_prefix = "speech_model"
    supports_gradient_checkpointing = False
    passthrough_inference_options: frozenset[str] | None = None

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
        self.inference_config = self.inference_config_class.from_model_config(config)
        self.processor = self.processor_class()
        self.model = None
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
        reject_serialized_secrets(
            overrides,
            owner=f"{cls.__name__} config overrides",
        )
        if isinstance(config, VoiceHubConfig):
            values = config.to_dict()
            configured_model_type = values.get("model_type")
            expected_model_type = cls.config_class.model_type
            if configured_model_type not in {
                    "voicehub",
                    expected_model_type,
            }:
                raise TypeError(
                    f"{cls.__name__} requires {expected_model_type!r} config, "
                    f"not {configured_model_type!r}.")
            if model_path is not None:
                values["name_or_path"] = normalize_model_source(model_path)
            values.update(overrides)
            normalized = cls.config_class.from_dict(values)
            normalized.name_or_path = normalize_model_source(normalized.name_or_path)
            return normalized
        if isinstance(config, (str, Path)):
            if model_path is not None:
                raise TypeError("Pass a path either as `config` or `model_path`, not both.")
            model_path = config
        source = (
            cls.default_model_name_or_path if model_path is None else normalize_model_source(model_path))
        return cls.config_class(name_or_path=source, **overrides)

    @property
    def sample_rate(self) -> int:
        return int(self.config.sample_rate)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def is_training_load(self) -> bool:
        return self._loading_for_training

    @property
    def inference_strategy(self) -> InferenceStrategy:
        return self._inference_strategy

    def set_inference_strategy(
        self,
        strategy: str | InferenceStrategy | None,
    ):
        resolved = get_inference_strategy(strategy)
        with self._lifecycle_lock:
            if self._inference_ready or self._inference_strategy_applied:
                raise RuntimeError(
                    "Cannot replace the inference strategy on an active "
                    "runtime. Call load_for_training() first.")
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
        source_name = normalize_model_source(pretrained_model_name_or_path)
        source = Path(source_name).expanduser()
        is_direct_checkpoint = source.is_file() and source.suffix.lower() != ".json"
        if config is None:
            config_values = dict(config_kwargs or {})
            if is_direct_checkpoint:
                config = cls.config_class(name_or_path=source_name, **config_values)
            elif source_name:
                try:
                    config = cls.config_class.from_pretrained(
                        source_name,
                        **config_values,
                    )
                except FileNotFoundError:
                    config = cls.config_class(
                        name_or_path=source_name,
                        **config_values,
                    )
            else:
                config = cls.config_class(**config_values)
        elif source_name:
            config.name_or_path = source_name

        model_state_path = source / MODEL_STATE_NAME
        has_voicehub_state = source.is_dir() and model_state_path.is_file()
        if has_voicehub_state:
            config_path = source / "config.json"
            if config_path.is_file():
                saved_config = json.loads(config_path.read_text(encoding="utf-8"))
                base_model = saved_config.get("name_or_path")
                if isinstance(base_model, str) and base_model.strip():
                    config.name_or_path = base_model

        model = cls(
            config,
            device=device,
            # Artifact-local processor and inference settings must be restored
            # before a provider allocates its runtime.
            lazy_load=True,
            **kwargs,
        )
        if inference_strategy is not None:
            model.set_inference_strategy(inference_strategy)
        if has_voicehub_state:
            model._pending_model_state_path = model_state_path

        if source.is_dir():
            inference_path = source / cls.inference_config_class.config_name
            processor_path = source / "processor_config.json"
            if inference_path.is_file():
                model.inference_config = cls.inference_config_class.from_pretrained(source)
            if processor_path.is_file():
                model.processor = cls.processor_class.from_pretrained(source)
        if not lazy_load:
            model.load()
        return model

    def _validate_inference_strategy(self) -> None:
        if self._inference_strategy_validated:
            return
        self._inference_strategy.validate(self)
        self._inference_strategy_validated = True

    def load(self):
        """Load model weights and enter inference mode exactly once."""
        with self._lifecycle_lock:
            requested_training = self._loading_for_training
            self._validate_optimization_transition("training" if requested_training else "inference")
            if not requested_training:
                self._validate_inference_strategy()
            if self.model is None:
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
                self._restore_voicehub_model_state(training=requested_training)
            if not requested_training and not self._inference_ready:
                self._training_ready = False
                self._prepare_for_inference()
                if not self._inference_strategy_applied:
                    prepared = self._inference_strategy.prepare(
                        self.model,
                        wrapper=self,
                    )
                    if prepared is None:
                        raise TypeError("InferenceStrategy.prepare() must return the model runtime.")
                    self.model = prepared
                    self._inference_strategy_applied = True
                self._inference_ready = True
        return self

    def load_for_training(self):
        """Load or restore the differentiable runtime."""
        with self._lifecycle_lock:
            self._validate_optimization_transition("training")
            self._validate_training_runtime()
            if self.model is None:
                self._loading_for_training = True
                try:
                    self.load()
                finally:
                    self._loading_for_training = False
            self._inference_ready = False
            if not self._training_ready:
                if self._inference_strategy_applied:
                    restored = self._inference_strategy.restore_for_training(
                        self.model,
                        wrapper=self,
                    )
                    if restored is None:
                        raise TypeError(
                            "InferenceStrategy.restore_for_training() must "
                            "return the trainable runtime.")
                    self.model = restored
                    self._inference_strategy_applied = False
                self._prepare_for_training()
                self._training_ready = True
            if self._pending_model_state_path is not None:
                self._restore_voicehub_model_state(training=True)
        return self

    def _restore_voicehub_model_state(self, *, training: bool) -> None:
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
            if isinstance(state, Mapping) and "__voicehub_training_adapter__" in state:
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
                raise TypeError("The restored runtime does not implement load_state_dict().")
            self.model.load_state_dict(state)
            if not training and hasattr(self.model, "eval"):
                self.model.eval()
        except BaseException:
            self._pending_model_state_path = state_path
            raise

    def prepare_inputs_for_inference(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return dict(self.processor(
            audio,
            sampling_rate=sampling_rate,
            **kwargs,
        ))

    def _validate_inference_inputs(self, model_inputs: dict[str, Any]) -> None:
        """Dependency-free task/backend input validation hook."""

    def _inference_callable(self):
        return self._run_inference

    def _validate_model_kwargs(self, model_kwargs: dict[str, Any]) -> None:
        parameters = signature(self._inference_callable()).parameters
        has_passthrough = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
        if has_passthrough and self.passthrough_inference_options is None:
            return
        supported = {
            name
            for name, parameter in parameters.items()
            if name != "self" and parameter.kind not in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL)
        }
        if self.passthrough_inference_options is not None:
            supported.update(self.passthrough_inference_options)
        unknown = sorted(set(model_kwargs) - supported)
        if unknown:
            raise ValueError(
                "Unsupported inference option(s): "
                f"{', '.join(unknown)}. {self.__class__.__name__} accepts: "
                f"{', '.join(sorted(supported))}.")

    def forward(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: SpeechInferenceConfig | None = None,
        **kwargs,
    ):
        defaults = self.inference_config.to_dict()
        if inference_config is not None:
            if not isinstance(inference_config, self.inference_config_class):
                raise TypeError(
                    "`inference_config` must be an instance of "
                    f"{self.inference_config_class.__name__}.")
            defaults.update(inference_config.to_dict())
        defaults.update(kwargs)
        options = self.inference_config_class.from_dict(defaults).to_dict()
        model_inputs = self.prepare_inputs_for_inference(
            audio,
            sampling_rate=sampling_rate,
            **options,
        )
        self._validate_model_kwargs(model_inputs)
        self._validate_inference_inputs(model_inputs)
        with self._lifecycle_lock:
            self.load()
            output = self._inference_callable()(**model_inputs)
        if not isinstance(output, self.output_type):
            raise TypeError(
                f"{self.__class__.__name__} inference must return "
                f"{self.output_type.__name__}, received {type(output).__name__}.")
        return output

    def __call__(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: SpeechInferenceConfig | None = None,
        **kwargs,
    ):
        return self.forward(
            audio,
            sampling_rate=sampling_rate,
            inference_config=inference_config,
            **kwargs,
        )

    def stream(
        self,
        *,
        sampling_rate: int,
        **inference_kwargs,
    ):
        """Create an isolated streaming session.

        The default session buffers chunks and invokes offline inference on
        ``flush``. Cache-aware backends can override this method without
        changing the public session contract.
        """
        from voicehub.streaming import BufferedSpeechSession

        return BufferedSpeechSession(
            self,
            sampling_rate=sampling_rate,
            inference_kwargs=inference_kwargs,
        )

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        include_native_export: bool = True,
    ) -> Path:
        output_directory = Path(save_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(output_directory)
        self.inference_config.save_pretrained(output_directory)
        self.processor.save_pretrained(output_directory)
        if include_native_export:
            self._save_pretrained(output_directory / NATIVE_EXPORT_DIR)
        return output_directory

    def export_native_pretrained(
        self,
        save_directory: str | Path,
    ) -> Path:
        """Write a flat provider-native artifact for trainer interop."""
        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        self._save_pretrained(destination)
        return destination

    def get_training_adapter(self):
        from voicehub.training.auto import AutoTrainingAdapter

        return AutoTrainingAdapter.from_model(self)

    def validate_training_support(self):
        adapter = self.get_training_adapter()
        adapter.validate_support()
        return adapter.spec

    @property
    def training_default_model_name_or_path(self) -> str:
        spec = self.get_training_adapter().spec
        return spec.training_default_model_name_or_path or self.config.name_or_path

    def create_training_dataset(self, records, **kwargs):
        return self.get_training_adapter().create_dataset(records, **kwargs)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        return dict(inputs)

    def _prepare_for_training(self) -> None:
        train = getattr(self.model, "train", None)
        if callable(train):
            train()

    def _prepare_for_inference(self) -> None:
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()

    def _validate_training_runtime(self) -> None:
        """Validate checkpoint/runtime training compatibility before load."""

    def _save_pretrained(self, save_directory: Path) -> None:
        """Optionally export a backend-native inference artifact."""

    @staticmethod
    def _resolve_device(device: str) -> str:
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
        ...

    @abstractmethod
    def _run_inference(self, audio: Any, **kwargs):
        ...


class PreTrainedASRModel(PreTrainedAudioModel, ABC):
    """Base class for speech-recognition models."""

    inference_config_class = ASRInferenceConfig
    output_type = ASROutput
    base_model_prefix = "asr_model"

    @abstractmethod
    def _transcribe(self, audio: Any, **kwargs) -> ASROutput:
        ...

    def _run_inference(self, audio: Any, **kwargs) -> ASROutput:
        return self._transcribe(audio, **kwargs)

    def _inference_callable(self):
        return self._transcribe

    def transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: ASRInferenceConfig | None = None,
        **kwargs,
    ) -> ASROutput:
        return self.forward(
            audio,
            sampling_rate=sampling_rate,
            inference_config=inference_config,
            **kwargs,
        )

    @classmethod
    def can_transcribe(cls) -> bool:
        return cls._transcribe is not PreTrainedASRModel._transcribe


class PreTrainedVADModel(PreTrainedAudioModel, ABC):
    """Base class for voice-activity-detection models."""

    inference_config_class = VADInferenceConfig
    output_type = VADOutput
    base_model_prefix = "vad_model"

    @abstractmethod
    def _detect(self, audio: Any, **kwargs) -> VADOutput:
        ...

    def _run_inference(self, audio: Any, **kwargs) -> VADOutput:
        return self._detect(audio, **kwargs)

    def _inference_callable(self):
        return self._detect

    def detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        inference_config: VADInferenceConfig | None = None,
        **kwargs,
    ) -> VADOutput:
        return self.forward(
            audio,
            sampling_rate=sampling_rate,
            inference_config=inference_config,
            **kwargs,
        )

    @classmethod
    def can_detect(cls) -> bool:
        return cls._detect is not PreTrainedVADModel._detect
