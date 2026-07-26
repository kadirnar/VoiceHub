"""Transformers-style pretrained model lifecycle for text-to-speech."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.base_model import BaseTTSModel
from voicehub.configuration_utils import VoiceHubConfig
from voicehub.generation_configuration import TTSGenerationConfig
from voicehub.modeling_outputs import TTSOutput
from voicehub.processing_utils import VoiceHubProcessor


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

        model = cls(
            config,
            device=device,
            lazy_load=lazy_load,
            **kwargs,
        )
        source = Path(pretrained_model_name_or_path).expanduser()
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
        if self.model is None:
            self.device = self._resolve_device(self.device)
            self._load_pretrained_model()
        return self

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

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Save configuration and backend-specific checkpoint artifacts."""
        output_directory = Path(save_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(output_directory)
        self.generation_config.save_pretrained(output_directory)
        self.processor.save_pretrained(output_directory)
        self._save_pretrained(output_directory)
        return output_directory

    def _save_pretrained(self, save_directory: Path) -> None:
        """Optional backend hook for serializing weights."""
