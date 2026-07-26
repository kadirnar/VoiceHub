"""Automatic configuration and text-to-speech model factories."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.inference_strategy import InferenceStrategy
from voicehub.processing_utils import VoiceHubProcessor
from voicehub.registry import get_model_spec


def _load_class(module_name: str, class_name: str):
    module = import_module(module_name)
    return getattr(module, class_name)


class AutoConfig:
    """Instantiate the registered configuration class for a model type."""

    def __init__(self):
        raise OSError("AutoConfig must be created with for_model/from_pretrained.")

    @classmethod
    def for_model(cls, model_type: str, **kwargs) -> VoiceHubConfig:
        spec = get_model_spec(model_type)
        config_class = _load_class(spec.config_module, spec.config_class)
        kwargs.setdefault("architectures", [spec.class_name])
        config = config_class(**kwargs)
        if config.model_type == "voicehub":
            config.model_type = spec.model_type
        return config

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        model_type: str | None = None,
        **kwargs,
    ) -> VoiceHubConfig:
        source = Path(pretrained_model_name_or_path).expanduser()
        is_direct_checkpoint = (source.is_file() and source.suffix.lower() != ".json")
        if model_type is None:
            if is_direct_checkpoint:
                raise ValueError(
                    "A raw checkpoint file does not identify its model type. "
                    "Pass `model_type` explicitly.")
            if source.is_file():
                config_path = source
            else:
                config_path = resolve_pretrained_file(
                    pretrained_model_name_or_path,
                    "config.json",
                    cache_dir=kwargs.get("cache_dir"),
                    revision=kwargs.get("revision"),
                    token=kwargs.get("token"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
            model_type = read_json_file(config_path).get("model_type")
            if not model_type:
                raise ValueError("config.json does not contain `model_type`; pass model_type explicitly.")

        spec = get_model_spec(model_type)
        config_class = _load_class(spec.config_module, spec.config_class)
        config = config_class.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        if not config.architectures:
            config.architectures = [spec.class_name]
        return config


class AutoModelForTextToSpeech:
    """Load a registered source-integrated model from its configuration."""

    def __init__(self):
        raise OSError("AutoModelForTextToSpeech must be created with from_config/from_pretrained.")

    @classmethod
    def from_config(
        cls,
        config: VoiceHubConfig,
        *,
        inference_strategy: str | InferenceStrategy | None = None,
        **kwargs,
    ):
        spec = get_model_spec(config.model_type)
        model_class = _load_class(spec.module, spec.class_name)
        eager_load = kwargs.get("lazy_load", True) is False
        if inference_strategy is not None and eager_load:
            kwargs["lazy_load"] = True
        model = model_class(config, **kwargs)
        if inference_strategy is not None:
            model.set_inference_strategy(inference_strategy)
            if eager_load:
                model.load()
        return model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        model_type: str | None = None,
        config: VoiceHubConfig | None = None,
        inference_strategy: str | InferenceStrategy | None = None,
        **kwargs,
    ):
        if config is None:
            if model_type is None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            else:
                spec = get_model_spec(model_type)
                config_class = _load_class(spec.config_module, spec.config_class)
                config = config_class(name_or_path=pretrained_model_name_or_path)
                config.model_type = spec.model_type
        spec = get_model_spec(config.model_type if model_type is None else model_type)
        model_class = _load_class(spec.module, spec.class_name)
        return model_class.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            inference_strategy=inference_strategy,
            **kwargs,
        )


class AutoProcessor:
    """Create the processor paired with a VoiceHub TTS configuration."""

    def __init__(self):
        raise OSError("AutoProcessor must be created with from_config/from_pretrained.")

    @classmethod
    def from_config(
        cls,
        config: VoiceHubConfig,
        **kwargs,
    ) -> VoiceHubProcessor:
        """Construct the processor class declared by the model."""
        spec = get_model_spec(config.model_type)
        model_class = _load_class(spec.module, spec.class_name)
        return model_class.processor_class(**kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        model_type: str | None = None,
        config: VoiceHubConfig | None = None,
        **kwargs,
    ) -> VoiceHubProcessor:
        """Construct a processor without importing a model runtime."""
        if config is None:
            if model_type is None:
                config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            else:
                config = AutoConfig.for_model(
                    model_type,
                    name_or_path=pretrained_model_name_or_path,
                )
        processor = cls.from_config(config, **kwargs)
        source = Path(pretrained_model_name_or_path).expanduser()
        processor_path = source / "processor_config.json"
        if source.is_dir() and processor_path.is_file():
            return processor.__class__.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
            )
        return processor
