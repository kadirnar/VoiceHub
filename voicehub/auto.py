"""Automatic configuration, processor, and task-specific model factories."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.errors import UnknownModelError
from voicehub.hub import read_json_file, resolve_pretrained_file
from voicehub.inference_strategy import InferenceStrategy
from voicehub.models.registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_model_spec,
    register_model_spec,
    unregister_model_spec,
)
from voicehub.processing_utils import VoiceHubProcessor
from voicehub.tasks import SpeechTask


def _load_class(module_name: str, class_name: str):
    module = import_module(module_name)
    return getattr(module, class_name)


_FACTORY_NAME_BY_TASK = {
    SpeechTask.TEXT_TO_SPEECH: "AutoModelForTextToSpeech",
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: "AutoModelForSpeechRecognition",
    SpeechTask.VOICE_ACTIVITY_DETECTION: "AutoModelForVoiceActivityDetection",
}


def _get_task_model_spec(
    model_type: str,
    *,
    expected_task: SpeechTask,
) -> ModelSpec:
    """Resolve *model_type* and reject cross-task factory usage early."""
    spec = get_model_spec(model_type)
    if spec.task is expected_task:
        return spec

    expected_factory = _FACTORY_NAME_BY_TASK[expected_task]
    registered_factory = _FACTORY_NAME_BY_TASK[spec.task]
    raise ValueError(
        f"Model {model_type!r} is registered for task {spec.task.value!r}, "
        f"so it cannot be loaded by {expected_factory}. "
        f"Use {registered_factory} instead.")


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
            normalized_model_type = (
                model_type.strip().lower() if isinstance(model_type, str) else model_type)
            if normalized_model_type not in MODEL_REGISTRY:
                raise UnknownModelError(
                    f"Checkpoint model type {model_type!r} is not a canonical "
                    "VoiceHub provider. Select a task-specific auto factory or "
                    "pass `model_type` explicitly.")

        spec = get_model_spec(model_type)
        config_class = _load_class(spec.config_module, spec.config_class)
        config = config_class.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        if not config.architectures:
            config.architectures = [spec.class_name]
        return config


class _BaseAutoModel:
    """Shared implementation for task-constrained speech-model factories."""

    task: SpeechTask
    default_model_type: str | None = None

    def __init__(self):
        raise OSError(f"{self.__class__.__name__} must be created with "
                      "from_config/from_pretrained.")

    @classmethod
    def _get_spec(cls, model_type: str) -> ModelSpec:
        return _get_task_model_spec(
            model_type,
            expected_task=cls.task,
        )

    @classmethod
    def available_models(cls) -> tuple[ModelSpec, ...]:
        """List compatible backends without importing their runtimes."""
        from voicehub.models.registry import list_model_specs

        return list_model_specs(task=cls.task)

    @classmethod
    def register(
        cls,
        config_class: type[VoiceHubConfig],
        model_class: type,
        *,
        model_type: str | None = None,
        default_model_path: str = "",
        aliases: tuple[str, ...] = (),
        capabilities: tuple[str, ...] = (),
        architecture: str | None = None,
        install_extra: str | None = None,
        exist_ok: bool = False,
    ) -> ModelSpec:
        """Register a model/config pair for this task-specific factory.

        The signature mirrors the extension flow used by Transformers
        auto classes while retaining VoiceHub's task boundary. The
        registry stores import paths, not the class objects, so normal
        discovery stays lazy.
        """
        if not isinstance(config_class, type) or not issubclass(
                config_class,
                VoiceHubConfig,
        ):
            raise TypeError("`config_class` must inherit VoiceHubConfig.")
        resolved_model_type = model_type or getattr(config_class, "model_type", None)
        if not isinstance(resolved_model_type, str) or not resolved_model_type.strip():
            raise ValueError("`model_type` is required when config_class has no model_type.")
        if resolved_model_type.strip().lower() == "voicehub":
            raise ValueError("Extension config classes must declare a unique `model_type`.")
        spec = ModelSpec.from_classes(
            model_type=resolved_model_type,
            model_class=model_class,
            config_class=config_class,
            default_model_path=default_model_path,
            install_extra=install_extra,
            capabilities=capabilities,
            task=cls.task,
            architecture=architecture,
        )
        register_model_spec(
            spec,
            aliases=aliases,
            exist_ok=exist_ok,
        )
        return spec

    @classmethod
    def unregister(
        cls,
        model_type: str,
        *,
        missing_ok: bool = False,
    ) -> ModelSpec | None:
        """Unregister a model owned by this task-specific factory."""
        try:
            spec = get_model_spec(model_type)
        except UnknownModelError:
            if missing_ok:
                return None
            raise
        if spec.task is not cls.task:
            raise ValueError(
                f"Model {model_type!r} belongs to {spec.task.value!r}, not "
                f"{cls.task.value!r}.")
        return unregister_model_spec(model_type, missing_ok=missing_ok)

    @classmethod
    def from_config(
        cls,
        config: VoiceHubConfig,
        *,
        inference_strategy: str | InferenceStrategy | None = None,
        llm_backend=None,
        llm_backend_config=None,
        optimization_config=None,
        attn_implementation: str | None = None,
        kernel_backend: str | None = None,
        torch_compile: bool | str | None = None,
        compile_config=None,
        diffusion_cache: bool | str | None = None,
        diffusion_cache_config=None,
        diffusion_sampling: bool | str | None = None,
        diffusion_sampling_config=None,
        **kwargs,
    ):
        spec = cls._get_spec(config.model_type)
        model_class = _load_class(spec.module, spec.class_name)
        from voicehub.optimization import tts_optimization_config_from_options

        resolved_optimization_config = (
            tts_optimization_config_from_options(
                optimization_config,
                attn_implementation=attn_implementation,
                kernel_backend=kernel_backend,
                torch_compile=torch_compile,
                compile_config=compile_config,
                diffusion_cache=diffusion_cache,
                diffusion_cache_config=diffusion_cache_config,
                diffusion_sampling=diffusion_sampling,
                diffusion_sampling_config=diffusion_sampling_config,
            ) if cls.task is SpeechTask.TEXT_TO_SPEECH else None)
        if cls.task is not SpeechTask.TEXT_TO_SPEECH and any(value is not None for value in (
                optimization_config,
                attn_implementation,
                kernel_backend,
                torch_compile,
                compile_config,
                diffusion_cache,
                diffusion_cache_config,
                diffusion_sampling,
                diffusion_sampling_config,
                llm_backend,
                llm_backend_config,
        )):
            raise TypeError(
                "TTS optimization and LLM-serving arguments are available only through "
                "AutoModelForTextToSpeech.")
        eager_load = kwargs.get("lazy_load", True) is False
        configure_external_backend = (llm_backend is not None or llm_backend_config is not None)
        if configure_external_backend and resolved_optimization_config is not None:
            raise ValueError(
                "Choose either an external LLM backend or an in-process TTS "
                "optimization configuration.")
        if ((inference_strategy is not None or resolved_optimization_config is not None or
             configure_external_backend) and eager_load):
            kwargs["lazy_load"] = True
        model = model_class(config, **kwargs)
        if inference_strategy is not None:
            model.set_inference_strategy(inference_strategy)
        if resolved_optimization_config is not None:
            model.set_optimization_config(resolved_optimization_config)
        if configure_external_backend:
            configured_backend = llm_backend
            if configured_backend is None and isinstance(llm_backend_config, Mapping):
                configured_backend = llm_backend_config.get("backend")
            if configured_backend is None:
                from voicehub.llm_serving import LLMBackendConfig

                if isinstance(llm_backend_config, LLMBackendConfig):
                    configured_backend = llm_backend_config.backend
            if configured_backend is None:
                raise ValueError(
                    "`llm_backend` is required when "
                    "`llm_backend_config` does not declare a backend.")
            model.set_llm_backend(
                configured_backend,
                config=llm_backend_config,
            )
        if eager_load and (inference_strategy is not None or resolved_optimization_config is not None or
                           configure_external_backend):
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
        config_kwargs: Mapping[str, object] | None = None,
        **kwargs,
    ):
        if config_kwargs is None:
            config_values = {}
        elif isinstance(config_kwargs, Mapping):
            config_values = dict(config_kwargs)
        else:
            raise TypeError("`config_kwargs` must be a mapping or None.")
        if any(not isinstance(key, str) or not key for key in config_values):
            raise ValueError("`config_kwargs` keys must be non-empty strings.")
        if "model_type" in config_values:
            raise ValueError(
                "Pass `model_type` as the top-level factory argument, not "
                "inside `config_kwargs`.")
        if config is not None and config_values:
            raise TypeError("Pass configuration through either `config` or "
                            "`config_kwargs`, not both.")

        empty_source = (
            isinstance(pretrained_model_name_or_path, str) and not pretrained_model_name_or_path.strip())
        if config is None:
            if model_type is None:
                if empty_source:
                    if cls.default_model_type is None:
                        raise ValueError(
                            f"{cls.__name__} has no default checkpoint. Pass a "
                            "model path or `model_type`.")
                    spec = cls._get_spec(cls.default_model_type)
                    config_class = _load_class(
                        spec.config_module,
                        spec.config_class,
                    )
                    config = config_class(
                        name_or_path=spec.default_model_path,
                        **config_values,
                    )
                else:
                    try:
                        config = AutoConfig.from_pretrained(
                            pretrained_model_name_or_path,
                            **config_values,
                        )
                        spec = cls._get_spec(config.model_type)
                    except UnknownModelError:
                        if cls.default_model_type is None:
                            raise
                        spec = cls._get_spec(cls.default_model_type)
                        config_class = _load_class(
                            spec.config_module,
                            spec.config_class,
                        )
                        config = config_class(
                            name_or_path=pretrained_model_name_or_path,
                            **config_values,
                        )
            else:
                spec = cls._get_spec(model_type)
                config_class = _load_class(spec.config_module, spec.config_class)
                config = config_class(
                    name_or_path=(spec.default_model_path if empty_source else pretrained_model_name_or_path),
                    **config_values,
                )
                config.model_type = spec.model_type
        else:
            if model_type is not None and config.model_type == "voicehub":
                spec = cls._get_spec(model_type)
                config.model_type = spec.model_type
            else:
                configured_spec = cls._get_spec(config.model_type)
                if model_type is None:
                    spec = configured_spec
                else:
                    spec = cls._get_spec(model_type)
                if spec.model_type != configured_spec.model_type:
                    raise ValueError(
                        f"Explicit model_type {model_type!r} resolves to "
                        f"{spec.model_type!r}, but the supplied config targets "
                        f"{configured_spec.model_type!r}.")
        model_class = _load_class(spec.module, spec.class_name)
        return model_class.from_pretrained(
            "" if empty_source else pretrained_model_name_or_path,
            config=config,
            inference_strategy=inference_strategy,
            **kwargs,
        )


class AutoModelForTextToSpeech(_BaseAutoModel):
    """Load a registered text-to-speech model."""

    task = SpeechTask.TEXT_TO_SPEECH


class AutoModelForSpeechRecognition(_BaseAutoModel):
    """Load a registered automatic speech-recognition model."""

    task = SpeechTask.AUTOMATIC_SPEECH_RECOGNITION
    default_model_type = "asr_transformers"


class AutoModelForVoiceActivityDetection(_BaseAutoModel):
    """Load a registered voice-activity-detection model."""

    task = SpeechTask.VOICE_ACTIVITY_DETECTION
    default_model_type = "vad_transformers"


_AUTO_MODEL_FACTORY_BY_TASK = {
    SpeechTask.TEXT_TO_SPEECH: AutoModelForTextToSpeech,
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: AutoModelForSpeechRecognition,
    SpeechTask.VOICE_ACTIVITY_DETECTION: AutoModelForVoiceActivityDetection,
}


class AutoModel:
    """Task-aware entry point for every registered speech model."""

    def __init__(self):
        raise OSError("AutoModel must be created with from_config/from_pretrained.")

    @classmethod
    def available_models(
        cls,
        *,
        task: SpeechTask | str | None = None,
    ) -> tuple[ModelSpec, ...]:
        """List models, optionally filtered by speech task."""
        del cls
        from voicehub.models.registry import list_model_specs

        return list_model_specs(task=task)

    @classmethod
    def from_config(cls, config: VoiceHubConfig, **kwargs):
        """Dispatch a typed configuration to its task-specific factory."""
        del cls
        spec = get_model_spec(config.model_type)
        return _AUTO_MODEL_FACTORY_BY_TASK[spec.task].from_config(config, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = "",
        *,
        model_type: str | None = None,
        config: VoiceHubConfig | None = None,
        config_kwargs: Mapping[str, object] | None = None,
        **kwargs,
    ):
        """Load any registered model and dispatch by its declared task."""
        del cls
        if config is not None:
            spec = get_model_spec(config.model_type)
        elif model_type is not None:
            spec = get_model_spec(model_type)
        else:
            if not str(pretrained_model_name_or_path).strip():
                raise ValueError(
                    "AutoModel needs `model_type`, `config`, or a checkpoint "
                    "containing config.json.")
            if config_kwargs is None:
                config_values = {}
            elif isinstance(config_kwargs, Mapping):
                config_values = dict(config_kwargs)
            else:
                raise TypeError("`config_kwargs` must be a mapping or None.")
            if "model_type" in config_values:
                raise ValueError(
                    "Pass `model_type` as the top-level factory argument, not "
                    "inside `config_kwargs`.")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **config_values,
            )
            spec = get_model_spec(config.model_type)
            config_kwargs = None
        return _AUTO_MODEL_FACTORY_BY_TASK[spec.task].from_pretrained(
            pretrained_model_name_or_path,
            model_type=model_type,
            config=config,
            config_kwargs=config_kwargs,
            **kwargs,
        )

    @classmethod
    def register(
        cls,
        config_class: type[VoiceHubConfig],
        model_class: type,
        *,
        task: SpeechTask | str,
        **kwargs,
    ) -> ModelSpec:
        """Register a model through the factory selected by ``task``."""
        del cls
        factory = _AUTO_MODEL_FACTORY_BY_TASK[SpeechTask.coerce(task)]
        return factory.register(config_class, model_class, **kwargs)

    @classmethod
    def unregister(
        cls,
        model_type: str,
        *,
        missing_ok: bool = False,
    ) -> ModelSpec | None:
        """Unregister a model through the factory that owns its task."""
        del cls
        try:
            spec = get_model_spec(model_type)
        except UnknownModelError:
            if missing_ok:
                return None
            raise
        return _AUTO_MODEL_FACTORY_BY_TASK[spec.task].unregister(
            model_type,
            missing_ok=missing_ok,
        )


class AutoProcessor:
    """Create the processor paired with a VoiceHub speech configuration."""

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
