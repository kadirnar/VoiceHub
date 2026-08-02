"""Public model factory backed by the lazy VoiceHub registry."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module
from pathlib import Path

from voicehub.dependencies import import_optional
from voicehub.inference_strategy import InferenceStrategy
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models.registry import MODEL_REGISTRY, ModelSpec, get_default_model_spec, get_model_spec, list_model_specs
from voicehub.tasks import SpeechTask


class _ModelClassNameMapping(Mapping[str, str]):
    """Live compatibility view over the mutable model registry."""

    def __getitem__(self, model_type: str) -> str:
        return MODEL_REGISTRY[model_type].class_name

    def __iter__(self) -> Iterator[str]:
        return iter(MODEL_REGISTRY)

    def __len__(self) -> int:
        return len(MODEL_REGISTRY)


# Kept as a read-only compatibility view for callers that used the old mapping.
MODEL_TYPE_TO_MODEL_CLASS_NAME: Mapping[str, str] = _ModelClassNameMapping()

_TASK_FACTORY_NAMES = {
    SpeechTask.AUTOMATIC_SPEECH_RECOGNITION: "AutoModelForSpeechRecognition",
    SpeechTask.VOICE_ACTIVITY_DETECTION: "AutoModelForVoiceActivityDetection",
}


class AutoInferenceModel:
    """Factory class that dynamically loads and instantiates TTS model
    backends.

    Uses a registry mapping to resolve short model-type strings to their
    concrete inference classes, importing the appropriate module on
    demand so that unused backends are never loaded.
    """

    @classmethod
    def available_models(cls) -> tuple[ModelSpec, ...]:
        """List TTS backends without importing their ML runtimes."""
        return list_model_specs(task=SpeechTask.TEXT_TO_SPEECH)

    @classmethod
    def from_pretrained(
        cls,
        model_type: str | None = None,
        model_path: str | Path | None = None,
        device: str = "cuda",
        inference_strategy: str | InferenceStrategy | None = None,
        **kwargs,
    ):
        """Dynamically load and instantiate the appropriate model class.

        Args:
            model_type: Registry key or documented alias. When omitted, use
                the registry-declared TTS default.
            model_path: Hugging Face id or local path. Each backend has a
                sensible default when this is omitted.
            device: Target device (``"cuda"`` or ``"cpu"``).
            inference_strategy: Registered strategy name or configured
                strategy instance applied before the first inference load.
            **kwargs: Additional keyword arguments passed to the model class
                (e.g. ``lang_code`` for Kokoro).

        Returns:
            An instance of the resolved inference model class, ready for use.

        Raises:
            UnknownModelError: If *model_type* is not registered.
            OptionalDependencyError: If the selected backend is not installed.
        """
        if model_type is None:
            spec = get_default_model_spec(SpeechTask.TEXT_TO_SPEECH)
            if spec is None:
                raise ValueError(
                    "AutoInferenceModel has no registry-declared TTS default. "
                    "Pass `model_type` explicitly.")
        else:
            spec = get_model_spec(model_type)
        if spec.task is not SpeechTask.TEXT_TO_SPEECH:
            factory_name = _TASK_FACTORY_NAMES[spec.task]
            raise ValueError(
                "AutoInferenceModel is the legacy text-to-speech factory, "
                f"but {model_type!r} is registered for task "
                f"{spec.task.value!r}. Use {factory_name} instead.")
        lazy_load = kwargs.pop("lazy_load", True)
        try:
            module = import_module(spec.module)
        except ModuleNotFoundError:
            module = import_optional(
                spec.module,
                model_type=spec.model_type,
                install_extra=spec.install_extra,
            )

        inference_model = getattr(module, spec.class_name)
        resolved_path = spec.default_model_path if model_path is None else model_path
        if issubclass(inference_model, PreTrainedTTSModel):
            from voicehub.auto import AutoConfig

            config = AutoConfig.for_model(
                spec.model_type,
                name_or_path=resolved_path,
                **kwargs,
            )
            model = inference_model(
                config,
                device=device,
                lazy_load=(True if inference_strategy is not None else lazy_load),
            )
            if inference_strategy is not None:
                model.set_inference_strategy(inference_strategy)
                if not lazy_load:
                    model.load()
            return model
        if inference_strategy is not None:
            raise TypeError(
                f"{inference_model.__name__} does not support "
                "`inference_strategy`; it must inherit PreTrainedTTSModel.")
        return inference_model(
            model_path=resolved_path,
            device=device,
            **kwargs,
        )
