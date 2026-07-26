"""Public model factory backed by the lazy VoiceHub registry."""

from __future__ import annotations

from importlib import import_module

from voicehub.dependencies import import_optional
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.registry import MODEL_REGISTRY, ModelSpec, get_model_spec, list_model_specs

# Kept as a read-only compatibility view for callers that used the old mapping.
MODEL_TYPE_TO_MODEL_CLASS_NAME = {name: spec.class_name for name, spec in MODEL_REGISTRY.items()}


class AutoInferenceModel:
    """
    Factory class that dynamically loads and instantiates TTS model backends.

    Uses a registry mapping to resolve short model-type strings to their concrete inference classes, importing
    the appropriate module on demand so that unused backends are never loaded.
    """

    @classmethod
    def available_models(cls) -> tuple[ModelSpec, ...]:
        """List supported backends without importing their ML runtimes."""
        return list_model_specs()

    @classmethod
    def from_pretrained(
        cls,
        model_type: str = "orpheustts",
        model_path: str | None = None,
        device: str = "cuda",
        **kwargs,
    ):
        """
        Dynamically load and instantiate the appropriate model class.

        Args:
            model_type: Registry key or documented alias.
            model_path: Hugging Face id or local path. Each backend has a
                sensible default when this is omitted.
            device: Target device (``"cuda"`` or ``"cpu"``).
            **kwargs: Additional keyword arguments passed to the model class
                (e.g. ``lang_code`` for Kokoro).

        Returns:
            An instance of the resolved inference model class, ready for use.

        Raises:
            UnknownModelError: If *model_type* is not registered.
            OptionalDependencyError: If the selected backend is not installed.
        """
        spec = get_model_spec(model_type)
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
            return inference_model(
                config,
                device=device,
                lazy_load=lazy_load,
            )
        return inference_model(
            model_path=resolved_path,
            device=device,
            **kwargs,
        )
