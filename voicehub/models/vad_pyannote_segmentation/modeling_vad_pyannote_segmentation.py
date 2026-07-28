"""Pyannote segmentation-3.0 model-to-pipeline adapter."""

from __future__ import annotations

from inspect import Parameter, signature

from voicehub.dependencies import import_optional
from voicehub.models.vad_pyannote.modeling_vad_pyannote import PyannoteVADForVoiceActivityDetection
from voicehub.models.vad_pyannote_segmentation.configuration_vad_pyannote_segmentation import (
    PyannoteSegmentationVADConfig, )


def _loader_options(loader, values: dict, *, required: tuple[str, ...]) -> dict:
    try:
        parameters = signature(loader).parameters
    except (TypeError, ValueError):
        return {name: value for name, value in values.items() if value is not None}
    accepts_kwargs = any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values())
    options = {
        name: value
        for name, value in values.items() if value is not None and (accepts_kwargs or name in parameters)
    }
    missing = sorted(name for name in required if name not in options)
    if missing:
        raise RuntimeError(
            "The installed pyannote.audio version does not support requested "
            f"model loading option(s): {', '.join(missing)}.")
    return options


class PyannoteSegmentationVADForVoiceActivityDetection(PyannoteVADForVoiceActivityDetection):
    """Use ``pyannote/segmentation-3.0`` as a full-recording VAD pipeline."""

    config_class = PyannoteSegmentationVADConfig
    default_model_name_or_path = "pyannote/segmentation-3.0"
    training_support = "upstream-custom"
    supports_generic_finetuning = False

    def _load_pretrained_model(self) -> None:
        pyannote_audio = import_optional(
            "pyannote.audio",
            model_type=self.config.model_type,
            install_extra=None,
        )
        pipelines = import_optional(
            "pyannote.audio.pipelines",
            model_type=self.config.model_type,
            install_extra=None,
        )
        model_class = getattr(pyannote_audio, "Model", None)
        loader = getattr(model_class, "from_pretrained", None)
        pipeline_class = getattr(pipelines, "VoiceActivityDetection", None)
        if not callable(loader) or not callable(pipeline_class):
            raise RuntimeError(
                "The installed pyannote.audio package must expose "
                "Model.from_pretrained() and pipelines.VoiceActivityDetection().")

        configured = {
            "revision": self.config.revision,
            "subfolder": self.config.subfolder,
            "cache_dir": self.config.cache_dir,
            "token": self._auth_token,
        }
        required = tuple(name for name, value in configured.items() if value is not None)
        options = _loader_options(
            loader,
            configured,
            required=required,
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        segmentation = loader(source, **options)
        if segmentation is None:
            raise RuntimeError(f"pyannote.audio could not load segmentation model {source!r}.")

        pipeline = pipeline_class(
            segmentation=segmentation,
            **dict(self.config.pipeline_kwargs),
        )
        if self.device != "cpu":
            move = getattr(pipeline, "to", None)
            if callable(move):
                torch = import_optional(
                    "torch",
                    model_type=self.config.model_type,
                    install_extra=None,
                )
                move(torch.device(self.device))
        self.model = pipeline

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "pyannote segmentation-3.0 fine-tuning is upstream-custom and "
            "requires a pyannote.audio segmentation task and database protocol.")
