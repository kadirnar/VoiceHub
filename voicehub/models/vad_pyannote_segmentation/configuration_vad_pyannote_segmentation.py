"""Configuration for the pyannote segmentation-3.0 VAD preset."""

from collections.abc import Mapping

from voicehub.models.vad_pyannote.configuration_vad_pyannote import PyannoteVADConfig


class PyannoteSegmentationVADConfig(PyannoteVADConfig):
    """Configure pyannote's powerset segmentation checkpoint as VAD."""

    model_type = "vad_pyannote_segmentation"

    def __init__(self, *, pipeline_kwargs: Mapping | None = None, **kwargs):
        if pipeline_kwargs is not None and not isinstance(pipeline_kwargs, Mapping):
            raise TypeError("`pipeline_kwargs` must be a mapping or None.")
        if pipeline_kwargs is not None and "segmentation" in pipeline_kwargs:
            raise ValueError(
                "`pipeline_kwargs` cannot replace VoiceHub's managed "
                "segmentation checkpoint.")
        super().__init__(
            pipeline_kwargs=pipeline_kwargs,
            **kwargs,
        )
