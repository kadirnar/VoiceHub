"""Configuration for pyannote's Brouhaha multi-task model."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from voicehub.models.vad_pyannote.configuration_vad_pyannote import PyannoteVADConfig


def _optional_positive_number(value, *, name: str) -> float | None:
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(float(value)) or
            float(value) <= 0):
        raise ValueError(f"`{name}` must be finite and greater than zero or None.")
    return float(value)


class PyannoteBrouhahaVADConfig(PyannoteVADConfig):
    """Configure Brouhaha frame inference and VAD post-processing."""

    model_type = "vad_pyannote_brouhaha"

    def __init__(
        self,
        *,
        batch_size: int = 32,
        inference_duration_s: float | None = None,
        inference_step_s: float | None = None,
        pipeline_kwargs: Mapping | None = None,
        **kwargs,
    ):
        if pipeline_kwargs is not None and not isinstance(pipeline_kwargs, Mapping):
            raise TypeError("`pipeline_kwargs` must be a mapping or None.")
        if pipeline_kwargs:
            raise ValueError(
                "Brouhaha uses pyannote.Inference directly; "
                "`pipeline_kwargs` is not supported.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("`batch_size` must be a positive integer.")
        inference_duration_s = _optional_positive_number(
            inference_duration_s,
            name="inference_duration_s",
        )
        inference_step_s = _optional_positive_number(
            inference_step_s,
            name="inference_step_s",
        )
        if (inference_duration_s is not None and inference_step_s is not None and
                inference_step_s > inference_duration_s):
            raise ValueError("`inference_step_s` cannot exceed `inference_duration_s`.")
        super().__init__(
            batch_size=batch_size,
            inference_duration_s=inference_duration_s,
            inference_step_s=inference_step_s,
            pipeline_kwargs={},
            **kwargs,
        )
