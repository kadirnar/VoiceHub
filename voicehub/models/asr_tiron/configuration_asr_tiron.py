"""Configuration for Tiron speaker-attributed speech recognition."""

from __future__ import annotations

from voicehub.models.asr_transformers.configuration_asr_transformers import TransformersASRConfig


class TironASRConfig(TransformersASRConfig):
    """Configure a native Trelis Tiron Whisper checkpoint.

    Tiron uses ordinary differentiable Whisper weights, but its decoding
    grammar emits interleaved speaker, timestamp, and text tokens.  The
    dedicated runtime therefore locks the architecture family while
    retaining the common Transformers checkpoint, processor, and safe-
    export controls.
    """

    model_type = "asr_tiron"
    architecture_family_name = "speech-seq2seq"

    def __init__(
        self,
        *,
        default_language: str = "en",
        architecture_family: str | None = None,
        **kwargs,
    ):
        requested_family = (
            self.architecture_family_name
            if architecture_family is None else str(architecture_family).strip().lower().replace("_", "-"))
        if requested_family != self.architecture_family_name:
            raise ValueError(
                "TironASRConfig requires "
                "`architecture_family='speech-seq2seq'`; received "
                f"{architecture_family!r}.")
        self.default_language = default_language
        super().__init__(
            architecture_family=self.architecture_family_name,
            **kwargs,
        )

    def validate(self) -> None:
        """Validate Tiron-specific controls and inherited Hub options."""
        super().validate()
        if (not isinstance(self.default_language, str) or not self.default_language.strip()):
            raise ValueError("`default_language` must be a non-empty Whisper language code.")
        self.default_language = self.default_language.strip().lower()
        if self.pipeline_kwargs:
            raise ValueError(
                "`pipeline_kwargs` are not supported by Tiron because its "
                "speaker/timestamp grammar requires direct generation.")


__all__ = ["TironASRConfig"]
