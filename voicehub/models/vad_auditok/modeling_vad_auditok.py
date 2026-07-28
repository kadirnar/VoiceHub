"""Lazy Auditok energy-based VAD wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedVADModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import SpeechSegment, VADOutput
from voicehub.models.vad_auditok.configuration_vad_auditok import AuditokVADConfig
from voicehub.vad_utils import merge_speech_segments


class AuditokVADForVoiceActivityDetection(PreTrainedVADModel):
    """Detect energetic speech regions without allocating neural weights.

    Auditok is an energy event detector rather than a calibrated speech
    classifier. VoiceHub therefore exposes its energy and calibration
    controls on :class:`AuditokVADConfig` and reports no frame
    probabilities.
    """

    config_class = AuditokVADConfig
    default_model_name_or_path = "auditok-energy-vad"
    training_support = "inference-only"
    supports_generic_finetuning = False

    def __init__(
        self,
        config: AuditokVADConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "cpu",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device not in {"auto", "cpu"}:
            raise ValueError("Auditok is CPU-only; use `device='cpu'`.")
        return "cpu"

    def _load_pretrained_model(self) -> None:
        auditok = import_optional(
            "auditok",
            model_type=self.config.model_type,
            install_extra=None,
        )
        if not callable(getattr(auditok, "split", None)):
            raise RuntimeError("The installed Auditok package does not expose auditok.split().")
        self.model = auditok

    def _detect(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        threshold: float = 0.5,
        onset: float | None = None,
        offset: float | None = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        max_speech_duration_s: float | None = None,
        window_size_samples: int | None = None,
        return_frames: bool = False,
    ) -> VADOutput:
        unsupported = []
        if threshold != 0.5:
            unsupported.append("threshold")
        if onset is not None:
            unsupported.append("onset")
        if offset is not None:
            unsupported.append("offset")
        if return_frames:
            unsupported.append("return_frames")
        if unsupported:
            formatted = ", ".join(f"`{name}`" for name in unsupported)
            raise ValueError(
                "Auditok exposes an energy threshold in decibels, not "
                f"calibrated speech probabilities, and cannot honor: {formatted}.")

        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        np = import_optional(
            "numpy",
            model_type=self.config.model_type,
            install_extra=None,
        )
        analysis_window_s = self.config.analysis_window_s
        if window_size_samples is not None:
            analysis_window_s = window_size_samples / materialized.sampling_rate
            if not 0.01 <= analysis_window_s <= 0.1:
                raise ValueError(
                    "`window_size_samples` must resolve to an Auditok analysis "
                    "window between 0.01 and 0.1 seconds.")

        pcm = np.clip(materialized.waveform, -1.0, 1.0)
        pcm_bytes = (pcm * 32767.0).round().astype("<i2", copy=False).tobytes()
        minimum_duration = max(
            min_speech_duration_ms / 1000,
            analysis_window_s,
        )
        if (max_speech_duration_s is not None and max_speech_duration_s < minimum_duration):
            raise ValueError(
                "`max_speech_duration_s` cannot be shorter than Auditok's "
                "effective minimum speech duration.")
        options = {
            "min_dur": minimum_duration,
            "max_dur": max_speech_duration_s,
            "max_silence": min_silence_duration_ms / 1000,
            "max_leading_silence": speech_pad_ms / 1000,
            "max_trailing_silence": speech_pad_ms / 1000,
            "strict_min_dur": self.config.strict_min_duration,
            "analysis_window": analysis_window_s,
            "sr": materialized.sampling_rate,
            "sw": 2,
            "ch": 1,
        }
        if self.config.threshold_method == "fixed":
            options["energy_threshold"] = self.config.energy_threshold_db
        else:
            options.update({
                "validator": self.config.threshold_method,
                "calibration_dur": self.config.calibration_duration_s,
                "min_energy_threshold": self.config.minimum_energy_threshold_db,
            })

        regions = tuple(self.model.split(pcm_bytes, **options))
        segments = []
        for region in regions:
            start = getattr(region, "start", None)
            end = getattr(region, "end", None)
            if start is None or end is None:
                raise RuntimeError("Auditok split regions must expose absolute `start` and `end` times.")
            start = max(0.0, float(start))
            end = min(materialized.duration, float(end))
            if end <= start:
                continue
            segments.append(
                SpeechSegment(
                    start=start,
                    end=end,
                    score=None,
                    metadata={
                        "decision": "energy",
                        "threshold_method": self.config.threshold_method,
                    },
                ))

        return VADOutput(
            segments=merge_speech_segments(segments),
            duration=materialized.duration,
            sample_rate=materialized.sampling_rate,
            probabilities=None,
            metadata={
                "backend":
                "auditok",
                "algorithm":
                "short-term-energy",
                "threshold_method":
                self.config.threshold_method,
                "energy_threshold_db":
                (self.config.energy_threshold_db if self.config.threshold_method == "fixed" else None),
                "analysis_window_s":
                analysis_window_s,
                "effective_min_speech_duration_s":
                minimum_duration,
                "frame_scores_available":
                False,
            },
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "Auditok is an algorithmic energy detector with no trainable "
            "checkpoint. Fine-tuning is not applicable.")
