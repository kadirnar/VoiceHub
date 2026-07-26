"""NeuTTS inference backed by vendored NeuTTS and NeuCodec source."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference


class NeuTTSConfig(VoiceHubConfig):
    """Configuration for NeuTTS Air, Nano, multilingual, and 2E variants."""

    model_type = "neutts"

    def __init__(
        self,
        *,
        codec_name_or_path: str = "neuphonic/neucodec",
        language: str | None = None,
        seed: int | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.language = language
        self.seed = seed


class NeuTTSForTextToSpeech(PreTrainedTTSModel):
    """Source-integrated NeuTTS synthesis with an embedded NeuCodec runtime."""

    config_class = NeuTTSConfig
    default_model_name_or_path = "neuphonic/neutts-2e"

    def __init__(
        self,
        config: NeuTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text")
        if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
            raise ValueError("NeuTTS requires `speaker_audio_path` and a non-empty "
                             "`reference_text`.")
        reference_path = Path(speaker_audio_path).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"NeuTTS reference audio was not found: {reference_path}.")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise ValueError("NeuTTS requires `speaker_audio_path` and a non-empty "
                             "`reference_text`.")
        temperature = model_inputs.get("temperature", 1.0)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("`temperature` must be a finite positive number.")
        top_k = model_inputs.get("top_k", 50)
        if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("`top_k` must be a positive integer.")
        emotion = model_inputs.get("emotion")
        if emotion is not None and (not isinstance(emotion, str) or not emotion.strip()):
            raise ValueError("`emotion` must be a non-empty string or None.")

    def _validate_training_runtime(self) -> None:
        if self.config.name_or_path.lower().endswith((".gguf", "-gguf")):
            raise ValueError(
                "NeuTTS fine-tuning requires a differentiable Transformers "
                "backbone; GGUF checkpoints are inference-only.")

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.neutts.source.neutts.neutts",
            model_type="neutts",
            install_extra="neutts",
        )
        self.model = runtime.NeuTTS(
            backbone_repo=self.config.name_or_path,
            backbone_device=self.device,
            codec_repo=self.config.codec_name_or_path,
            codec_device=self.device,
            language=self.config.language,
            seed=self.config.seed,
        )
        self.config.sample_rate = int(self.model.sample_rate)

    def _prepare_for_inference(self) -> None:
        """Restore eval/cache state on the existing NeuTTS components."""
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "eval"):
            backbone.eval()
        codec = getattr(self.model, "codec", None)
        if codec is not None and hasattr(codec, "eval"):
            codec.eval()
        model_config = getattr(backbone, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        emotion: str | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
        seed: int | None = None,
    ) -> TTSOutput:
        reference_codes = self.model.encode_reference(str(Path(speaker_audio_path).expanduser()))
        requested_seed = self.config.seed if seed is None else seed
        with seeded_inference(
                requested_seed,
                device=self.device,
                model_type="neutts",
        ) as fallback_seed:
            audio = self.model.infer(
                text,
                reference_codes,
                reference_text,
                emotion=emotion,
                temperature=temperature,
                top_k=top_k,
                seed=fallback_seed,
            )
            effective_seed = getattr(self.model, "last_seed", None)
            if effective_seed is None:
                effective_seed = fallback_seed
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "emotion": emotion,
                "seed": effective_seed,
                "requested_seed": requested_seed,
                "voice_cloned": True,
            },
        )


NeuTTSModel = NeuTTSForTextToSpeech
