"""XTTS v2 inference backed by the vendored Coqui architecture source."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, seeded_inference


class XTTSConfig(VoiceHubConfig):
    """Configuration for multilingual XTTS v2 voice cloning."""

    model_type = "xtts"

    def __init__(
        self,
        *,
        language: str = "en",
        training_text_loss_weight: float = 0.01,
        training_mel_loss_weight: float = 1.0,
        training_dvae_checkpoint: str | None = None,
        training_mel_norm_file: str | None = None,
        training_lr_milestones: tuple[int, ...] = (
            900_000,
            2_700_000,
            5_400_000,
        ),
        training_lr_gamma: float = 0.5,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.language = language
        self.training_text_loss_weight = training_text_loss_weight
        self.training_mel_loss_weight = training_mel_loss_weight
        self.training_dvae_checkpoint = training_dvae_checkpoint
        self.training_mel_norm_file = training_mel_norm_file
        self.training_lr_milestones = training_lr_milestones
        self.training_lr_gamma = training_lr_gamma


class XTTSForTextToSpeech(PreTrainedTTSModel):
    """Source-integrated XTTS v2 synthesis."""

    config_class = XTTSConfig
    default_model_name_or_path = "coqui/XTTS-v2"

    def __init__(
        self,
        config: XTTSConfig | str | None = None,
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
        self._xtts_config = None
        self._model_directory = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _checkpoint_sample_rate(xtts_config: Any) -> int:
        output_sample_rate = getattr(
            getattr(xtts_config, "audio", None),
            "output_sample_rate",
            None,
        )
        if (not isinstance(output_sample_rate, int) or isinstance(output_sample_rate, bool) or
                output_sample_rate <= 0):
            raise ValueError(
                "XTTS checkpoint configuration must define a positive "
                "`audio.output_sample_rate`.")
        return output_sample_rate

    def _load_pretrained_model(self) -> None:
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="xtts",
        )
        config_module = import_optional(
            "voicehub.models.xtts.source.TTS.tts.configs.xtts_config",
            model_type="xtts",
            install_extra="xtts",
        )
        model_module = import_optional(
            "voicehub.models.xtts.source.TTS.tts.models.xtts",
            model_type="xtts",
            install_extra="xtts",
        )
        xtts_config = config_module.XttsConfig()
        xtts_config.load_json(str(model_directory / "config.json"))
        model = model_module.Xtts.init_from_config(xtts_config)
        model.load_checkpoint(
            xtts_config,
            checkpoint_dir=str(model_directory),
            eval=not self.is_training_load,
        )
        model.to(self.device)
        self.config.sample_rate = self._checkpoint_sample_rate(xtts_config)
        self._xtts_config = xtts_config
        self._model_directory = model_directory
        self._loaded_for_training = self.is_training_load
        self.model = model

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            return
        self.model = None
        self._loading_for_training = True
        try:
            self.load()
        finally:
            self._loading_for_training = False

    def _prepare_for_inference(self) -> None:
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip():
            raise ValueError("`speaker_audio_path` must point to a local XTTS reference-audio file.")
        reference_path = Path(speaker_audio_path).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"XTTS reference audio was not found: {reference_path}.")

        language = model_inputs.get("language") or self.config.language
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty XTTS language code.")

    def _resolve_language(self, language: str | None) -> str:
        normalized = (language or self.config.language).strip().lower()
        supported = tuple(getattr(self._xtts_config, "languages", ()))
        candidate = "zh-cn" if normalized == "zh" else normalized
        if supported and candidate not in supported:
            available = ", ".join(supported)
            raise ValueError(f"Unsupported XTTS language {language!r}. Supported: {available}.")
        return candidate

    @staticmethod
    def _extract_audio(result: Any):
        if isinstance(result, Mapping):
            if "wav" not in result:
                raise RuntimeError("XTTS result does not contain the expected `wav` field.")
            audio = result["wav"]
        else:
            audio = result
        if audio is None:
            raise RuntimeError("XTTS returned no audio waveform.")
        if hasattr(audio, "numel"):
            sample_count = audio.numel()
        elif hasattr(audio, "size"):
            sample_count = audio.size
        else:
            sample_count = len(audio)
        if sample_count == 0:
            raise RuntimeError("XTTS returned an empty audio waveform.")
        return audio

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str | None = None,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        language = self._resolve_language(language)
        with seeded_inference(
                seed,
                device=self.device,
                model_type="xtts",
        ) as effective_seed:
            result = self.model.synthesize(
                text,
                self._xtts_config,
                speaker_wav=str(Path(speaker_audio_path).expanduser()),
                language=language,
                **generation_options,
            )
        return finish_audio_output(
            self._extract_audio(result),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "language": language,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


XTTS = XTTSForTextToSpeech
