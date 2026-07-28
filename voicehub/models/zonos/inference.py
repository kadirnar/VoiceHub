"""Zonos v0.1 integration backed by vendored Zyphra source."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference


class ZonosConfig(VoiceHubConfig):
    """Configuration for Zonos transformer and hybrid checkpoints."""

    model_type = "zonos"

    def __init__(self, *, sample_rate: int = 44100, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)


class ZonosForTextToSpeech(PreTrainedTTSModel):
    """Expressive multilingual synthesis and zero-shot voice cloning."""

    config_class = ZonosConfig
    default_model_name_or_path = "Zyphra/Zonos-v0.1-transformer"

    def __init__(
        self,
        config: ZonosConfig | str | None = None,
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
        self._conditioning = None
        self._requested_device = device
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _runtime_device(self) -> str:
        # The released Zonos implementation explicitly does not support MPS
        # yet. ``auto`` otherwise selects it on Apple Silicon and fails while
        # moving the bfloat16 checkpoint.
        if str(self.device).split(":", 1)[0].lower() == "mps":
            self.device = "cpu"
        return self.device

    def _load_pretrained_model(self) -> None:
        runtime_device = self._runtime_device()
        modeling = import_optional(
            "voicehub.models.zonos.source.zonos.model",
            model_type="zonos",
            install_extra=None,
        )
        self._conditioning = import_optional(
            "voicehub.models.zonos.source.zonos.conditioning",
            model_type="zonos",
            install_extra=None,
        )
        source = Path(self.config.name_or_path).expanduser()
        if source.is_dir():
            config_path = source / "config.json"
            weights_path = source / "model.safetensors"
            missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError("Missing local Zonos checkpoint asset(s): " + ", ".join(missing))
            self.model = modeling.Zonos.from_local(
                str(config_path),
                str(weights_path),
                device=runtime_device,
            )
        elif source.exists():
            raise NotADirectoryError(f"Expected a local Zonos model directory, received file: {source}.")
        else:
            self.model = modeling.Zonos.from_pretrained(
                self.config.name_or_path,
                device=runtime_device,
            )
        self.config.sample_rate = int(self.model.autoencoder.sampling_rate)

    def _prepare_for_training(self) -> None:
        """Keep the codec frozen and expose the unfused token model."""
        self.model._cg_graph = None
        self.model._cg_batch_size = None
        autoencoder = getattr(self.model, "autoencoder", None)
        if autoencoder is not None:
            codec = getattr(autoencoder, "dac", autoencoder)
            if hasattr(codec, "eval"):
                codec.eval()
            if hasattr(codec, "parameters"):
                for parameter in codec.parameters():
                    parameter.requires_grad_(False)
        self.model.train()

    def _prepare_for_inference(self) -> None:
        self.model._cg_graph = None
        self.model._cg_batch_size = None
        self.model.eval()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        language = model_inputs.get("language", "en-us")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty eSpeak language code.")

        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if speaker_audio_path is not None:
            if not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip():
                raise ValueError("`speaker_audio_path` must be a local audio path or None.")
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"Zonos reference audio was not found: {reference_path}.")

        for name, default, maximum in (
            ("speaking_rate", 15.0, 40.0),
            ("pitch_std", 20.0, 400.0),
        ):
            value = model_inputs.get(name, default)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or
                    not 0 <= value <= maximum):
                raise ValueError(f"`{name}` must be finite and in the interval "
                                 f"[0, {maximum:g}].")
        cfg_scale = model_inputs.get("cfg_scale", 2.0)
        if (not isinstance(cfg_scale, (int, float)) or isinstance(cfg_scale, bool) or
                not math.isfinite(cfg_scale) or cfg_scale < 0):
            raise ValueError("`cfg_scale` must be a finite non-negative number.")
        if cfg_scale == 1:
            raise ValueError(
                "`cfg_scale` cannot be 1 because Zonos classifier-free "
                "guidance requires distinct conditional and unconditional batches.")

        max_new_tokens = model_inputs.get("max_new_tokens", 2580)
        if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        seed = model_inputs.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("`seed` must be an integer or None.")

        emotion = model_inputs.get("emotion")
        if emotion is not None:
            if not isinstance(emotion, (list, tuple)) or len(emotion) != 8:
                raise ValueError("`emotion` must contain eight values in the order expected "
                                 "by Zonos.")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                   not math.isfinite(value) or not 0 <= value <= 1 for value in emotion):
                raise ValueError("Every `emotion` value must be finite and in the "
                                 "interval [0, 1].")
            if sum(emotion) <= 0:
                raise ValueError("At least one `emotion` value must be positive.")

    def _speaker_embedding(self, speaker_audio_path: str | None):
        if speaker_audio_path is None:
            return None
        torchaudio = import_optional(
            "torchaudio",
            model_type="zonos",
            install_extra=None,
        )
        waveform, sample_rate = torchaudio.load(str(Path(speaker_audio_path).expanduser()))
        if waveform.numel() == 0:
            raise ValueError("Zonos reference audio contains no samples.")
        return self.model.make_speaker_embedding(waveform, sample_rate)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str = "en-us",
        emotion: list[float] | None = None,
        speaking_rate: float = 15.0,
        pitch_std: float = 20.0,
        cfg_scale: float = 2.0,
        max_new_tokens: int = 2580,
        seed: int | None = None,
        **sampling_options,
    ) -> TTSOutput:
        normalized_language = language.strip().lower()
        supported_languages = getattr(
            self._conditioning,
            "supported_language_codes",
            (),
        )
        if supported_languages and normalized_language not in supported_languages:
            raise ValueError(
                f"Unsupported Zonos language {language!r}. "
                "Use an eSpeak-compatible language code.")
        with seeded_inference(
                seed,
                device=self.device,
                model_type="zonos",
        ) as effective_seed:
            speaker = self._speaker_embedding(speaker_audio_path)
            condition_kwargs = {
                "text": text,
                "language": normalized_language,
                "speaker": speaker,
                "speaking_rate": speaking_rate,
                "pitch_std": pitch_std,
                "device": self.device,
            }
            if emotion is not None:
                condition_kwargs["emotion"] = emotion
            condition = self._conditioning.make_cond_dict(**condition_kwargs)
            prefix = self.model.prepare_conditioning(condition)
            codes = self.model.generate(
                prefix,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                sampling_params=sampling_options or {"min_p": 0.1},
            )
        if codes is None or codes.numel() == 0:
            raise RuntimeError("Zonos returned no audio codes.")
        audio = self.model.autoencoder.decode(codes).cpu()[0]
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "language": normalized_language,
                "seed": effective_seed,
                "voice_cloned": speaker_audio_path is not None,
            },
        )


ZonosTTS = ZonosForTextToSpeech
