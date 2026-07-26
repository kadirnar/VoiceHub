"""F5-TTS integration backed by the vendored upstream implementation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output


class F5TTSConfig(VoiceHubConfig):
    """Configuration for the source-integrated F5-TTS architecture."""

    model_type = "f5tts"

    def __init__(
        self,
        *,
        model_name: str = "F5TTS_v1_Base",
        checkpoint_path: str = "",
        vocabulary_path: str = "",
        ode_method: str = "euler",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        ema_update_after_step: int = 0,
        ema_update_every: int = 1,
        vocoder_path: str | None = None,
        cache_dir: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.vocabulary_path = vocabulary_path
        self.ode_method = ode_method
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_after_step = ema_update_after_step
        self.ema_update_every = ema_update_every
        self.vocoder_path = vocoder_path
        self.cache_dir = cache_dir
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("`model_name` must be a non-empty string.")
        if not isinstance(self.ode_method, str) or not self.ode_method.strip():
            raise ValueError("`ode_method` must be a non-empty string.")
        if not 0 < float(self.ema_decay) <= 1:
            raise ValueError("`ema_decay` must be in the interval (0, 1].")
        for name in ("ema_update_after_step", "ema_update_every"):
            value = getattr(self, name)
            minimum = 0 if name == "ema_update_after_step" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"`{name}` must be an integer greater than or equal to {minimum}.")


class F5TTSForTextToSpeech(PreTrainedTTSModel):
    """F5-TTS voice cloning without an external ``f5-tts`` package."""

    config_class = F5TTSConfig
    default_model_name_or_path = "F5TTS_v1_Base"

    def __init__(
        self,
        config: F5TTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        explicit_model_source = (
            model_path is not None or isinstance(config, (str, Path)) or
            (isinstance(config, F5TTSConfig) and bool(config.name_or_path)))
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        source = Path(config.name_or_path).expanduser()
        if source.is_file():
            configured_checkpoint = str(config.checkpoint_path).strip()
            if configured_checkpoint:
                existing_checkpoint = Path(configured_checkpoint).expanduser()
                if existing_checkpoint.resolve() != source.resolve():
                    raise ValueError(
                        "The direct F5-TTS checkpoint and "
                        "`checkpoint_path` refer to different files.")
            config.checkpoint_path = str(source.resolve())
        elif explicit_model_source:
            config.model_name = config.name_or_path
        else:
            config.name_or_path = config.model_name
        config.validate()
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        api = import_optional(
            "voicehub.models.f5tts.source.f5_tts.api",
            model_type="f5tts",
            install_extra="f5tts",
        )
        self.model = api.F5TTS(
            model=self.config.model_name,
            ckpt_file=self.config.checkpoint_path,
            vocab_file=self.config.vocabulary_path,
            ode_method=self.config.ode_method,
            use_ema=self.config.use_ema,
            vocoder_local_path=self.config.vocoder_path,
            device=self.device,
            hf_cache_dir=self.config.cache_dir,
        )
        if not callable(getattr(self.model, "infer", None)):
            raise TypeError("The loaded F5-TTS runtime does not implement infer().")
        sample_rate = int(getattr(self.model, "target_sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("The loaded F5-TTS runtime reported an invalid sample rate.")
        self.config.sample_rate = sample_rate

    def _set_training_device(self, device: str) -> None:
        """Synchronize source runtime state after Trainer moves the model."""
        super()._set_training_device(device)
        if self.model is None:
            return
        self.model.device = str(device)
        vocoder = getattr(self.model, "vocoder", None)
        move = getattr(vocoder, "to", None)
        if callable(move):
            moved = move(device)
            if moved is not None:
                self.model.vocoder = moved

    def _prepare_for_inference(self) -> None:
        """Restore serving mode on the modules owned by the plain API shell."""
        if self.model is None:
            return
        for component_name in ("ema_model", "vocoder"):
            component = getattr(self.model, component_name, None)
            evaluate = getattr(component, "eval", None)
            if callable(evaluate):
                evaluate()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker_audio = model_inputs.get("speaker_audio_path")
        if not isinstance(speaker_audio, (str, Path)) or not str(speaker_audio).strip():
            raise ValueError("`speaker_audio_path` must be a non-empty local path.")
        reference_path = Path(speaker_audio).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"F5-TTS reference audio was not found: {reference_path}.")
        reference_text = model_inputs.get("reference_text", "")
        if not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string.")

        numeric_values = {
            "speed": model_inputs.get("speed", 1.0),
            "cfg_strength": model_inputs.get("cfg_strength", 2.0),
            "sway_sampling_coef": model_inputs.get(
                "sway_sampling_coef",
                -1.0,
            ),
            "cross_fade_duration": model_inputs.get("cross_fade_duration", 0.15),
        }
        for name, value in numeric_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"`{name}` must be finite.")
        if numeric_values["speed"] <= 0:
            raise ValueError("`speed` must be greater than zero.")
        if numeric_values["cfg_strength"] < 0:
            raise ValueError("`cfg_strength` must be non-negative.")
        if numeric_values["cross_fade_duration"] < 0:
            raise ValueError("`cross_fade_duration` must be non-negative.")

        nfe_steps = model_inputs.get("nfe_steps", 32)
        if isinstance(nfe_steps, bool) or not isinstance(nfe_steps, int) or nfe_steps <= 0:
            raise ValueError("`nfe_steps` must be a positive integer.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("`seed` must be an integer or None.")
        if not isinstance(model_inputs.get("remove_silence", False), bool):
            raise TypeError("`remove_silence` must be a boolean.")

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str,
        reference_text: str = "",
        output_file: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
        nfe_steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        cross_fade_duration: float = 0.15,
        remove_silence: bool = False,
    ) -> TTSOutput:
        if self.model is None:
            raise RuntimeError("F5-TTS must be loaded before generation.")
        waveform, sample_rate, spectrogram = self.model.infer(
            ref_file=str(Path(speaker_audio_path).expanduser()),
            ref_text=reference_text,
            gen_text=text,
            file_wave=None,
            speed=speed,
            seed=seed,
            nfe_step=nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            cross_fade_duration=cross_fade_duration,
            remove_silence=remove_silence,
        )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("F5-TTS inference returned an invalid sample rate.")
        self.config.sample_rate = sample_rate
        output = finish_audio_output(
            waveform,
            sample_rate,
            output_file=output_file,
            metadata={
                "seed": getattr(self.model, "seed", seed),
                "spectrogram": spectrogram,
            },
        )
        return output


F5TTS = F5TTSForTextToSpeech
