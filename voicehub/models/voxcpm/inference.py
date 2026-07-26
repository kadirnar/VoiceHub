"""VoxCPM and VoxCPM2 integration backed by vendored source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output


class VoxCPMConfig(VoiceHubConfig):
    """Configuration for VoxCPM generations and optional denoising."""

    model_type = "voxcpm"

    def __init__(
        self,
        *,
        load_denoiser: bool = False,
        optimize: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.load_denoiser = load_denoiser
        self.optimize = optimize


class VoxCPMForTextToSpeech(PreTrainedTTSModel):
    """Multilingual synthesis, voice design, and controllable cloning."""

    config_class = VoxCPMConfig
    default_model_name_or_path = "openbmb/VoxCPM2"

    def __init__(
        self,
        config: VoxCPMConfig | str | None = None,
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
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            "voicehub.models.voxcpm.source.voxcpm",
            model_type="voxcpm",
            install_extra="voxcpm",
        )
        self.model = runtime.VoxCPM.from_pretrained(
            self.config.name_or_path,
            load_denoiser=(self.config.load_denoiser and not self.is_training_load),
            optimize=(self.config.optimize and not self.is_training_load),
            training=self.is_training_load,
            device=self.device,
        )
        self.config.sample_rate = int(self.model.tts_model.sample_rate)
        self._loaded_for_training = self.is_training_load

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            return
        self.model = None
        self._loading_for_training = True
        try:
            self.load()
        finally:
            self._loading_for_training = False

    def _set_training_device(self, device: str) -> None:
        """Keep the source forward's explicit device routing synchronized."""
        super()._set_training_device(device)
        runtime = getattr(self.model, "tts_model", None)
        if runtime is None:
            return
        runtime.device = str(device)
        config = getattr(runtime, "config", None)
        if config is not None and hasattr(config, "device"):
            config.device = str(device)

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        prompt_audio_path: str | None = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        options = {
            "text": text,
            "reference_wav_path": speaker_audio_path,
            "prompt_text": reference_text,
            "prompt_wav_path": prompt_audio_path,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "seed": seed,
            **generation_options,
        }
        audio = self.model.generate(**{key: value for key, value in options.items() if value is not None})
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={"seed": seed},
        )


VoxCPMTTS = VoxCPMForTextToSpeech
