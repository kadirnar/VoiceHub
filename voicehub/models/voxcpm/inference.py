"""VoxCPM and VoxCPM2 integration backed by vendored source."""

from __future__ import annotations

import math
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, validate_local_file


class VoxCPMConfig(VoiceHubConfig):
    """Configuration for VoxCPM generations and optional denoising."""

    model_type = "voxcpm"

    def __init__(
        self,
        *,
        load_denoiser: bool = False,
        denoiser_name_or_path: str = ("iic/speech_zipenhancer_ans_multiloss_16k_base"),
        optimize: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.load_denoiser = load_denoiser
        self.denoiser_name_or_path = denoiser_name_or_path
        self.optimize = optimize


class VoxCPMForTextToSpeech(PreTrainedTTSModel):
    """Multilingual synthesis, voice design, and controllable cloning."""

    config_class = VoxCPMConfig
    default_model_name_or_path = "openbmb/VoxCPM2"
    passthrough_generation_options = frozenset({
        "denoise",
        "max_len",
        "min_len",
        "normalize",
        "retry_badcase",
        "retry_badcase_max_times",
        "retry_badcase_ratio_threshold",
    })

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

    def _build_runtime(self, *, training: bool):
        runtime = import_optional(
            "voicehub.models.voxcpm.source.voxcpm",
            model_type="voxcpm",
            install_extra="voxcpm",
        )
        model = runtime.VoxCPM.from_pretrained(
            self.config.name_or_path,
            load_denoiser=(self.config.load_denoiser and not training),
            zipenhancer_model_id=self.config.denoiser_name_or_path,
            optimize=(self.config.optimize and not training),
            training=training,
            device=self.device,
        )
        if not callable(getattr(model, "generate", None)):
            raise TypeError("The loaded VoxCPM runtime does not implement generate().")
        source_model = getattr(model, "tts_model", None)
        sample_rate = int(getattr(source_model, "sample_rate", 0))
        if sample_rate <= 0:
            raise ValueError("The loaded VoxCPM runtime reported an invalid sample rate.")
        return model, sample_rate

    def _load_pretrained_model(self) -> None:
        model, sample_rate = self._build_runtime(training=self.is_training_load)
        self.model = model
        self.config.sample_rate = sample_rate
        self._loaded_for_training = self.is_training_load

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            source_model = getattr(self.model, "tts_model", None)
            if source_model is not None:
                deoptimize = getattr(source_model, "deoptimize", None)
                if callable(deoptimize):
                    deoptimize()
                if hasattr(source_model, "train"):
                    source_model.train()
            return
        previous_model = self.model
        previous_training_state = self._loaded_for_training
        self._loading_for_training = True
        try:
            model, sample_rate = self._build_runtime(training=True)
            self.model = model
            self.config.sample_rate = sample_rate
            self._loaded_for_training = True
        except BaseException:
            self.model = previous_model
            self._loaded_for_training = previous_training_state
            raise
        finally:
            self._loading_for_training = False

    @staticmethod
    def _parameter_dtype_name(source_model) -> str | None:
        parameters = getattr(source_model, "parameters", None)
        if not callable(parameters):
            return None
        first_parameter = next(iter(parameters()), None)
        if first_parameter is None:
            return None
        aliases = {
            "torch.bfloat16": "bfloat16",
            "torch.float16": "float16",
            "torch.float32": "float32",
        }
        return aliases.get(str(getattr(first_parameter, "dtype", "")))

    def _prepare_for_inference(self) -> None:
        """Restore a serving-safe view of the optimizer-owned VoxCPM model."""
        source_model = getattr(self.model, "tts_model", None)
        if source_model is None:
            return

        # Mixed-precision training normally keeps parameters in FP32. Match
        # generation inputs and caches to the actual parameter dtype rather
        # than casting optimizer-owned parameters in place.
        dtype_name = self._parameter_dtype_name(source_model)
        model_config = getattr(source_model, "config", None)
        if (self._loaded_for_training and dtype_name is not None and model_config is not None):
            model_config.dtype = dtype_name
            runtime_dtype = source_model._dtype()
            max_length = int(model_config.max_length)
            for language_model_name in ("base_lm", "residual_lm"):
                language_model = getattr(
                    source_model,
                    language_model_name,
                    None,
                )
                setup_cache = getattr(language_model, "setup_cache", None)
                if callable(setup_cache):
                    setup_cache(
                        1,
                        max_length,
                        source_model.device,
                        runtime_dtype,
                    )

        if hasattr(source_model, "eval"):
            source_model.eval()
        optimize = getattr(source_model, "optimize", None)
        if self.config.optimize and callable(optimize):
            optimize()

        if (self.config.load_denoiser and getattr(self.model, "denoiser", None) is None):
            denoiser_module = import_optional(
                "voicehub.models.voxcpm.source.voxcpm.zipenhancer",
                model_type="voxcpm",
                install_extra="voxcpm",
            )
            self.model.denoiser = denoiser_module.ZipEnhancer(self.config.denoiser_name_or_path, )

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

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        prompt_audio = model_inputs.get("prompt_audio_path")
        prompt_text = model_inputs.get("reference_text")
        if prompt_text is not None and (not isinstance(prompt_text, str) or not prompt_text.strip()):
            raise ValueError("`reference_text` must be a non-empty string or None.")
        if (prompt_audio is None) != (prompt_text is None):
            raise ValueError("`prompt_audio_path` and `reference_text` must be provided together.")
        speaker_path = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
        )
        prompt_path = validate_local_file(
            prompt_audio,
            option_name="prompt_audio_path",
        )
        if speaker_path is not None:
            model_inputs["speaker_audio_path"] = str(speaker_path)
        if prompt_path is not None:
            model_inputs["prompt_audio_path"] = str(prompt_path)

        cfg_value = model_inputs.get("cfg_value", 2.0)
        if isinstance(cfg_value, bool) or not isinstance(cfg_value, (int, float)):
            raise TypeError("`cfg_value` must be a finite number.")
        if not math.isfinite(float(cfg_value)) or cfg_value < 0:
            raise ValueError("`cfg_value` must be finite and non-negative.")

        inference_timesteps = model_inputs.get("inference_timesteps", 10)
        if (isinstance(inference_timesteps, bool) or not isinstance(inference_timesteps, int) or
                inference_timesteps <= 0):
            raise ValueError("`inference_timesteps` must be a positive integer.")
        min_len = model_inputs.get("min_len", 2)
        max_len = model_inputs.get("max_len", 4096)
        for name, value in (("min_len", min_len), ("max_len", max_len)):
            if (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"`{name}` must be a positive integer.")
        if min_len > max_len:
            raise ValueError("`min_len` cannot exceed `max_len`.")
        for name, default in (
            ("normalize", False),
            ("denoise", False),
            ("retry_badcase", True),
        ):
            if not isinstance(model_inputs.get(name, default), bool):
                raise TypeError(f"`{name}` must be a boolean.")
        retry_count = model_inputs.get("retry_badcase_max_times", 3)
        if (isinstance(retry_count, bool) or not isinstance(retry_count, int) or retry_count <= 0):
            raise ValueError("`retry_badcase_max_times` must be a positive integer.")
        ratio = model_inputs.get("retry_badcase_ratio_threshold", 6.0)
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or
                not math.isfinite(float(ratio)) or ratio <= 0):
            raise ValueError("`retry_badcase_ratio_threshold` must be finite and "
                             "greater than zero.")

    @staticmethod
    def _generation_options(
        *,
        text: str,
        speaker_audio_path: str | None,
        reference_text: str | None,
        prompt_audio_path: str | None,
        cfg_value: float,
        inference_timesteps: int,
        seed: int | None,
        extra_options: dict[str, Any],
    ) -> dict[str, Any]:
        options = {
            "text": text,
            "reference_wav_path": speaker_audio_path,
            "prompt_text": reference_text,
            "prompt_wav_path": prompt_audio_path,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
            "seed": seed,
            **extra_options,
        }
        return {key: value for key, value in options.items() if value is not None}

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
        if self.model is None:
            raise RuntimeError("VoxCPM must be loaded before generation.")
        options = self._generation_options(
            text=text,
            speaker_audio_path=speaker_audio_path,
            reference_text=reference_text,
            prompt_audio_path=prompt_audio_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            seed=seed,
            extra_options=generation_options,
        )
        audio = self.model.generate(**options)
        actual_seed = getattr(
            getattr(self.model, "tts_model", None),
            "last_successful_seed",
            None,
        )
        if actual_seed is None:
            actual_seed = seed
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": actual_seed,
                "requested_seed": seed,
            },
        )


VoxCPMTTS = VoxCPMForTextToSpeech
