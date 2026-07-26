"""OmniVoice integration backed by vendored k2-fsa source."""

from __future__ import annotations

import math
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference, validate_local_file


class OmniVoiceConfig(VoiceHubConfig):
    """Configuration for multilingual OmniVoice synthesis."""

    model_type = "omnivoice"

    def __init__(
        self,
        *,
        torch_dtype: str = "float16",
        training_torch_dtype: str = "float32",
        load_asr: bool = False,
        asr_model_name: str | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.training_torch_dtype = training_torch_dtype
        self.load_asr = load_asr
        self.asr_model_name = asr_model_name


class OmniVoiceForTextToSpeech(PreTrainedTTSModel):
    """Massively multilingual cloning, design, and automatic voices."""

    config_class = OmniVoiceConfig
    default_model_name_or_path = "k2-fsa/OmniVoice"
    _SERVING_AUXILIARIES = (
        "text_tokenizer",
        "audio_tokenizer",
        "feature_extractor",
        "duration_estimator",
        "_asr_pipe",
        "_asr_model_name",
        "_asr_device",
        "sampling_rate",
    )
    _TRAINING_OMISSIONS = (
        "text_tokenizer",
        "audio_tokenizer",
        "feature_extractor",
        "duration_estimator",
        "_asr_pipe",
    )
    _GENERATION_OPTION_NAMES = frozenset({
        "num_step",
        "guidance_scale",
        "t_shift",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
        "denoise",
        "preprocess_prompt",
        "postprocess_output",
        "audio_chunk_duration",
        "audio_chunk_threshold",
        "pad_duration",
        "fade_duration",
    })

    def __init__(
        self,
        config: OmniVoiceConfig | str | None = None,
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

    def _runtime_dtype(self, torch, *, training: bool):
        dtype_name = (self.config.training_torch_dtype if training else self.config.torch_dtype)
        return resolve_torch_dtype(torch, dtype_name, self.device)

    def _build_runtime(self, *, training: bool):
        torch = import_optional(
            "torch",
            model_type="omnivoice",
            install_extra="omnivoice",
        )
        runtime = import_optional(
            "voicehub.models.omnivoice.source.omnivoice",
            model_type="omnivoice",
            install_extra="omnivoice",
        )
        model = runtime.OmniVoice.from_pretrained(
            self.config.name_or_path,
            device_map=self.device,
            dtype=self._runtime_dtype(torch, training=training),
            load_asr=self.config.load_asr,
            asr_model_name=self.config.asr_model_name,
            train=training,
        )
        sample_rate = getattr(model, "sampling_rate", None)
        if not training:
            missing = [
                name for name in ("text_tokenizer", "audio_tokenizer") if getattr(model, name, None) is None
            ]
            if missing:
                raise RuntimeError(
                    "OmniVoice inference runtime is missing required serving "
                    f"components: {', '.join(missing)}.")
            if not callable(getattr(model, "generate", None)):
                raise TypeError("The loaded OmniVoice runtime does not implement generate().")
        if sample_rate is not None:
            sample_rate = int(sample_rate)
            if sample_rate <= 0:
                raise ValueError("The loaded OmniVoice runtime reported an invalid sample rate.")
        return model, sample_rate

    def _load_pretrained_model(self) -> None:
        self.model, sample_rate = self._build_runtime(training=self.is_training_load, )
        if sample_rate is not None:
            self.config.sample_rate = sample_rate
        self._loaded_for_training = self.is_training_load

    def _remove_serving_auxiliaries(self) -> None:
        for name in self._TRAINING_OMISSIONS:
            if hasattr(self.model, name):
                setattr(self.model, name, None)

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            return
        torch = import_optional(
            "torch",
            model_type="omnivoice",
            install_extra="omnivoice",
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.training_torch_dtype,
            self.device,
        )
        self.model.to(device=self.device, dtype=dtype)
        self._remove_serving_auxiliaries()
        if hasattr(self.model, "train"):
            self.model.train()
        self._loaded_for_training = True

    def _prepare_for_inference(self) -> None:
        if not self._loaded_for_training:
            return
        # The source uses the same neural module for training and inference;
        # only tokenizers, feature extraction, duration estimation, and ASR are
        # omitted in train mode. Keep the exact optimizer-owned module and
        # borrow those serving auxiliaries from a temporary inference load.
        trained_model = self.model
        previous_mode = self._loading_for_training
        self._loading_for_training = False
        try:
            serving_model, sample_rate = self._build_runtime(training=False)
            for name in self._SERVING_AUXILIARIES:
                if hasattr(serving_model, name):
                    setattr(trained_model, name, getattr(serving_model, name))
            if sample_rate is not None:
                self.config.sample_rate = sample_rate
            self._loaded_for_training = False
            if hasattr(trained_model, "eval"):
                trained_model.eval()
        except BaseException:
            self._loaded_for_training = True
            raise
        finally:
            self._loading_for_training = previous_mode

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        speaker_path = validate_local_file(
            model_inputs.get("speaker_audio_path"),
            option_name="speaker_audio_path",
        )
        if speaker_path is not None:
            model_inputs["speaker_audio_path"] = str(speaker_path)
        for name in ("speed", "duration"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number or None.")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"`{name}` must be finite and greater than zero.")

        language = model_inputs.get("language")
        if language is not None and (not isinstance(language, str) or not language.strip()):
            raise ValueError("`language` must be a non-empty string or None.")
        if not isinstance(model_inputs.get("normalize_text", False), bool):
            raise TypeError("`normalize_text` must be a boolean.")

        for unsupported in ("top_p", "max_new_tokens"):
            if model_inputs.get(unsupported) is not None:
                raise ValueError(
                    f"OmniVoice does not support `{unsupported}`. Use its "
                    "iterative-decoding options instead.")
        if (model_inputs.get("temperature") is not None and
                model_inputs.get("class_temperature") is not None):
            raise ValueError("Pass either `temperature` or `class_temperature`, not both.")

        wrapper_options = {
            "text",
            "output_file",
            "language",
            "speaker_audio_path",
            "reference_text",
            "instruct",
            "speed",
            "duration",
            "normalize_text",
            "seed",
            "temperature",
            "top_p",
            "max_new_tokens",
        }
        unknown = sorted(set(model_inputs) - wrapper_options - self._GENERATION_OPTION_NAMES)
        if unknown:
            raise ValueError("Unsupported OmniVoice generation option(s): "
                             f"{', '.join(unknown)}.")

        num_step = model_inputs.get("num_step", 32)
        if isinstance(num_step, bool) or not isinstance(num_step, int) or num_step <= 0:
            raise ValueError("`num_step` must be a positive integer.")
        for name in (
                "guidance_scale",
                "t_shift",
                "layer_penalty_factor",
                "position_temperature",
                "class_temperature",
                "audio_chunk_duration",
                "audio_chunk_threshold",
                "pad_duration",
                "fade_duration",
        ):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
        for name in ("denoise", "preprocess_prompt", "postprocess_output"):
            value = model_inputs.get(name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")

    @staticmethod
    def _resolve_language(language: str | None) -> str | None:
        if language is None or language.strip().lower() == "none":
            return None
        from voicehub.models.omnivoice.source.omnivoice.utils.lang_map import LANG_IDS, LANG_NAME_TO_ID

        normalized = language.strip().lower()
        if normalized in LANG_IDS:
            return normalized
        if normalized in LANG_NAME_TO_ID:
            return LANG_NAME_TO_ID[normalized]
        raise ValueError(
            f"Unsupported OmniVoice language {language!r}. Pass a supported "
            "language ID/name or None for language-agnostic generation.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        language: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        instruct: str | None = None,
        speed: float | None = None,
        duration: float | None = None,
        normalize_text: bool = False,
        seed: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        if self.model is None:
            raise RuntimeError("OmniVoice must be loaded before generation.")
        del top_p, max_new_tokens
        resolved_language = self._resolve_language(language)
        if temperature is not None:
            generation_options["class_temperature"] = temperature
        with seeded_inference(
                seed,
                device=self.device,
                model_type="omnivoice",
        ) as effective_seed:
            audios = self.model.generate(
                text=text,
                language=resolved_language,
                ref_audio=speaker_audio_path,
                ref_text=reference_text,
                instruct=instruct,
                speed=speed,
                duration=duration,
                normalize_text=normalize_text,
                **generation_options,
            )
        if not isinstance(audios, (list, tuple)) or not audios:
            raise RuntimeError("OmniVoice generation returned no audio waveforms.")
        return finish_audio_output(
            audios[0],
            self.sample_rate,
            output_file=output_file,
            metadata={
                "language": resolved_language,
                "requested_language": language,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


OmniVoiceTTS = OmniVoiceForTextToSpeech
