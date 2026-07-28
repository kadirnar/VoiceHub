"""Higgs Audio TTS integration backed by vendored BosonAI source."""

from __future__ import annotations

import math
from numbers import Real

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


class HiggsTTSConfig(VoiceHubConfig):
    """Configuration for Higgs Audio v2/v2.5 generation."""

    model_type = "higgstts"

    def __init__(
        self,
        *,
        audio_tokenizer_name_or_path: str = "bosonai/higgs-audio-v2-tokenizer",
        torch_dtype: str = "bfloat16",
        training_text_loss_weight: float = 1.0,
        training_audio_loss_weight: float = 1.0,
        system_prompt: str = (
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n"
            "<|scene_desc_end|>"),
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.audio_tokenizer_name_or_path = audio_tokenizer_name_or_path
        self.torch_dtype = torch_dtype
        self.training_text_loss_weight = training_text_loss_weight
        self.training_audio_loss_weight = training_audio_loss_weight
        self.system_prompt = system_prompt


class HiggsTTSForTextToSpeech(PreTrainedTTSModel):
    """Expressive Higgs Audio generation through the local serve engine."""

    config_class = HiggsTTSConfig
    default_model_name_or_path = "bosonai/higgs-audio-v2-generation-3B-base"
    passthrough_generation_options = frozenset({
        "ras_win_len",
        "ras_win_max_num_repeat",
        "stop_strings",
    })

    def __init__(
        self,
        config: HiggsTTSConfig | str | None = None,
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
        self._types = None
        self._training_backend = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _load_data_types():
        return import_optional(
            "voicehub.models.higgstts.source.boson_multimodal.data_types",
            model_type="higgstts",
            install_extra=None,
        )

    def _load_training_runtime(self) -> None:
        from voicehub.models.higgstts.training import load_higgs_training_backend

        backend = load_higgs_training_backend(
            self.config.name_or_path,
            self.config.audio_tokenizer_name_or_path,
            device=self.device,
            torch_dtype=self.config.torch_dtype,
        )
        self.model = backend
        self._training_backend = backend
        self._types = None
        self.config.sample_rate = backend.sample_rate

    def _load_serving_runtime(self) -> None:
        torch = import_optional(
            "torch",
            model_type="higgstts",
            install_extra=None,
        )
        runtime = import_optional(
            "voicehub.models.higgstts.source.boson_multimodal.serve."
            "serve_engine",
            model_type="higgstts",
            install_extra=None,
        )
        self._types = self._load_data_types()
        self.model = runtime.HiggsAudioServeEngine(
            self.config.name_or_path,
            self.config.audio_tokenizer_name_or_path,
            device=self.device,
            torch_dtype=resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            ),
        )
        self._training_backend = None
        self.config.sample_rate = int(self.model.audio_tokenizer.sampling_rate)

    def _load_pretrained_model(self) -> None:
        if self.is_training_load:
            self._load_training_runtime()
        else:
            self._load_serving_runtime()

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        max_new_tokens = model_inputs.get("max_new_tokens", 1024)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        temperature = model_inputs.get("temperature", 0.3)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("`temperature` must be a finite non-negative number.")
        top_p = model_inputs.get("top_p", 0.95)
        valid_top_p = (0 <= top_p <= 1 if isinstance(top_p, Real) and not isinstance(top_p, bool) else False)
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or
                not valid_top_p or (temperature > 0 and top_p == 0)):
            interval = "[0, 1]" if temperature == 0 else "(0, 1]"
            raise ValueError(f"`top_p` must be in the interval {interval}.")
        top_k = model_inputs.get("top_k", 50)
        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("`top_k` must be a positive integer or None.")
        system_prompt = model_inputs.get("system_prompt")
        if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
            raise ValueError("`system_prompt` must be a non-empty string or None.")
        if not isinstance(model_inputs.get("force_audio_gen", True), bool):
            raise TypeError("`force_audio_gen` must be a boolean.")
        stop_strings = model_inputs.get("stop_strings")
        if stop_strings is not None and (not isinstance(stop_strings, (list, tuple)) or
                                         any(not isinstance(value, str) or not value
                                             for value in stop_strings)):
            raise TypeError("`stop_strings` must be a sequence of non-empty strings or "
                            "None.")
        ras_win_len = model_inputs.get("ras_win_len", 7)
        if ras_win_len is not None and (isinstance(ras_win_len, bool) or not isinstance(ras_win_len, int)):
            raise TypeError("`ras_win_len` must be an integer or None.")
        ras_repeats = model_inputs.get("ras_win_max_num_repeat", 2)
        if (isinstance(ras_repeats, bool) or not isinstance(ras_repeats, int) or ras_repeats <= 0):
            raise ValueError("`ras_win_max_num_repeat` must be a positive integer.")

    @property
    def training_backend(self):
        """Return the cache-free runtime after a training load."""
        if (self._training_backend is not None and self.model is self._training_backend):
            return self._training_backend
        return None

    def _prepare_for_training(self) -> None:
        if self._training_backend is not None:
            # A Trainer artifact may have been converted to a serving shell
            # for generation. Reattach its same trained model and discard the
            # fresh inference caches instead of reloading the base checkpoint.
            self.model = self._training_backend
            self._types = None
            self._training_backend.prepare_for_training()
            return

        # Drop the serving engine as a unit. This releases its StaticCache
        # buckets before the trainable model is loaded.
        self.model = None
        self._types = None
        self._training_backend = None
        self._loading_for_training = True
        try:
            self.load()
        finally:
            self._loading_for_training = False

    def _ensure_serving_runtime(self) -> None:
        if self.training_backend is None:
            return
        self.model = self._training_backend.build_inference_runtime(device=self.device, )
        self._types = self._load_data_types()

    def _build_chat_sample(
        self,
        text: str,
        *,
        system_prompt: str | None,
    ):
        return self._types.ChatMLSample(
            messages=[
                self._types.Message(
                    role="system",
                    content=system_prompt or self.config.system_prompt,
                ),
                self._types.Message(role="user", content=text),
            ])

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        seed: int | None = None,
        force_audio_gen: bool = True,
        **generation_options,
    ) -> TTSOutput:
        with seeded_inference(
                seed,
                device=self.device,
                model_type="higgstts",
        ) as effective_seed:
            self._ensure_serving_runtime()
            sample = self._build_chat_sample(
                text,
                system_prompt=system_prompt,
            )
            response = self.model.generate(
                chat_ml_sample=sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=effective_seed,
                force_audio_gen=force_audio_gen,
                **generation_options,
            )
        if response.audio is None:
            raise RuntimeError("Higgs Audio returned text but did not generate audio.")
        sample_rate = int(response.sampling_rate)
        if sample_rate <= 0:
            raise RuntimeError(
                "Higgs Audio returned an invalid sampling rate: "
                f"{response.sampling_rate!r}.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            response.audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "generated_text": response.generated_text,
                "usage": response.usage,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


HiggsTTS = HiggsTTSForTextToSpeech
