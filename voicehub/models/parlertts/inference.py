"""Parler-TTS integration backed by vendored model source."""

from __future__ import annotations

from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference

DEFAULT_DESCRIPTION = (
    "A clear, expressive speaker delivers high-quality speech at a moderate "
    "speed and pitch in a close, noise-free recording.")


class ParlerTTSConfig(VoiceHubConfig):
    """VoiceHub loading configuration for Parler-TTS."""

    model_type = "parlertts"

    def __init__(
        self,
        *,
        attention_implementation: str | None = "sdpa",
        compile_model: bool = False,
        torch_dtype: str | None = None,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.attention_implementation = attention_implementation
        self.compile_model = compile_model
        self.torch_dtype = torch_dtype


class ParlerTTSForTextToSpeech(PreTrainedTTSModel):
    """Prompt-controlled TTS without the external ``parler-tts`` package."""

    config_class = ParlerTTSConfig
    default_model_name_or_path = "parler-tts/parler-tts-mini-v1"

    def __init__(
        self,
        config: ParlerTTSConfig | str | None = None,
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
        self.tokenizer = None
        self._torch = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="parlertts",
            install_extra="parlertts",
        )
        source = import_optional(
            "voicehub.models.parlertts.source.parler_tts",
            model_type="parlertts",
            install_extra="parlertts",
        )
        transformers = import_optional(
            "transformers",
            model_type="parlertts",
            install_extra="parlertts",
        )
        model_options = {}
        if self.config.attention_implementation:
            model_options["attn_implementation"] = self.config.attention_implementation
        if self.config.torch_dtype:
            model_options["torch_dtype"] = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )

        model = source.ParlerTTSForConditionalGeneration.from_pretrained(
            self.config.name_or_path,
            **model_options,
        ).to(self.device)
        should_compile = self.config.compile_model and not self.is_training_load
        self.model = torch.compile(model) if should_compile else model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.name_or_path)
        self._torch = torch
        self.config.sample_rate = self.model.config.sampling_rate
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

    def _prepare_for_inference(self) -> None:
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        description = model_inputs.get("description", DEFAULT_DESCRIPTION)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("`description` must be a non-empty voice description.")

    def _tokenize(self, text: str):
        encoded = self.tokenizer(text, return_tensors="pt")
        input_ids = getattr(encoded, "input_ids", None)
        if input_ids is None:
            raise RuntimeError("The Parler-TTS tokenizer returned no `input_ids`.")
        return input_ids.to(self.device)

    @staticmethod
    def _extract_waveform(generation):
        audio = getattr(generation, "audio_values", None)
        if audio is None:
            audio = getattr(generation, "sequences", generation)
        if audio is None or not hasattr(audio, "detach"):
            raise RuntimeError("Parler-TTS returned no tensor audio waveform.")
        if hasattr(audio, "numel") and audio.numel() == 0:
            raise RuntimeError("Parler-TTS returned an empty audio waveform.")
        return audio.detach().float().cpu().numpy().squeeze()

    def _generate(
        self,
        text: str,
        *,
        description: str = DEFAULT_DESCRIPTION,
        output_file: str | None = None,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        input_ids = self._tokenize(description)
        prompt_input_ids = self._tokenize(text)

        with seeded_inference(
                seed,
                device=self.device,
                model_type="parlertts",
        ) as effective_seed:
            with self._torch.inference_mode():
                generation = self.model.generate(
                    input_ids=input_ids,
                    prompt_input_ids=prompt_input_ids,
                    **generation_options,
                )
        return finish_audio_output(
            self._extract_waveform(generation),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "description": description,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


ParlerVoiceHubConfig = ParlerTTSConfig
ParlerTTS = ParlerTTSForTextToSpeech
