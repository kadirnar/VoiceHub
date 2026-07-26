"""Higgs Audio TTS integration backed by vendored BosonAI source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype


class HiggsTTSConfig(VoiceHubConfig):
    """Configuration for Higgs Audio v2/v2.5 generation."""

    model_type = "higgstts"

    def __init__(
        self,
        *,
        audio_tokenizer_name_or_path: str = "bosonai/higgs-audio-v2-tokenizer",
        torch_dtype: str = "bfloat16",
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
        self.system_prompt = system_prompt


class HiggsTTSForTextToSpeech(PreTrainedTTSModel):
    """Expressive Higgs Audio generation through the local serve engine."""

    config_class = HiggsTTSConfig
    default_model_name_or_path = "bosonai/higgs-audio-v2-generation-3B-base"

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
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="higgstts",
            install_extra="higgstts",
        )
        runtime = import_optional(
            "voicehub.models.higgstts.source.boson_multimodal.serve."
            "serve_engine",
            model_type="higgstts",
            install_extra="higgstts",
        )
        self._types = import_optional(
            "voicehub.models.higgstts.source.boson_multimodal.data_types",
            model_type="higgstts",
            install_extra="higgstts",
        )
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
        self.config.sample_rate = int(self.model.audio_tokenizer.sampling_rate)

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
        force_audio_gen: bool = False,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        sample = self._types.ChatMLSample(
            messages=[
                self._types.Message(
                    role="system",
                    content=system_prompt or self.config.system_prompt,
                ),
                self._types.Message(role="user", content=text),
            ])
        response = self.model.generate(
            chat_ml_sample=sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            force_audio_gen=force_audio_gen,
            **generation_options,
        )
        if response.audio is None:
            raise RuntimeError("Higgs Audio returned text but did not generate audio.")
        self.config.sample_rate = int(response.sampling_rate)
        return finish_audio_output(
            response.audio,
            response.sampling_rate,
            output_file=output_file,
            metadata={
                "generated_text": response.generated_text,
                "usage": response.usage,
            },
        )


HiggsTTS = HiggsTTSForTextToSpeech
