"""Sesame CSM inference backed by vendored CSM and Moshi source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output


class CSMConfig(VoiceHubConfig):
    """Configuration for conversational Sesame CSM checkpoints."""

    model_type = "csm"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype


class CSMForTextToSpeech(PreTrainedTTSModel):
    """Conversational speech generation with optional speaker context."""

    config_class = CSMConfig
    default_model_name_or_path = "sesame/csm-1b"

    def __init__(
        self,
        config: CSMConfig | str | None = None,
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
        self._runtime = None
        self._torch = None
        self._torchaudio = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="csm",
            install_extra="csm",
        )
        torchaudio = import_optional(
            "torchaudio",
            model_type="csm",
            install_extra="csm",
        )
        runtime = import_optional(
            "voicehub.models.csm.source.csm.generator",
            model_type="csm",
            install_extra="csm",
        )
        models = import_optional(
            "voicehub.models.csm.source.csm.models",
            model_type="csm",
            install_extra="csm",
        )
        model = models.Model.from_pretrained(self.config.name_or_path)
        dtype = getattr(torch, self.config.torch_dtype)
        if self.device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
            dtype = torch.float32
        model.to(device=self.device, dtype=dtype)
        self.model = runtime.Generator(model)
        self.config.sample_rate = int(self.model.sample_rate)
        self._runtime = runtime
        self._torch = torch
        self._torchaudio = torchaudio

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: int = 0,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        max_audio_length_ms: float = 90000,
        temperature: float = 0.9,
        top_k: int = 50,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        context = []
        if speaker_audio_path:
            if not reference_text:
                raise ValueError("CSM speaker context requires reference_text.")
            audio, sample_rate = self._torchaudio.load(speaker_audio_path)
            audio = audio.mean(dim=0)
            if sample_rate != self.sample_rate:
                audio = self._torchaudio.functional.resample(
                    audio,
                    sample_rate,
                    self.sample_rate,
                )
            context.append(self._runtime.Segment(
                speaker=speaker,
                text=reference_text,
                audio=audio,
            ))
        audio = self.model.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=top_k,
            **generation_options,
        )
        return finish_audio_output(
            audio.detach().float().cpu(),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "speaker": speaker,
                "context_segments": len(context)
            },
        )


CSMTTS = CSMForTextToSpeech
