"""Sesame CSM inference backed by vendored CSM and Moshi source."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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
        self._training_backend = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        if self.is_training_load:
            from voicehub.models.csm.training import load_csm_training_backend

            backend = load_csm_training_backend(
                self.config.name_or_path,
                device=self.device,
                torch_dtype=self.config.torch_dtype,
            )
            self.model = backend.model
            self._training_backend = backend
            self.config.sample_rate = backend.sample_rate
            return

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
        self._training_backend = None
        self.config.sample_rate = int(self.model.sample_rate)
        self._runtime = runtime
        self._torch = torch
        self._torchaudio = torchaudio

    @property
    def training_backend(self):
        """Loaded official Transformers backend, if training was requested."""
        return self._training_backend

    def _prepare_for_training(self) -> None:
        if (self._training_backend is not None and self.model is self._training_backend.model):
            return
        self.model = None
        self._loading_for_training = True
        try:
            self.load()
        finally:
            self._loading_for_training = False

    def prepare_training_inputs(
        self,
        inputs: dict,
        *,
        phase: str,
    ) -> dict:
        """Prepare native CSM audio-frame labels through CsmProcessor."""
        del phase
        if self._training_backend is None:
            raise RuntimeError("CSM training inputs require load_for_training() before "
                               "preparation.")
        return self._training_backend.prepare_inputs(inputs)

    @property
    def _uses_transformers_backend(self) -> bool:
        return (self._training_backend is not None and self.model is self._training_backend.model)

    @staticmethod
    def _move_processor_output_to_device(
        inputs: Any,
        device: str,
    ) -> Mapping[str, Any]:
        move = getattr(inputs, "to", None)
        if callable(move):
            return move(device)
        if not isinstance(inputs, Mapping):
            raise TypeError("CsmProcessor.apply_chat_template() must return a mapping.")
        return {
            name: (value.to(device) if callable(getattr(value, "to", None)) else value)
            for name, value in inputs.items()
        }

    def _load_transformers_reference_audio(
        self,
        speaker_audio_path: str,
    ):
        torchaudio = import_optional(
            "torchaudio",
            model_type="csm",
            install_extra="csm",
        )
        audio, sample_rate = torchaudio.load(speaker_audio_path)
        audio = audio.mean(dim=0)
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                sample_rate,
                self.sample_rate,
            )
        return audio

    def _generate_transformers(
        self,
        text: str,
        *,
        speaker: int,
        speaker_audio_path: str | None,
        reference_text: str | None,
        max_audio_length_ms: float,
        temperature: float,
        top_k: int,
        generation_options: Mapping[str, Any],
    ) -> tuple[Any, int]:
        """Generate through a restored ``CsmForConditionalGeneration``."""
        if max_audio_length_ms <= 0:
            raise ValueError("max_audio_length_ms must be greater than zero.")
        processor = self._training_backend.processor
        conversation = []
        if speaker_audio_path:
            if not reference_text:
                raise ValueError("CSM speaker context requires reference_text.")
            reference_audio = self._load_transformers_reference_audio(speaker_audio_path, )
            conversation.append({
                "role":
                str(speaker),
                "content": [
                    {
                        "type": "text",
                        "text": reference_text,
                    },
                    {
                        "type": "audio",
                        "path": reference_audio,
                    },
                ],
            })
        conversation.append({
            "role": str(speaker),
            "content": [{
                "type": "text",
                "text": text,
            }],
        })
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_processor_output_to_device(
            inputs,
            self.device,
        )

        options = dict(generation_options)
        options.setdefault(
            "max_new_tokens",
            max(1, int(max_audio_length_ms / 80)),
        )
        options.setdefault("do_sample", True)
        options.setdefault("temperature", temperature)
        options.setdefault("top_k", top_k)
        options.setdefault("depth_decoder_do_sample", True)
        options.setdefault("depth_decoder_temperature", temperature)
        options.setdefault("depth_decoder_top_k", top_k)
        options.setdefault("use_cache", True)
        requested_audio = options.pop("output_audio", True)
        if requested_audio is not True:
            raise ValueError("CSM text-to-speech generation requires output_audio=True.")

        if hasattr(self.model, "eval"):
            self.model.eval()
        generated = self.model.generate(
            **inputs,
            output_audio=True,
            **options,
        )
        audio = getattr(generated, "audio", generated)
        if isinstance(audio, (list, tuple)):
            if not audio:
                raise RuntimeError("CsmForConditionalGeneration returned no audio.")
            audio = audio[0]
        if audio is None:
            raise RuntimeError("CsmForConditionalGeneration returned no audio.")
        return audio, len(conversation) - 1

    def _generate_vendored(
        self,
        text: str,
        *,
        speaker: int,
        speaker_audio_path: str | None,
        reference_text: str | None,
        max_audio_length_ms: float,
        temperature: float,
        top_k: int,
        generation_options: Mapping[str, Any],
    ) -> tuple[Any, int]:
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
        return audio, len(context)

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
        if self._uses_transformers_backend:
            audio, context_segments = self._generate_transformers(
                text,
                speaker=speaker,
                speaker_audio_path=speaker_audio_path,
                reference_text=reference_text,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                top_k=top_k,
                generation_options=generation_options,
            )
            backend_name = "transformers"
        else:
            audio, context_segments = self._generate_vendored(
                text,
                speaker=speaker,
                speaker_audio_path=speaker_audio_path,
                reference_text=reference_text,
                max_audio_length_ms=max_audio_length_ms,
                temperature=temperature,
                top_k=top_k,
                generation_options=generation_options,
            )
            backend_name = "vendored"
        if callable(getattr(audio, "detach", None)):
            audio = audio.detach().float().cpu()
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "speaker": speaker,
                "context_segments": context_segments,
                "backend": backend_name,
            },
        )


CSMTTS = CSMForTextToSpeech
