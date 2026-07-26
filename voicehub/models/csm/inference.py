"""Sesame CSM inference backed by vendored CSM and Moshi source."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


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
    _AUDIO_FRAME_MILLISECONDS = 80
    _AUDIO_VOCAB_SIZE = 2_051

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

    def _load_training_runtime(self) -> None:
        from voicehub.models.csm.training import load_csm_training_backend

        backend = load_csm_training_backend(
            self.config.name_or_path,
            device=self.device,
            torch_dtype=self.config.torch_dtype,
        )
        self.model = backend.model
        self._training_backend = backend
        self._runtime = None
        self._torch = None
        self._torchaudio = None
        self.config.sample_rate = backend.sample_rate

    def _load_vendored_runtime(self) -> None:
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
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        model.to(device=self.device, dtype=dtype)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        self.model = runtime.Generator(model)
        self._training_backend = None
        self.config.sample_rate = int(self.model.sample_rate)
        self._runtime = runtime
        self._torch = torch
        self._torchaudio = torchaudio

    def _load_pretrained_model(self) -> None:
        if self.is_training_load:
            self._load_training_runtime()
        else:
            self._load_vendored_runtime()

    def _prepare_for_inference(self) -> None:
        """Evaluate both public backends and vendored serving components."""
        candidates = (
            self.model,
            getattr(self.model, "_model", None),
            getattr(self.model, "_audio_tokenizer", None),
            getattr(self.model, "_watermarker", None),
        )
        for component in candidates:
            evaluate = getattr(component, "eval", None)
            if callable(evaluate):
                evaluate()
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty local path or None.")
        if reference_text is not None and not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string or None.")
        if (speaker_audio_path is not None) != bool(isinstance(reference_text, str) and
                                                    reference_text.strip()):
            raise ValueError(
                "CSM speaker context requires `speaker_audio_path` and a "
                "non-empty `reference_text` together.")
        if speaker_audio_path is not None:
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"CSM reference audio was not found: {reference_path}.")
        speaker = model_inputs.get("speaker", 0)
        if (isinstance(speaker, bool) or not isinstance(speaker, int) or speaker < 0):
            raise ValueError("`speaker` must be a non-negative integer.")
        max_audio_length_ms = model_inputs.get(
            "max_audio_length_ms",
            90_000,
        )
        if (isinstance(max_audio_length_ms, bool) or not isinstance(max_audio_length_ms, (int, float)) or
                not math.isfinite(max_audio_length_ms) or max_audio_length_ms <= 0):
            raise ValueError("`max_audio_length_ms` must be finite and greater than zero.")
        if max_audio_length_ms < self._AUDIO_FRAME_MILLISECONDS:
            raise ValueError(
                "`max_audio_length_ms` must be finite and at least "
                f"{self._AUDIO_FRAME_MILLISECONDS}.")
        temperature = model_inputs.get("temperature", 0.9)
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("`temperature` must be a finite non-negative number.")
        top_k = model_inputs.get("top_k", 50)
        minimum_top_k = 0 if temperature == 0 else 1
        if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < minimum_top_k):
            qualifier = "non-negative" if temperature == 0 else "positive"
            raise ValueError(f"`top_k` must be a {qualifier} integer for this sampling mode.")
        if top_k > self._AUDIO_VOCAB_SIZE:
            raise ValueError(
                "`top_k` cannot exceed the CSM audio vocabulary size "
                f"({self._AUDIO_VOCAB_SIZE}).")
        if model_inputs.get("output_audio", True) is not True:
            raise ValueError("CSM text-to-speech generation requires "
                             "`output_audio=True`.")

    @property
    def training_backend(self):
        """Loaded official Transformers backend, if training was requested."""
        return self._training_backend

    def _prepare_for_training(self) -> None:
        if (self._training_backend is not None and self.model is self._training_backend.model):
            return
        self.model = None
        self._runtime = None
        self._torch = None
        self._torchaudio = None
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
            moved = move(device)
            if not isinstance(moved, Mapping):
                raise TypeError("CsmProcessor batch `.to()` must return a mapping.")
            return moved
        if not isinstance(inputs, Mapping):
            raise TypeError("CsmProcessor.apply_chat_template() must return a mapping.")
        return {
            name: (value.to(device) if callable(getattr(value, "to", None)) else value)
            for name, value in inputs.items()
        }

    def _load_reference_audio(
        self,
        speaker_audio_path: str,
        *,
        torchaudio=None,
    ):
        if torchaudio is None:
            torchaudio = import_optional(
                "torchaudio",
                model_type="csm",
                install_extra="csm",
            )
        audio, sample_rate = torchaudio.load(str(speaker_audio_path))
        if audio.numel() == 0:
            raise ValueError("CSM reference audio contains no samples.")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"CSM reference audio has an invalid sample rate: {sample_rate!r}.")
        audio = audio.mean(dim=0)
        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                sample_rate,
                self.sample_rate,
            )
        return audio

    def _build_transformers_conversation(
        self,
        text: str,
        *,
        speaker: int,
        speaker_audio_path: str | None,
        reference_text: str | None,
    ) -> list[dict[str, Any]]:
        conversation = []
        if speaker_audio_path:
            reference_audio = self._load_reference_audio(speaker_audio_path, )
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
        return conversation

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
        processor = self._training_backend.processor
        conversation = self._build_transformers_conversation(
            text,
            speaker=speaker,
            speaker_audio_path=speaker_audio_path,
            reference_text=reference_text,
        )
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
        if temperature == 0:
            # Transformers sampling divides logits by temperature. Use its
            # native deterministic path for both token-generation levels.
            options["do_sample"] = False
            options["depth_decoder_do_sample"] = False
            for sampling_option in (
                    "temperature",
                    "top_k",
                    "depth_decoder_temperature",
                    "depth_decoder_top_k",
            ):
                options.pop(sampling_option, None)
        else:
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
        audio = self._extract_transformers_audio(generated)
        return audio, len(conversation) - 1

    @staticmethod
    def _extract_transformers_audio(generated):
        audio = getattr(generated, "audio", generated)
        if isinstance(audio, (list, tuple)):
            if not audio:
                raise RuntimeError("CsmForConditionalGeneration returned no audio.")
            audio = audio[0]
        if audio is None:
            raise RuntimeError("CsmForConditionalGeneration returned no audio.")
        return audio

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
            audio = self._load_reference_audio(
                speaker_audio_path,
                torchaudio=self._torchaudio,
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
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        generation_options.pop("output_audio", None)
        with seeded_inference(
                seed,
                device=self.device,
                model_type="csm",
        ) as effective_seed:
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
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


CSMTTS = CSMForTextToSpeech
