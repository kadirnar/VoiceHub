"""LLaSA integration using Transformers and vendored XCodec2 source."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


class LlasaConfig(VoiceHubConfig):
    """Configuration for LLaSA language model and XCodec2 decoder."""

    model_type = "llasa"

    def __init__(
        self,
        *,
        codec_name_or_path: str = "HKUSTAudio/xcodec2",
        torch_dtype: str = "float32",
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 1.0,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p


class LlasaForTextToSpeech(PreTrainedTTSModel):
    """Multilingual LLaSA synthesis and voice cloning without ``xcodec2``."""

    config_class = LlasaConfig
    default_model_name_or_path = "HKUSTAudio/Llasa-1B-Multilingual"
    _TEXT_START = "<|TEXT_UNDERSTANDING_START|>"
    _TEXT_END = "<|TEXT_UNDERSTANDING_END|>"
    _SPEECH_START = "<|SPEECH_GENERATION_START|>"
    _SPEECH_END = "<|SPEECH_GENERATION_END|>"
    _CODEC_TOKENS_PER_SECOND = 50

    def __init__(
        self,
        config: LlasaConfig | str | None = None,
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
        self.codec = None
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="llasa",
            install_extra=None,
        )
        transformers = import_optional(
            "transformers",
            model_type="llasa",
            install_extra=None,
        )
        codec_module = import_optional(
            "voicehub.models.llasa.source.xcodec2.modeling_xcodec2",
            model_type="llasa",
            install_extra=None,
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.name_or_path)
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                self.config.name_or_path,
                torch_dtype=dtype,
            ).eval().to(self.device))
        self.codec = (
            codec_module.XCodec2Model.from_pretrained(self.config.codec_name_or_path, ).eval().to(
                self.device))
        sample_rate = getattr(
            getattr(self.codec, "feature_extractor", None),
            "sampling_rate",
            self.config.sample_rate,
        )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError("LLaSA codec reported an invalid sample rate: "
                               f"{sample_rate}.")
        self.config.sample_rate = sample_rate
        self._torch = torch

    def _prepare_for_inference(self) -> None:
        """Restore eval/cache state on the optimizer-owned runtime."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.codec is not None and hasattr(self.codec, "eval"):
            self.codec.eval()
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text", "")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty local path or None.")
        if not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string.")
        if (speaker_audio_path is not None) != bool(reference_text.strip()):
            raise ValueError(
                "LLaSA voice cloning requires `speaker_audio_path` and a "
                "non-empty `reference_text` together.")
        if speaker_audio_path is not None:
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"LLaSA reference audio was not found: {reference_path}.")
        max_new_tokens = model_inputs.get("max_new_tokens")
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        temperature = model_inputs.get("temperature")
        if temperature is None:
            temperature = self.config.temperature
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("LLaSA sampling requires `temperature` greater than zero.")
        top_p = model_inputs.get("top_p")
        if top_p is None:
            top_p = self.config.top_p
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or
                not 0 < top_p <= 1):
            raise ValueError("LLaSA sampling requires `top_p` in the interval (0, 1].")

    @staticmethod
    def _ids_to_speech_tokens(speech_ids) -> list[str]:
        return [f"<|s_{int(speech_id)}|>" for speech_id in speech_ids]

    @classmethod
    def _extract_speech_ids(cls, tokens: list[str]) -> list[int]:
        speech_ids = []
        for token in tokens:
            if token.startswith("<|s_") and token.endswith("|>"):
                value = token[4:-2]
                try:
                    speech_ids.append(int(value))
                except ValueError as exc:
                    raise RuntimeError(f"LLaSA generated a malformed speech token: {token!r}.") from exc
        if not speech_ids:
            raise RuntimeError("LLaSA generated no XCodec2 speech tokens.")
        return speech_ids

    def _load_reference(self, audio_path: str):
        np = import_optional(
            "numpy",
            model_type="llasa",
            install_extra=None,
        )
        sf = import_optional(
            "soundfile",
            model_type="llasa",
            install_extra=None,
        )
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        if getattr(audio, "size", 0) == 0:
            raise ValueError("LLaSA reference audio contains no samples.")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError(f"LLaSA reference audio has an invalid sample rate: {sample_rate!r}.")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = self._torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if sample_rate != self.sample_rate:
            torchaudio = import_optional(
                "torchaudio",
                model_type="llasa",
                install_extra=None,
            )
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate,
            )
        return waveform.unsqueeze(0)

    def _encode_reference(self, audio_path: str) -> tuple[list[int], int]:
        reference = self._load_reference(audio_path)
        with self._torch.inference_mode():
            encoded = self.codec.encode_code(
                input_waveform=reference,
                sample_rate=self.sample_rate,
            )
        prefix_ids = [int(value) for value in encoded[0, 0].detach().cpu().tolist()]
        prompt_samples = (len(prefix_ids) * self.sample_rate // self._CODEC_TOKENS_PER_SECOND)
        return prefix_ids, prompt_samples

    def _build_generation_prompt(
        self,
        text: str,
        *,
        reference_text: str,
        prefix_ids: list[int],
    ):
        formatted_text = (self._TEXT_START + reference_text + text + self._TEXT_END)
        prefix = "".join(self._ids_to_speech_tokens(prefix_ids))
        conversation = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text,
            },
            {
                "role": "assistant",
                "content": self._SPEECH_START + prefix,
            },
        ]
        return self.tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).to(self.device)

    def _generate(
        self,
        text: str,
        *,
        speaker_audio_path: str | None = None,
        reference_text: str = "",
        output_file: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        """Synthesize text, optionally conditioning on aligned reference
        audio."""
        torch = self._torch
        with seeded_inference(
                seed,
                device=self.device,
                model_type="llasa",
        ) as effective_seed:
            prefix_ids: list[int] = []
            prompt_samples = 0
            if speaker_audio_path:
                prefix_ids, prompt_samples = self._encode_reference(speaker_audio_path, )

            input_ids = self._build_generation_prompt(
                text,
                reference_text=reference_text,
                prefix_ids=prefix_ids,
            )
            speech_end_id = self.tokenizer.convert_tokens_to_ids(self._SPEECH_END, )
            with torch.inference_mode():
                generated = self.model.generate(
                    input_ids,
                    max_new_tokens=(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens),
                    eos_token_id=speech_end_id,
                    do_sample=True,
                    top_p=self.config.top_p if top_p is None else top_p,
                    temperature=(self.config.temperature if temperature is None else temperature),
                )
                generated_ids = generated[0, input_ids.shape[1]:]
                token_strings = self.tokenizer.convert_ids_to_tokens(generated_ids.detach().cpu().tolist(), )
                speech_ids = prefix_ids + self._extract_speech_ids(token_strings, )
                codec_tokens = torch.tensor(
                    speech_ids,
                    device=self.device,
                ).unsqueeze(0).unsqueeze(0)
                waveform = self.codec.decode_code(codec_tokens)[0, 0]

            if prompt_samples:
                waveform = waveform[prompt_samples:]
        waveform = waveform.detach().cpu()
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "model_type": self.config.model_type,
                "seed": effective_seed,
                "requested_seed": seed,
                "voice_cloned": bool(speaker_audio_path),
            },
        )


LlasaTTS = LlasaForTextToSpeech
