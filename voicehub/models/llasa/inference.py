"""LLaSA integration using Transformers and vendored XCodec2 source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


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
            install_extra="llasa",
        )
        transformers = import_optional(
            "transformers",
            model_type="llasa",
            install_extra="llasa",
        )
        codec_module = import_optional(
            "voicehub.models.llasa.source.xcodec2.modeling_xcodec2",
            model_type="llasa",
            install_extra="llasa",
        )
        dtype = getattr(torch, self.config.torch_dtype)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.name_or_path)
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                self.config.name_or_path,
                torch_dtype=dtype,
            ).eval().to(self.device))
        self.codec = (
            codec_module.XCodec2Model.from_pretrained(self.config.codec_name_or_path, ).eval().to(
                self.device))
        self._torch = torch

    @staticmethod
    def _ids_to_speech_tokens(speech_ids) -> list[str]:
        return [f"<|s_{int(speech_id)}|>" for speech_id in speech_ids]

    @staticmethod
    def _extract_speech_ids(tokens: list[str]) -> list[int]:
        speech_ids = []
        for token in tokens:
            if token.startswith("<|s_") and token.endswith("|>"):
                speech_ids.append(int(token[4:-2]))
        if not speech_ids:
            raise RuntimeError("LLaSA generated no XCodec2 speech tokens.")
        return speech_ids

    def _load_reference(self, audio_path: str):
        np = import_optional(
            "numpy",
            model_type="llasa",
            install_extra="llasa",
        )
        sf = import_optional(
            "soundfile",
            model_type="llasa",
            install_extra="llasa",
        )
        audio, sample_rate = sf.read(audio_path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = self._torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if sample_rate != self.sample_rate:
            torchaudio = import_optional(
                "torchaudio",
                model_type="llasa",
                install_extra="llasa",
            )
            waveform = torchaudio.functional.resample(
                waveform,
                sample_rate,
                self.sample_rate,
            )
        return waveform.unsqueeze(0)

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
        self.load()
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)

        prefix_ids: list[int] = []
        prompt_samples = 0
        if speaker_audio_path:
            reference = self._load_reference(speaker_audio_path)
            with torch.no_grad():
                encoded = self.codec.encode_code(
                    input_waveform=reference,
                    sample_rate=self.sample_rate,
                )
            prefix_ids = [int(value) for value in encoded[0, 0].detach().cpu().tolist()]
            # XCodec2 emits 50 tokens per second at 16 kHz.
            prompt_samples = len(prefix_ids) * (self.sample_rate // 50)

        formatted_text = (
            "<|TEXT_UNDERSTANDING_START|>"
            f"{reference_text}{text}"
            "<|TEXT_UNDERSTANDING_END|>")
        prefix = "".join(self._ids_to_speech_tokens(prefix_ids))
        chat = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + formatted_text,
            },
            {
                "role": "assistant",
                "content": "<|SPEECH_GENERATION_START|>" + prefix,
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors="pt",
            continue_final_message=True,
        ).to(self.device)
        speech_end_id = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=(self.config.max_new_tokens if max_new_tokens is None else max_new_tokens),
                eos_token_id=speech_end_id,
                do_sample=True,
                top_p=self.config.top_p if top_p is None else top_p,
                temperature=(self.config.temperature if temperature is None else temperature),
            )
            generated_ids = generated[0, input_ids.shape[1]:]
            token_strings = self.tokenizer.convert_ids_to_tokens(generated_ids.detach().cpu().tolist())
            speech_ids = prefix_ids + self._extract_speech_ids(token_strings)
            codec_tokens = torch.tensor(
                speech_ids,
                device=self.device,
            ).unsqueeze(0).unsqueeze(0)
            waveform = self.codec.decode_code(codec_tokens)[0, 0]

        if prompt_samples:
            waveform = waveform[prompt_samples:]
        waveform = waveform.detach().cpu()
        file_path = (self.save_audio(output_file, waveform, self.sample_rate) if output_file else None)
        return TTSOutput(
            audio=waveform,
            sample_rate=self.sample_rate,
            file_path=file_path,
            metadata={
                "model_type": self.config.model_type,
                "seed": seed,
                "voice_cloned": bool(speaker_audio_path),
            },
        )


LlasaTTS = LlasaForTextToSpeech
