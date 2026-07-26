"""Fish Speech/OpenAudio inference backed by vendored model source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, resolve_torch_dtype


class FishTTSConfig(VoiceHubConfig):
    """Configuration for Fish Speech S2/OpenAudio checkpoints."""

    model_type = "fishtts"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        compile: bool = False,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.compile = compile


class FishTTSForTextToSpeech(PreTrainedTTSModel):
    """Fish Speech semantic generation and vendored DAC decoding."""

    config_class = FishTTSConfig
    default_model_name_or_path = "fishaudio/s2-pro"

    def __init__(
        self,
        config: FishTTSConfig | str | None = None,
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
        self._decode_one_token = None
        self._codec = None
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="fishtts",
        )
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="fishtts",
        )
        runtime = import_optional(
            "voicehub.models.fishtts.source.fish_speech.models."
            "text2semantic.inference",
            model_type="fishtts",
            install_extra="fishtts",
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        model, decode_one_token = runtime.init_model(
            model_directory,
            self.device,
            dtype,
            compile=self.config.compile,
        )
        with torch.device(self.device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        codec = runtime.load_codec_model(
            model_directory / "codec.pth",
            self.device,
            dtype,
        )
        self.config.sample_rate = int(
            getattr(
                getattr(codec, "spec_transform", None),
                "sample_rate",
                getattr(codec, "sample_rate", self.sample_rate),
            ))
        self._runtime = runtime
        self._decode_one_token = decode_one_token
        self._codec = codec
        self._torch = torch
        self.model = model

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        max_new_tokens: int = 1024,
        top_p: float = 0.9,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        temperature: float = 1.0,
        chunk_length: int = 512,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        prompt_tokens = None
        prompt_text = None
        if speaker_audio_path:
            if not reference_text:
                raise ValueError("Fish TTS voice cloning requires reference_text.")
            prompt_tokens = self._runtime.encode_audio(
                speaker_audio_path,
                self._codec,
                self.device,
            )
            prompt_text = reference_text
        pieces = []
        for response in self._runtime.generate_long(
                model=self.model,
                device=self.device,
                decode_one_token=self._decode_one_token,
                text=text,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=self.config.compile,
                iterative_prompt=chunk_length > 0,
                chunk_length=chunk_length,
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
                **generation_options,
        ):
            if response.action == "sample" and response.codes is not None:
                pieces.append(
                    self._runtime.decode_to_audio(
                        response.codes,
                        self._codec,
                    ).detach().float().cpu())
        if not pieces:
            raise RuntimeError("Fish TTS did not generate any audio codes.")
        return finish_audio_output(
            self._torch.cat(pieces),
            self.sample_rate,
            output_file=output_file,
            metadata={"voice_cloned": bool(speaker_audio_path)},
        )


FishTTS = FishTTSForTextToSpeech
