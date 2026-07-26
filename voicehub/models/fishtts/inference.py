"""Fish Speech/OpenAudio inference backed by vendored model source."""

from __future__ import annotations

import math
from numbers import Real
from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, resolve_torch_dtype, seeded_inference


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
        self._model_directory = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _load_runtime_module():
        return import_optional(
            "voicehub.models.fishtts.source.fish_speech.models."
            "text2semantic.inference",
            model_type="fishtts",
            install_extra="fishtts",
        )

    @staticmethod
    def _codec_sample_rate(codec, fallback: int) -> int:
        sample_rate = getattr(
            getattr(codec, "spec_transform", None),
            "sample_rate",
            getattr(codec, "sample_rate", fallback),
        )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError(f"Fish TTS codec returned an invalid sample rate: {sample_rate}.")
        return sample_rate

    def _setup_semantic_caches(self, model, torch) -> None:
        with torch.device(self.device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        model._cache_setup_done = True

    def _resolve_model_directory(self):
        if self._model_directory is None:
            self._model_directory = resolve_model_directory(
                self.config.name_or_path,
                model_type="fishtts",
            )
        return self._model_directory

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        speaker_audio_path = model_inputs.get("speaker_audio_path")
        reference_text = model_inputs.get("reference_text")
        if speaker_audio_path is not None:
            if (not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip()):
                raise ValueError("`speaker_audio_path` must be a non-empty local path or None.")
        if reference_text is not None and not isinstance(reference_text, str):
            raise TypeError("`reference_text` must be a string or None.")
        if reference_text and speaker_audio_path is None:
            raise ValueError("`reference_text` is only valid with "
                             "`speaker_audio_path`.")
        if speaker_audio_path is not None and not (isinstance(reference_text, str) and
                                                   reference_text.strip()):
            raise ValueError("Fish TTS voice cloning requires a non-empty "
                             "`reference_text`.")
        if speaker_audio_path is not None:
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"Fish TTS reference audio was not found: {reference_path}.")
        max_new_tokens = model_inputs.get("max_new_tokens", 1024)
        if (isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        chunk_length = model_inputs.get("chunk_length", 512)
        if (isinstance(chunk_length, bool) or not isinstance(chunk_length, int) or chunk_length < 0):
            raise ValueError("`chunk_length` must be a non-negative integer.")
        temperature = model_inputs.get("temperature", 1.0)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or not 0 < temperature < 2):
            raise ValueError("Fish TTS sampling requires `temperature` in the "
                             "interval (0, 2).")
        top_p = model_inputs.get("top_p", 0.9)
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or
                not 0 < top_p <= 1):
            raise ValueError("Fish TTS sampling requires `top_p` in the "
                             "interval (0, 1].")
        top_k = model_inputs.get("top_k", 30)
        if (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("`top_k` must be a positive integer.")
        repetition_penalty = model_inputs.get("repetition_penalty", 1.1)
        if (isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, Real) or
                not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
            raise ValueError("`repetition_penalty` must be a finite positive number.")
        num_samples = model_inputs.get("num_samples", 1)
        if (isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples != 1):
            raise ValueError("VoiceHub's Fish TTS waveform contract currently requires "
                             "`num_samples=1`.")

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="fishtts",
            install_extra="fishtts",
        )
        model_directory = self._resolve_model_directory()
        runtime = self._load_runtime_module()
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
        self._setup_semantic_caches(model, torch)
        codec = runtime.load_codec_model(
            model_directory / "codec.pth",
            self.device,
            dtype,
        )
        self.config.sample_rate = self._codec_sample_rate(
            codec,
            self.sample_rate,
        )
        self._runtime = runtime
        self._decode_one_token = decode_one_token
        self._codec = codec
        self._torch = torch
        self.model = model
        self._loaded_for_training = False

    @staticmethod
    def _clear_semantic_caches(model) -> None:
        """Release serving-only KV caches without replacing ``model``."""
        for layer_group_name in ("layers", "fast_layers"):
            layer_group = getattr(model, layer_group_name, None) or ()
            for layer in layer_group:
                attention = getattr(layer, "attention", None)
                if attention is not None and hasattr(attention, "kv_cache"):
                    attention.kv_cache = None
        for name in ("max_batch_size", "max_seq_len"):
            if hasattr(model, name):
                setattr(model, name, -1)
        if hasattr(model, "_cache_setup_done"):
            model._cache_setup_done = False

    def _prepare_for_training(self) -> None:
        if self._loaded_for_training:
            return
        # Fish compiles only the token decoder, not the semantic module.
        # Keep the exact semantic object so an existing optimizer remains
        # attached across generation -> continued-training transitions.
        self._runtime = None
        self._decode_one_token = None
        self._clear_semantic_caches(self.model)
        if hasattr(self.model, "train"):
            self.model.train()
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        self._loaded_for_training = True

    def _prepare_for_inference(self) -> None:
        if not self._loaded_for_training:
            return

        torch = self._torch
        if torch is None:
            torch = import_optional(
                "torch",
                model_type="fishtts",
                install_extra="fishtts",
            )
        runtime = self._load_runtime_module()
        model_directory = self._resolve_model_directory()

        semantic_model = self.model
        codec = self._codec
        try:
            dtype = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
            decode_one_token = runtime.prepare_model_for_inference(
                semantic_model,
                self.device,
                compile=self.config.compile,
            )
            self._setup_semantic_caches(semantic_model, torch)
            if codec is None:
                codec = runtime.load_codec_model(
                    model_directory / "codec.pth",
                    self.device,
                    dtype,
                )
            elif callable(getattr(codec, "to", None)):
                moved_codec = codec.to(device=self.device, dtype=dtype)
                if moved_codec is not None:
                    codec = moved_codec
            sample_rate = self._codec_sample_rate(
                codec,
                self.sample_rate,
            )
        except Exception:
            self._clear_semantic_caches(semantic_model)
            if hasattr(semantic_model, "train"):
                semantic_model.train()
            raise

        self._model_directory = model_directory
        self._runtime = runtime
        self._decode_one_token = decode_one_token
        self._codec = codec
        self._torch = torch
        self.config.sample_rate = sample_rate
        self._loaded_for_training = False

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
        num_samples: int = 1,
        seed: int | None = None,
    ) -> TTSOutput:
        prompt_tokens = None
        prompt_text = None
        pieces = []
        with seeded_inference(
                seed,
                device=self.device,
                model_type="fishtts",
        ) as effective_seed:
            with self._torch.inference_mode():
                if speaker_audio_path:
                    prompt_tokens = self._runtime.encode_audio(
                        str(Path(speaker_audio_path).expanduser()),
                        self._codec,
                        self.device,
                    )
                    prompt_text = reference_text
                responses = self._runtime.generate_long(
                    model=self.model,
                    device=self.device,
                    decode_one_token=self._decode_one_token,
                    text=text,
                    num_samples=num_samples,
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
                )
                for response in responses:
                    if (getattr(response, "action", None) == "sample" and
                            getattr(response, "codes", None) is not None):
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
            metadata={
                "seed": effective_seed,
                "requested_seed": seed,
                "voice_cloned": bool(speaker_audio_path),
            },
        )


FishTTS = FishTTSForTextToSpeech
