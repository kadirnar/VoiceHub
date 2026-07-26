"""ConversationTTS inference backed by vendored CC BY-NC source."""

from __future__ import annotations

from pathlib import Path

from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype
from voicehub.models.conversationtts.configuration_conversationtts import ConversationTTSConfig
from voicehub.models.conversationtts.runtime import resume_for_inference


class ConversationTTSForTextToSpeech(PreTrainedTTSModel):
    """Multilingual conversational synthesis with optional speaker context."""

    config_class = ConversationTTSConfig
    default_model_name_or_path = "AudioFoundation/SpeechFoundation"

    def __init__(
        self,
        config: ConversationTTSConfig | str | None = None,
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
        self._generator = None
        self._generator_module = None
        self._loaded_for_training = False
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _hub_file(self, repository_id: str, filename: str) -> Path:
        huggingface_hub = import_optional(
            "huggingface_hub",
            model_type="conversationtts",
            install_extra="conversationtts",
        )
        return Path(huggingface_hub.hf_hub_download(
            repo_id=repository_id,
            filename=filename,
        ))

    def _checkpoint_path(self) -> Path:
        source = Path(self.config.name_or_path).expanduser()
        if source.is_file():
            return source.resolve()
        if source.is_dir():
            checkpoint = source / self.config.checkpoint_filename
            if checkpoint.is_file():
                return checkpoint.resolve()
            raise FileNotFoundError(f"ConversationTTS checkpoint not found: {checkpoint}")
        return self._hub_file(
            self.config.name_or_path,
            self.config.checkpoint_filename,
        )

    def _text_tokenizer_path(self) -> Path:
        if self.config.text_tokenizer_path:
            path = Path(self.config.text_tokenizer_path).expanduser()
        else:
            path = (Path(__file__).parent / "source" / "conversationtts" / "llama3_2")
        if not path.is_dir():
            raise FileNotFoundError(f"ConversationTTS text tokenizer not found: {path}")
        return path.resolve()

    def _audio_tokenizer_path(self) -> Path:
        if self.config.audio_tokenizer_path:
            path = Path(self.config.audio_tokenizer_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"ConversationTTS audio tokenizer not found: {path}")
            return path.resolve()
        return self._hub_file(
            self.config.audio_tokenizer_repo_id,
            self.config.audio_tokenizer_filename,
        )

    def _build_raw_model(self):
        """Construct the differentiable source model without serving state."""
        torch = import_optional(
            "torch",
            model_type="conversationtts",
            install_extra="conversationtts",
        )
        model_module = import_optional(
            "voicehub.models.conversationtts.source.conversationtts."
            "models.model_new",
            model_type="conversationtts",
            install_extra="conversationtts",
        )
        model = model_module.Model(model_module.ModelArgs(**self.config.model_args))
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        model.to(device=self.device, dtype=dtype)
        return model

    def _attach_inference_runtime(self) -> None:
        """Attach tokenizers and KV caches to the current trained weights."""
        if self.model is None:
            raise RuntimeError(
                "ConversationTTS cannot build its inference runtime before "
                "the source model is loaded.")
        if self._generator is not None:
            self.model.eval()
            self._loaded_for_training = False
            return

        generator_module = import_optional(
            "voicehub.models.conversationtts.source.conversationtts."
            "inference.generator",
            model_type="conversationtts",
            install_extra="conversationtts",
        )
        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            generator = generator_module.Generator(
                self.model,
                text_tokenizer_path=str(self._text_tokenizer_path()),
                audio_tokenizer_path=str(self._audio_tokenizer_path()),
            )
        except BaseException:
            # Generator.setup_caches() runs before its tokenizer loads. A
            # tokenizer failure must not strand a cache-mutated training graph.
            self._clear_inference_caches()
            if was_training:
                self.model.train()
            raise
        self._generator = generator
        self._generator_module = generator_module
        self._loaded_for_training = False

    @staticmethod
    def _clear_transformer_caches(transformer) -> None:
        """Release TorchTune KV caches while preserving parameter identity."""
        modules = getattr(transformer, "modules", None)
        if not callable(modules):
            return
        for module in tuple(modules()):
            if hasattr(module, "kv_cache"):
                module.kv_cache = None
            if hasattr(module, "cache_enabled"):
                module.cache_enabled = False

        caches_are_setup = getattr(transformer, "caches_are_setup", None)
        if callable(caches_are_setup) and caches_are_setup():
            raise RuntimeError(
                "ConversationTTS could not remove the inference KV cache from "
                "its training graph.")
        caches_are_enabled = getattr(transformer, "caches_are_enabled", None)
        if callable(caches_are_enabled) and caches_are_enabled():
            raise RuntimeError("ConversationTTS could not disable the inference KV cache for "
                               "training.")

    def _clear_inference_caches(self) -> None:
        """Remove serving-only cache modules and masks from the source
        model."""
        if self.model is None:
            return
        for transformer_name in ("backbone", "decoder"):
            transformer = getattr(self.model, transformer_name, None)
            if transformer is not None:
                self._clear_transformer_caches(transformer)
        for buffer_name in (
                "backbone_causal_mask",
                "decoder_causal_mask",
        ):
            buffers = getattr(self.model, "_buffers", {})
            if buffer_name in buffers:
                delattr(self.model, buffer_name)

    def _prepare_for_training(self) -> None:
        """Return to the raw differentiable graph without changing weights."""
        self._generator = None
        self._generator_module = None
        self._clear_inference_caches()
        if self.model is not None and hasattr(self.model, "train"):
            self.model.train()
        self._loaded_for_training = True

    def _prepare_for_inference(self) -> None:
        """Build serving objects lazily around the current trained weights."""
        self._attach_inference_runtime()

    def _load_pretrained_model(self) -> None:
        model = self._build_raw_model()
        resume_for_inference(
            self._checkpoint_path(),
            None,
            model,
            self.device,
        )
        self.model = model
        self._generator = None
        self._generator_module = None
        self._loaded_for_training = self.is_training_load
        if self.is_training_load:
            model.train()
            return
        try:
            self._attach_inference_runtime()
        except BaseException:
            self.model = None
            self._loaded_for_training = False
            raise

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker: int = 0,
        speaker_audio_path: str | None = None,
        reference_text: str | None = None,
        max_audio_length_ms: float = 30_000,
        temperature: float = 0.9,
        top_k: int = 30,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        context = []
        if speaker_audio_path:
            if not reference_text:
                raise ValueError("reference_text is required with speaker_audio_path.")
            context.append(
                self._generator_module.prepare_prompt(
                    reference_text,
                    speaker_audio_path,
                    segment_id=speaker,
                ))
        audio = self._generator.generate_v1(
            text=text,
            speaker=speaker,
            max_audio_length_ms=max_audio_length_ms,
            context=context,
            temperature=temperature,
            topk=top_k,
            **generation_options,
        )
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "speaker": speaker,
                "voice_cloned": bool(context),
                "license": "CC BY-NC 4.0",
                "commercial_use": False,
            },
        )


ConversationTTS = ConversationTTSForTextToSpeech
