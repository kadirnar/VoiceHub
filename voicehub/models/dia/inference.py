"""Dia inference with legacy Nari and official Transformers backends."""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class DiaConfig(VoiceHubConfig):
    """VoiceHub loading and generation configuration for Dia.

    ``backend="auto"`` preserves the original Nari runtime for the
    legacy ``Dia-1.6B`` artifact and selects Transformers for the
    converted ``Dia-1.6B-0626`` checkpoint. Fine-tuning always requires
    the latter.
    """

    model_type = "dia"

    def __init__(
        self,
        *,
        backend: str = "auto",
        compute_dtype: str = "bfloat16",
        use_torch_compile: bool = False,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        normalized_backend = str(backend).strip().lower()
        if normalized_backend not in {"auto", "legacy", "transformers"}:
            raise ValueError("Dia backend must be 'auto', 'legacy', or 'transformers'.")
        self.backend = normalized_backend
        self.compute_dtype = compute_dtype
        self.use_torch_compile = use_torch_compile


class DiaForTextToSpeech(PreTrainedTTSModel):
    """Dialogue synthesis and fine-tuning across both released Dia formats."""

    config_class = DiaConfig
    default_model_name_or_path = "nari-labs/Dia-1.6B-0626"
    default_transformers_model_name_or_path = "nari-labs/Dia-1.6B-0626"

    def __init__(
        self,
        config: DiaConfig | str | None = None,
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
        self._dia_backend = None
        self._loaded_backend: str | None = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _is_legacy_checkpoint(name_or_path: str) -> bool:
        normalized = str(name_or_path).rstrip("/").lower()
        if normalized == "nari-labs/dia-1.6b":
            return True
        source = Path(name_or_path).expanduser()
        config_path = (
            source if source.is_file() and source.name == "config.json" else source / "config.json")
        if not config_path.is_file():
            return False
        try:
            values = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return (
            isinstance(values, Mapping) and "data" in values and "model" in values and
            values.get("model_type") != "dia")

    def _select_backend(self, *, for_training: bool) -> str:
        requested = self.config.backend
        if requested != "auto":
            return requested
        if for_training:
            return "transformers"
        if self._is_legacy_checkpoint(self.config.name_or_path):
            return "legacy"
        return "transformers"

    def _validate_training_runtime(self) -> None:
        if self.config.backend == "legacy":
            raise ValueError(
                "Dia's legacy Nari runtime is inference-only. Fine-tuning "
                "requires backend='transformers' (or 'auto') and the "
                f"{self.default_transformers_model_name_or_path!r} checkpoint.")
        if self._is_legacy_checkpoint(self.config.name_or_path):
            raise ValueError(
                f"{self.config.name_or_path!r} uses Dia's original "
                "inference-only checkpoint format. Fine-tune "
                f"{self.default_transformers_model_name_or_path!r} instead.")

    def _load_pretrained_model(self) -> None:
        selected_backend = self._select_backend(for_training=self.is_training_load, )
        if selected_backend == "transformers":
            from voicehub.models.dia.training import load_dia_transformers_backend

            backend = load_dia_transformers_backend(
                self.config.name_or_path,
                device=self.device,
                compute_dtype=self.config.compute_dtype,
                for_training=self.is_training_load,
            )
            self.model = backend.model
            self._dia_backend = backend
            self._loaded_backend = "transformers"
            self.config.sample_rate = backend.sample_rate
            return

        from voicehub.models.dia.model import Dia

        self.model = Dia.from_pretrained(
            self.config.name_or_path,
            compute_dtype=self.config.compute_dtype,
            device=self.device,
        )
        self._dia_backend = None
        self._loaded_backend = "legacy"

    @property
    def training_backend(self):
        """Return the official backend after a training load."""
        if (self._loaded_backend == "transformers" and self._dia_backend is not None and
                not getattr(self.model.config, "use_cache", True)):
            return self._dia_backend
        return None

    def _prepare_for_training(self) -> None:
        if (self._loaded_backend == "transformers" and self._dia_backend is not None and
                self.model is self._dia_backend.model):
            self._dia_backend.prepare_for_training()
            return

        self.model = None
        self._dia_backend = None
        self._loaded_backend = None
        self._loading_for_training = True
        try:
            self.load()
        finally:
            self._loading_for_training = False

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Prepare official delayed codec inputs and labels from raw
        records."""
        del phase
        backend = self.training_backend
        if backend is None:
            raise RuntimeError("Dia training inputs require load_for_training() before "
                               "preparation.")
        return backend.prepare_inputs(inputs)

    @staticmethod
    def _audio_prompt(audio: Any, sample_rate: int) -> Any:
        source_rate = None
        if isinstance(audio, Mapping):
            source_rate = audio.get("sampling_rate")
            if "array" in audio:
                audio = audio["array"]
            elif "path" in audio:
                audio = audio["path"]
            else:
                raise ValueError("Dia audio prompts require an 'array' or 'path' field.")
        if isinstance(audio, (str, PathLike)):
            soundfile = import_optional(
                "soundfile",
                model_type="dia",
                install_extra="dia",
            )
            numpy = import_optional(
                "numpy",
                model_type="dia",
                install_extra="dia",
            )
            audio, source_rate = soundfile.read(
                str(audio),
                dtype="float32",
                always_2d=False,
            )
            if audio.ndim > 1:
                audio = numpy.mean(audio, axis=-1)
        if source_rate is not None and int(source_rate) != sample_rate:
            raise ValueError("Dia audio prompts must be "
                             f"{sample_rate} Hz; received {source_rate} Hz.")
        return audio

    @staticmethod
    def _seed_generation(options: dict[str, Any]) -> None:
        seed = options.pop("seed", None)
        if seed is None:
            return
        torch = import_optional(
            "torch",
            model_type="dia",
            install_extra="dia",
        )
        torch.manual_seed(int(seed))

    def _generate_legacy(
        self,
        text: str,
        generation_options: dict[str, Any],
    ) -> Any:
        if ("max_new_tokens" in generation_options and "max_tokens" not in generation_options):
            generation_options["max_tokens"] = generation_options.pop("max_new_tokens")
        self._seed_generation(generation_options)
        return self.model.generate(
            text,
            use_torch_compile=self.config.use_torch_compile,
            **generation_options,
        )

    def _generate_transformers(
        self,
        text: str,
        generation_options: dict[str, Any],
    ) -> Any:
        processor = self._dia_backend.processor
        self._seed_generation(generation_options)
        aliases = (
            ("max_tokens", "max_new_tokens"),
            ("cfg_scale", "guidance_scale"),
            ("cfg_filter_top_k", "top_k"),
        )
        for source, target in aliases:
            if source not in generation_options:
                continue
            if target in generation_options:
                raise ValueError(f"Pass either {source!r} or {target!r}, not both.")
            generation_options[target] = generation_options.pop(source)
        generation_options.pop("use_cfg_filter", None)
        generation_options.pop("verbose", None)

        audio = generation_options.pop("audio", None)
        for alias in ("audio_prompt", "audio_prompt_path"):
            value = generation_options.pop(alias, None)
            if value is None:
                continue
            if audio is not None:
                raise ValueError("Pass only one of 'audio', 'audio_prompt', or "
                                 "'audio_prompt_path'.")
            audio = value
        if audio is not None:
            audio = self._audio_prompt(audio, self.sample_rate)

        inputs = processor(
            text=[text],
            audio=audio,
            padding=True,
            return_tensors="pt",
        )
        prompt_length = (
            processor.get_audio_prompt_len(inputs["decoder_attention_mask"], ) if audio is not None else None)
        inputs = inputs.to(self.device)
        generated = self.model.generate(
            **inputs,
            **generation_options,
        )
        sequences = getattr(generated, "sequences", generated)
        return processor.batch_decode(
            sequences,
            audio_prompt_len=prompt_length,
        )[0]

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        options = dict(generation_options)
        if self._loaded_backend == "legacy":
            audio = self._generate_legacy(text, options)
        else:
            audio = self._generate_transformers(text, options)
        output = TTSOutput(
            audio=audio,
            sample_rate=self.sample_rate,
            metadata={
                "backend": self._loaded_backend,
            },
        )
        if output_file:
            output.save(output_file)
        return output

    def _save_pretrained(self, save_directory: Path) -> None:
        if self._loaded_backend == "transformers" and self._dia_backend:
            self._dia_backend.save_pretrained(save_directory)


DiaVoiceHubConfig = DiaConfig
DiaTTS = DiaForTextToSpeech
