"""Dia inference with legacy Nari and official Transformers backends."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference, validate_local_file


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
        self.validate()

    def validate(self) -> None:
        if self.backend not in {"auto", "legacy", "transformers"}:
            raise ValueError("Dia backend must be 'auto', 'legacy', or 'transformers'.")
        if not isinstance(self.compute_dtype, str) or not self.compute_dtype.strip():
            raise ValueError("`compute_dtype` must be a non-empty string.")
        if not isinstance(self.use_torch_compile, bool):
            raise TypeError("`use_torch_compile` must be a boolean.")


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
        config.validate()
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
            if backend.model is None or backend.processor is None:
                raise RuntimeError("The Transformers Dia loader returned an incomplete backend.")
            sample_rate = int(backend.sample_rate)
            if sample_rate <= 0:
                raise ValueError("The Transformers Dia backend reported an invalid sample rate.")
            self.model = backend.model
            self._dia_backend = backend
            self._loaded_backend = "transformers"
            self.config.sample_rate = sample_rate
            return

        from voicehub.models.dia.model import Dia

        model = Dia.from_pretrained(
            self.config.name_or_path,
            compute_dtype=self.config.compute_dtype,
            device=self.device,
        )
        if not callable(getattr(model, "generate", None)):
            raise TypeError("The legacy Dia runtime does not implement generate().")
        self.model = model
        self._dia_backend = None
        self._loaded_backend = "legacy"
        self.config.sample_rate = 44_100

    @property
    def training_backend(self):
        """Return the official backend after a training load."""
        model_config = getattr(self.model, "config", None)
        if (self._loaded_backend == "transformers" and self._dia_backend is not None and
                not getattr(model_config, "use_cache", True)):
            return self._dia_backend
        return None

    def _prepare_for_training(self) -> None:
        if (self._loaded_backend == "transformers" and self._dia_backend is not None and
                self.model is self._dia_backend.model):
            self._dia_backend.prepare_for_training()
            return

        previous_state = (
            self.model,
            self._dia_backend,
            self._loaded_backend,
        )
        self.model = None
        self._dia_backend = None
        self._loaded_backend = None
        previous_loading_mode = self._loading_for_training
        self._loading_for_training = True
        try:
            self.load()
        except BaseException:
            self.model, self._dia_backend, self._loaded_backend = previous_state
            raise
        finally:
            self._loading_for_training = previous_loading_mode

    def _prepare_for_inference(self) -> None:
        """Restore eval and cache state for either Dia backend."""
        if self._loaded_backend == "transformers":
            if self.model is not None and hasattr(self.model, "eval"):
                self.model.eval()
            model_config = getattr(self.model, "config", None)
            if model_config is not None and hasattr(model_config, "use_cache"):
                model_config.use_cache = True
            return

        source_model = getattr(self.model, "model", None)
        if source_model is not None and hasattr(source_model, "eval"):
            source_model.eval()
        codec = getattr(self.model, "dac_model", None)
        if codec is not None and hasattr(codec, "eval"):
            codec.eval()

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

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        super()._validate_generation_inputs(model_inputs)
        for aliases in (
            ("max_tokens", "max_new_tokens"),
            ("cfg_scale", "guidance_scale"),
            ("cfg_filter_top_k", "top_k"),
        ):
            if all(model_inputs.get(name) is not None for name in aliases):
                raise ValueError(f"Pass either {aliases[0]!r} or {aliases[1]!r}, not both.")

        prompt_names = ("audio", "audio_prompt", "audio_prompt_path")
        supplied_prompts = [name for name in prompt_names if model_inputs.get(name) is not None]
        if len(supplied_prompts) > 1:
            raise ValueError("Pass only one of 'audio', 'audio_prompt', or "
                             "'audio_prompt_path'.")
        if supplied_prompts:
            prompt = model_inputs[supplied_prompts[0]]
            if isinstance(prompt, Mapping) and "path" in prompt:
                prompt = prompt["path"]
            if isinstance(prompt, (str, PathLike)):
                prompt_path = validate_local_file(
                    prompt,
                    option_name=supplied_prompts[0],
                )
                prompt_value = model_inputs[supplied_prompts[0]]
                if isinstance(prompt_value, Mapping):
                    prompt_value = dict(prompt_value)
                    prompt_value["path"] = str(prompt_path)
                    model_inputs[supplied_prompts[0]] = prompt_value
                else:
                    model_inputs[supplied_prompts[0]] = str(prompt_path)

        for name in ("max_tokens", "max_new_tokens", "cfg_filter_top_k", "top_k"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        for name in ("cfg_scale", "guidance_scale"):
            value = model_inputs.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"`{name}` must be a finite number.")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"`{name}` must be finite and non-negative.")
        seed = model_inputs.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("`seed` must be an integer or None.")

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
    def _rename_generation_options(
        options: dict[str, Any],
        aliases: tuple[tuple[str, str], ...],
    ) -> None:
        for source, target in aliases:
            if source not in options:
                continue
            if target in options:
                raise ValueError(f"Pass either {source!r} or {target!r}, not both.")
            options[target] = options.pop(source)

    @staticmethod
    def _pop_audio_prompt(options: dict[str, Any]) -> Any:
        prompt = None
        for name in ("audio", "audio_prompt", "audio_prompt_path"):
            if name not in options:
                continue
            value = options.pop(name)
            if value is None:
                continue
            if prompt is not None:
                raise ValueError("Pass only one of 'audio', 'audio_prompt', or "
                                 "'audio_prompt_path'.")
            prompt = value
        return prompt

    def _generate_legacy(
        self,
        text: str,
        generation_options: dict[str, Any],
    ) -> Any:
        self._rename_generation_options(
            generation_options,
            (
                ("max_new_tokens", "max_tokens"),
                ("guidance_scale", "cfg_scale"),
                ("top_k", "cfg_filter_top_k"),
            ),
        )
        audio_prompt = self._pop_audio_prompt(generation_options)
        if audio_prompt is not None:
            generation_options["audio_prompt"] = audio_prompt
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
        backend = self._dia_backend
        if backend is None or backend.processor is None:
            raise RuntimeError("Dia Transformers generation requires a loaded processor.")
        processor = backend.processor
        self._rename_generation_options(
            generation_options,
            (
                ("max_tokens", "max_new_tokens"),
                ("cfg_scale", "guidance_scale"),
                ("cfg_filter_top_k", "top_k"),
            ),
        )
        generation_options.pop("use_cfg_filter", None)
        generation_options.pop("verbose", None)

        audio = self._pop_audio_prompt(generation_options)
        if audio is not None:
            audio = self._audio_prompt(audio, self.sample_rate)

        inputs = processor(
            text=[text],
            audio=audio,
            padding=True,
            return_tensors="pt",
        )
        prompt_length = None
        if audio is not None:
            prompt_length = processor.get_audio_prompt_len(inputs["decoder_attention_mask"], )
        inputs = inputs.to(self.device)
        generated = self.model.generate(
            **inputs,
            **generation_options,
        )
        sequences = getattr(generated, "sequences", generated)
        decoded = processor.batch_decode(
            sequences,
            audio_prompt_len=prompt_length,
        )
        if not decoded:
            raise RuntimeError("Dia Transformers decoding returned no audio.")
        return decoded[0]

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        **generation_options,
    ) -> TTSOutput:
        options = dict(generation_options)
        requested_seed = options.pop("seed", None)
        with seeded_inference(
                requested_seed,
                device=self.device,
                model_type="dia",
        ) as effective_seed:
            if self._loaded_backend == "legacy":
                audio = self._generate_legacy(text, options)
            elif self._loaded_backend == "transformers":
                audio = self._generate_transformers(text, options)
            else:
                raise RuntimeError("Dia must select and load a backend before generation.")
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend": self._loaded_backend,
                "seed": effective_seed,
                "requested_seed": requested_seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        if self._loaded_backend == "transformers" and self._dia_backend:
            self._dia_backend.save_pretrained(save_directory)


DiaVoiceHubConfig = DiaConfig
DiaTTS = DiaForTextToSpeech
