"""Shared loading primitives for native Hugging Face TTS integrations.

The concrete Bark, SpeechT5, and VITS wrappers intentionally remain
separate: their generation and training contracts are materially
different. This module only owns the checkpoint, processor, device, and
waveform mechanics that are safe to share across current and future
Transformers TTS families.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig, reject_serialized_secrets
from voicehub.dependencies import import_optional
from voicehub.errors import OptionalDependencyError
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import resolve_torch_dtype

_WEIGHT_SUFFIXES = frozenset({".bin", ".safetensors"})


class TransformersTTSConfigBase(VoiceHubConfig):
    """Serializable loading controls shared by Transformers TTS providers."""

    model_type = "transformers_tts"

    def __init__(
        self,
        *,
        config_name_or_path: str | Path | None = None,
        processor_name_or_path: str | Path | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        use_safetensors: bool | None = None,
        torch_dtype: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        reject_serialized_secrets(
            {
                "model_kwargs": model_kwargs,
                "processor_kwargs": processor_kwargs,
                **kwargs,
            },
            owner=self.__class__.__name__,
        )
        super().__init__(**kwargs)
        self.config_name_or_path = config_name_or_path
        self.processor_name_or_path = processor_name_or_path
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.use_safetensors = use_safetensors
        self.torch_dtype = torch_dtype
        self.model_kwargs = self._copy_mapping(
            model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            processor_kwargs,
            name="processor_kwargs",
        )
        self.validate()

    @staticmethod
    def _copy_mapping(
        value: Mapping[str, Any] | None,
        *,
        name: str,
    ) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a mapping or None.")
        return dict(value)

    def validate(self) -> None:
        """Validate loading controls without importing Transformers or
        Torch."""
        reject_serialized_secrets(
            self.__dict__,
            owner=self.__class__.__name__,
        )
        for option_name in ("config_name_or_path", "processor_name_or_path"):
            value = getattr(self, option_name)
            if value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f"`{option_name}` must be a non-empty local path, Hub ID, "
                                 "or None.")
            setattr(self, option_name, str(value))
        if not isinstance(self.trust_remote_code, bool):
            raise TypeError("`trust_remote_code` must be a boolean.")
        if self.revision is not None:
            if not isinstance(self.revision, str) or not self.revision.strip():
                raise ValueError("`revision` must be a non-empty string or None.")
            self.revision = self.revision.strip()
        if self.cache_dir is not None:
            if not isinstance(self.cache_dir, (str, Path)):
                raise TypeError("`cache_dir` must be path-like or None.")
            self.cache_dir = str(self.cache_dir)
        if not isinstance(self.local_files_only, bool):
            raise TypeError("`local_files_only` must be a boolean.")
        if self.use_safetensors is not None and not isinstance(
                self.use_safetensors,
                bool,
        ):
            raise TypeError("`use_safetensors` must be a boolean or None.")
        if self.torch_dtype is not None:
            if not isinstance(self.torch_dtype, str) or not self.torch_dtype.strip():
                raise ValueError("`torch_dtype` must be a non-empty string or None.")
            self.torch_dtype = self.torch_dtype.strip()
        if (isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, Integral) or
                self.sample_rate <= 0):
            raise ValueError("`sample_rate` must be a positive integer.")
        self.sample_rate = int(self.sample_rate)

        self.model_kwargs = self._copy_mapping(
            self.model_kwargs,
            name="model_kwargs",
        )
        self.processor_kwargs = self._copy_mapping(
            self.processor_kwargs,
            name="processor_kwargs",
        )
        reserved_model_options = {
            "config",
            "state_dict",
            "token",
            "torch_dtype",
            "trust_remote_code",
            "use_safetensors",
        }
        model_conflicts = reserved_model_options.intersection(self.model_kwargs)
        if model_conflicts:
            names = ", ".join(sorted(model_conflicts))
            raise ValueError("`model_kwargs` cannot override provider-owned option(s): "
                             f"{names}.")
        processor_conflicts = {
            "token",
            "trust_remote_code",
        }.intersection(self.processor_kwargs)
        if processor_conflicts:
            names = ", ".join(sorted(processor_conflicts))
            raise ValueError("`processor_kwargs` cannot override provider-owned option(s): "
                             f"{names}.")

    def to_dict(self) -> dict[str, Any]:
        """Validate mutable overrides before serializing this configuration."""
        self.validate()
        return super().to_dict()


class TransformersTTSModelBase(PreTrainedTTSModel):
    """Checkpoint lifecycle shared by native Transformers TTS wrappers."""

    transformers_model_class = ""
    transformers_processor_class = ""

    def __init__(
        self,
        config: TransformersTTSConfigBase,
        *,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
    ):
        if token is not None and (not isinstance(token, (str, bool)) or
                                  isinstance(token, str) and not token.strip()):
            raise ValueError("`token` must be a non-empty string, boolean, or None.")
        self.native_config = None
        self.transformers_processor = None
        self._torch = None
        self.training_model = None
        # Authentication belongs to the live loader, never the serializable
        # configuration or saved checkpoint manifest.
        self._token = token.strip() if isinstance(token, str) else token
        super().__init__(config, device=device, lazy_load=lazy_load)

    @property
    def training_processor(self):
        """Return the processor paired with the differentiable checkpoint."""
        return self.transformers_processor

    @staticmethod
    def _local_weight_file(name_or_path: str | Path) -> Path | None:
        path = Path(name_or_path).expanduser()
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            return path.resolve()
        return None

    def _model_source(self) -> str:
        source = self.config.name_or_path or self.default_model_name_or_path
        weight_file = self._local_weight_file(source)
        return str(weight_file.parent) if weight_file is not None else str(source)

    def _config_source(self) -> str:
        configured = self.config.config_name_or_path
        return str(configured) if configured is not None else self._model_source()

    def _processor_source(self) -> str:
        configured = self.config.processor_name_or_path
        return str(configured) if configured is not None else self._config_source()

    def _hub_kwargs(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "revision": self.config.revision,
                "cache_dir": self.config.cache_dir,
                "local_files_only": self.config.local_files_only,
                "token": self._token,
            }.items() if value is not None
        }

    def _direct_state_dict(self) -> Mapping[str, Any] | None:
        weight_file = self._local_weight_file(self.config.name_or_path)
        if weight_file is None:
            return None
        if weight_file.suffix.lower() == ".safetensors":
            safetensors = import_optional(
                "safetensors.torch",
                model_type=self.config.model_type,
                install_extra=None,
            )
            state_dict = safetensors.load_file(
                str(weight_file),
                device="cpu",
            )
        else:
            torch = import_optional(
                "torch",
                model_type=self.config.model_type,
                install_extra=None,
            )
            try:
                state_dict = torch.load(
                    str(weight_file),
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                state_dict = torch.load(
                    str(weight_file),
                    map_location="cpu",
                )
        if not isinstance(state_dict, Mapping):
            raise TypeError("The direct Transformers checkpoint did not contain a "
                            "state-dict mapping.")
        return state_dict

    def _model_load_kwargs(self, torch: Any) -> dict[str, Any]:
        options = {
            **self._hub_kwargs(),
            **self.config.model_kwargs,
            "config": self.native_config,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.use_safetensors is not None:
            options["use_safetensors"] = self.config.use_safetensors
        if self.config.torch_dtype is not None:
            options["torch_dtype"] = resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            )
        state_dict = self._direct_state_dict()
        if state_dict is not None:
            options["state_dict"] = state_dict
        return options

    @staticmethod
    def _required_transformers_class(transformers: Any, name: str):
        model_class = getattr(transformers, name, None)
        if model_class is None:
            raise OptionalDependencyError(
                f"This TTS backend requires a Transformers release exposing "
                f"`{name}`. Upgrade the default VoiceHub installation and "
                "retry.")
        return model_class

    def _load_transformers_model_and_processor(self) -> tuple[Any, Any, Any]:
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        transformers = import_optional(
            "transformers",
            model_type=self.config.model_type,
            install_extra=None,
        )
        self.native_config = transformers.AutoConfig.from_pretrained(
            self._config_source(),
            trust_remote_code=self.config.trust_remote_code,
            **self._hub_kwargs(),
        )
        model_class = self._required_transformers_class(
            transformers,
            self.transformers_model_class,
        )
        processor_class = self._required_transformers_class(
            transformers,
            self.transformers_processor_class,
        )
        model = model_class.from_pretrained(
            self._model_source(),
            **self._model_load_kwargs(torch),
        )
        processor = processor_class.from_pretrained(
            self._processor_source(),
            trust_remote_code=self.config.trust_remote_code,
            **self._hub_kwargs(),
            **self.config.processor_kwargs,
        )
        if not self._is_dispatched(model):
            moved = model.to(self.device)
            if moved is not None:
                model = moved
        self._torch = torch
        self.transformers_processor = processor
        return transformers, model, processor

    def _is_dispatched(self, model: Any) -> bool:
        return ("device_map" in self.config.model_kwargs or bool(getattr(model, "hf_device_map", None)))

    @staticmethod
    def _move_to_device(value: Any, device: str):
        move = getattr(value, "to", None)
        if callable(move):
            moved = move(device)
            return value if moved is None else moved
        if isinstance(value, Mapping):
            return value.__class__(
                (key, TransformersTTSModelBase._move_to_device(item, device)) for key, item in value.items())
        if isinstance(value, tuple):
            return tuple(TransformersTTSModelBase._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [TransformersTTSModelBase._move_to_device(item, device) for item in value]
        return value

    def _processor_inputs(self, text: str, **kwargs) -> dict[str, Any]:
        encoded = self.transformers_processor(
            text=text,
            return_tensors="pt",
            **kwargs,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError(f"{self.transformers_processor_class} must return a mapping.")
        return dict(self._move_to_device(encoded, self.device))

    def _normalize_waveform(
        self,
        audio: Any,
        *,
        output_length: Any | None = None,
    ):
        """Return one finite, mono float32 waveform from a backend result."""
        if (isinstance(audio, (tuple, list)) and not hasattr(audio, "shape") and audio and
                not isinstance(audio[0], Real)):
            if not audio:
                raise RuntimeError("The Transformers TTS backend returned no audio.")
            audio = audio[0]
        if hasattr(audio, "detach"):
            audio = audio.detach().float().cpu().numpy()
        numpy = import_optional(
            "numpy",
            model_type=self.config.model_type,
            install_extra=None,
        )
        waveform = numpy.asarray(audio, dtype=numpy.float32)
        if waveform.ndim == 2:
            if waveform.shape[0] != 1:
                raise RuntimeError(
                    "VoiceHub requested one utterance, but the Transformers "
                    f"backend returned a batch of {waveform.shape[0]}.")
            waveform = waveform[0]
        waveform = numpy.squeeze(waveform)
        if waveform.ndim == 0:
            waveform = waveform.reshape(1)
        if waveform.ndim != 1:
            raise RuntimeError(
                "The Transformers TTS backend returned non-mono audio with "
                f"shape {waveform.shape}.")
        if output_length is not None:
            if hasattr(output_length, "detach"):
                output_length = output_length.detach().cpu()
            if isinstance(output_length, (tuple, list)):
                output_length = output_length[0]
            if hasattr(output_length, "item"):
                output_length = output_length.item()
            if (isinstance(output_length, bool) or not isinstance(output_length, Integral) or
                    output_length <= 0):
                raise RuntimeError("The Transformers TTS backend returned an invalid output "
                                   "length.")
            waveform = waveform[:int(output_length)]
        if waveform.size == 0:
            raise RuntimeError("The Transformers TTS backend returned empty audio.")
        if not numpy.isfinite(waveform).all():
            raise RuntimeError("The Transformers TTS backend returned NaN or infinite audio.")
        return waveform

    @staticmethod
    def _positive_real(value: Any, *, name: str, allow_zero: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"`{name}` must be a real number.")
        value = float(value)
        valid = value >= 0 if allow_zero else value > 0
        if not isfinite(value) or not valid:
            qualifier = "non-negative" if allow_zero else "greater than zero"
            raise ValueError(f"`{name}` must be finite and {qualifier}.")
        return value

    def _prepare_for_training(self) -> None:
        train = getattr(self.model, "train", None)
        if callable(train):
            train()
        native_config = getattr(self.model, "config", None)
        if native_config is not None and hasattr(native_config, "use_cache"):
            native_config.use_cache = False

    def _prepare_for_inference(self) -> None:
        evaluate = getattr(self.model, "eval", None)
        if callable(evaluate):
            evaluate()
        native_config = getattr(self.model, "config", None)
        if native_config is not None and hasattr(native_config, "use_cache"):
            native_config.use_cache = True

    def _save_native_bundle(
        self,
        save_directory: Path,
        *,
        processor: Any | None = None,
    ) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        save_model = getattr(self.model, "save_pretrained", None)
        if not callable(save_model):
            raise TypeError(
                "The native Transformers TTS model cannot be exported with "
                "save_pretrained().")
        save_model(
            save_directory,
            safe_serialization=True,
        )
        processor = self.transformers_processor if processor is None else processor
        save_processor = getattr(processor, "save_pretrained", None)
        if callable(save_processor):
            save_processor(save_directory)


__all__ = [
    "TransformersTTSConfigBase",
    "TransformersTTSModelBase",
]
