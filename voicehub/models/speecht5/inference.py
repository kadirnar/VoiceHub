"""Native Hugging Face SpeechT5 inference and supervised fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.audio import AudioInput
from voicehub.configuration_utils import reject_serialized_secrets
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.models._shared import finish_audio_output, seeded_inference
from voicehub.models._transformers_tts import TransformersTTSConfigBase, TransformersTTSModelBase


class SpeechT5Config(TransformersTTSConfigBase):
    """Loading controls for SpeechT5 and its HiFi-GAN vocoder."""

    model_type = "speecht5"

    def __init__(
        self,
        *,
        vocoder_name_or_path: str | Path = "microsoft/speecht5_hifigan",
        vocoder_kwargs: Mapping[str, Any] | None = None,
        default_speaker_embedding_path: str | Path | None = None,
        sample_rate: int = 16_000,
        **kwargs,
    ):
        reject_serialized_secrets(
            {"vocoder_kwargs": vocoder_kwargs},
            owner=self.__class__.__name__,
        )
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.vocoder_name_or_path = vocoder_name_or_path
        self.vocoder_kwargs = self._copy_mapping(
            vocoder_kwargs,
            name="vocoder_kwargs",
        )
        self.default_speaker_embedding_path = default_speaker_embedding_path
        self.validate()

    def validate(self) -> None:
        super().validate()
        vocoder_source = getattr(
            self,
            "vocoder_name_or_path",
            "microsoft/speecht5_hifigan",
        )
        if (not isinstance(vocoder_source, (str, Path)) or not str(vocoder_source).strip()):
            raise ValueError("`vocoder_name_or_path` must be a non-empty path or Hub ID.")
        self.vocoder_name_or_path = str(vocoder_source)
        self.vocoder_kwargs = self._copy_mapping(
            getattr(self, "vocoder_kwargs", None),
            name="vocoder_kwargs",
        )
        conflicts = {
            "token",
            "torch_dtype",
            "trust_remote_code",
            "use_safetensors",
        }.intersection(self.vocoder_kwargs)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError("`vocoder_kwargs` cannot override provider-owned option(s): "
                             f"{names}.")
        speaker_path = getattr(
            self,
            "default_speaker_embedding_path",
            None,
        )
        if speaker_path is not None:
            if (not isinstance(speaker_path, (str, Path)) or not str(speaker_path).strip()):
                raise ValueError("`default_speaker_embedding_path` must be a non-empty "
                                 "path or None.")
            self.default_speaker_embedding_path = str(speaker_path)


class SpeechT5ForTextToSpeech(TransformersTTSModelBase):
    """SpeechT5 synthesis with native spectrogram-loss fine-tuning."""

    config_class = SpeechT5Config
    default_model_name_or_path = "microsoft/speecht5_tts"
    transformers_model_class = "SpeechT5ForTextToSpeech"
    transformers_processor_class = "SpeechT5Processor"
    passthrough_generation_options = frozenset()

    def __init__(
        self,
        config: SpeechT5Config | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        config.validate()
        self.vocoder = None
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
            token=token,
        )

    def _load_pretrained_model(self) -> None:
        transformers, model, _ = self._load_transformers_model_and_processor()
        vocoder_class = self._required_transformers_class(
            transformers,
            "SpeechT5HifiGan",
        )
        vocoder_options = {
            **self._hub_kwargs(),
            **self.config.vocoder_kwargs,
        }
        if self.config.use_safetensors is not None:
            vocoder_options["use_safetensors"] = self.config.use_safetensors
        if self.config.torch_dtype is not None:
            from voicehub.models._shared import resolve_torch_dtype

            vocoder_options["torch_dtype"] = resolve_torch_dtype(
                self._torch,
                self.config.torch_dtype,
                self.device,
            )
        vocoder = vocoder_class.from_pretrained(
            self.config.vocoder_name_or_path,
            **vocoder_options,
        )
        if not ("device_map" in self.config.vocoder_kwargs or bool(getattr(vocoder, "hf_device_map", None))):
            moved = vocoder.to(self.device)
            if moved is not None:
                vocoder = moved
        self.model = model
        self.vocoder = vocoder
        sample_rate = getattr(
            getattr(vocoder, "config", None),
            "sampling_rate",
            self.config.sample_rate,
        )
        self.config.sample_rate = int(sample_rate)

    @staticmethod
    def _single_tensor_from_mapping(
        values: Mapping[str, Any],
        *,
        source: Path,
    ) -> Any:
        preferred_names = (
            "speaker_embeddings",
            "speaker_embedding",
            "xvector",
            "embedding",
        )
        for name in preferred_names:
            if name in values:
                return values[name]
        if len(values) == 1:
            return next(iter(values.values()))
        available = ", ".join(sorted(str(name) for name in values))
        raise ValueError(
            f"Speaker embedding file {source} must contain one tensor or one "
            f"of {preferred_names}; found: {available}.")

    def _load_speaker_embedding_file(self, value: str | Path):
        path = Path(value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"SpeechT5 speaker embedding was not found: {path}.")
        suffix = path.suffix.lower()
        if suffix == ".safetensors":
            safetensors = import_optional(
                "safetensors.torch",
                model_type=self.config.model_type,
                install_extra=None,
            )
            values = safetensors.load_file(str(path), device="cpu")
            return self._single_tensor_from_mapping(values, source=path)
        if suffix == ".npy":
            numpy = import_optional(
                "numpy",
                model_type=self.config.model_type,
                install_extra=None,
            )
            return numpy.load(path, allow_pickle=False)
        if suffix in {".bin", ".pt", ".pth"}:
            try:
                values = self._torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                values = self._torch.load(path, map_location="cpu")
            if isinstance(values, Mapping):
                return self._single_tensor_from_mapping(values, source=path)
            return values
        raise ValueError("SpeechT5 speaker embeddings must use .safetensors, .npy, .bin, "
                         ".pt, or .pth.")

    def _coerce_speaker_embeddings(
        self,
        speaker_embeddings: Any | None,
        *,
        speaker_embedding_path: str | Path | None = None,
        device: str | None = None,
    ):
        if speaker_embeddings is not None and speaker_embedding_path is not None:
            raise ValueError("Pass `speaker_embeddings` or `speaker_embedding_path`, not "
                             "both.")
        if speaker_embeddings is None:
            source = (speaker_embedding_path or self.config.default_speaker_embedding_path)
            if source is not None:
                speaker_embeddings = self._load_speaker_embedding_file(source)

        embedding_dim = int(getattr(
            getattr(self.model, "config", None),
            "speaker_embedding_dim",
            512,
        ))
        if speaker_embeddings is None:
            tensor = self._torch.zeros((1, embedding_dim))
        elif hasattr(speaker_embeddings, "detach"):
            tensor = speaker_embeddings
        else:
            tensor = self._torch.as_tensor(speaker_embeddings)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != embedding_dim:
            raise ValueError(
                "SpeechT5 speaker embeddings must have shape "
                f"(batch, {embedding_dim}); received {tuple(tensor.shape)}.")

        model_dtype = None
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                model_dtype = next(parameters()).dtype
            except (StopIteration, TypeError):
                model_dtype = None
        destination = self.device if device is None else device
        if model_dtype is None:
            return tensor.to(destination)
        return tensor.to(device=destination, dtype=model_dtype)

    @staticmethod
    def _training_audio(
        value: Any,
        sampling_rate: Any,
    ) -> tuple[Any, int]:
        if isinstance(value, AudioInput):
            return value.waveform, int(value.sampling_rate)
        if isinstance(value, Mapping):
            waveform = None
            for name in ("array", "waveform", "audio", "input_values"):
                if name in value:
                    waveform = value[name]
                    break
            if waveform is None:
                raise ValueError(
                    "SpeechT5 audio mappings require array, waveform, audio, "
                    "or input_values.")
            sampling_rate = value.get(
                "sampling_rate",
                value.get("sample_rate", sampling_rate),
            )
            value = waveform
        if hasattr(sampling_rate, "detach"):
            rates = sampling_rate.detach().reshape(-1)
            if rates.numel() == 0:
                raise ValueError("SpeechT5 training sampling rates cannot be empty.")
            first = int(rates[0].item())
            if rates.numel() > 1 and not bool((rates == first).all().item()):
                raise ValueError("Every item in a SpeechT5 training batch must share one "
                                 "sampling rate.")
            sampling_rate = first
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError("Raw SpeechT5 training audio requires a positive "
                             "`sampling_rate`.")
        return value, int(sampling_rate)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Create SpeechT5 token and mel targets from raw or prepared data."""
        if phase != "spectrogram":
            raise ValueError(f"Unknown SpeechT5 training phase {phase!r}.")
        prepared = dict(inputs)
        embedding_path = prepared.pop("speaker_embedding_path", None)
        if "input_ids" not in prepared or "labels" not in prepared:
            text = prepared.get("text")
            audio_target = prepared.get(
                "audio_target",
                prepared.get("audio", prepared.get("audio_values")),
            )
            if text is None or audio_target is None:
                raise ValueError(
                    "SpeechT5 fine-tuning requires prepared input_ids/labels "
                    "or raw text/audio.")
            audio_target, sampling_rate = self._training_audio(
                audio_target,
                prepared.get("sampling_rate"),
            )
            encoded = self.transformers_processor(
                text=text,
                audio_target=audio_target,
                sampling_rate=sampling_rate,
                padding=True,
                return_tensors="pt",
            )
            if not isinstance(encoded, Mapping):
                raise TypeError("SpeechT5Processor must return a mapping for training.")
            prepared = dict(encoded)

        if ("speaker_embeddings" in inputs or embedding_path is not None or
                self.config.default_speaker_embedding_path is not None):
            prepared["speaker_embeddings"] = self._coerce_speaker_embeddings(
                inputs.get("speaker_embeddings"),
                speaker_embedding_path=embedding_path,
                device="cpu",
            )
        return prepared

    def _generate(
        self,
        text: str,
        *,
        speaker_embeddings: Any | None = None,
        speaker_embedding_path: str | Path | None = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        output_file: str | Path | None = None,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        threshold = self._positive_real(
            threshold,
            name="threshold",
            allow_zero=True,
        )
        if threshold > 1:
            raise ValueError("`threshold` must be between 0 and 1.")
        minlenratio = self._positive_real(
            minlenratio,
            name="minlenratio",
            allow_zero=True,
        )
        maxlenratio = self._positive_real(
            maxlenratio,
            name="maxlenratio",
        )
        if minlenratio > maxlenratio:
            raise ValueError("`minlenratio` cannot exceed `maxlenratio`.")
        if "return_output_lengths" in generation_options:
            raise ValueError("`return_output_lengths` is managed by VoiceHub and cannot "
                             "be overridden.")
        inputs = self._processor_inputs(text)
        speaker = self._coerce_speaker_embeddings(
            speaker_embeddings,
            speaker_embedding_path=speaker_embedding_path,
        )
        generation_options.update({
            "speaker_embeddings": speaker,
            "threshold": threshold,
            "minlenratio": minlenratio,
            "maxlenratio": maxlenratio,
            "vocoder": self.vocoder,
            "return_output_lengths": True,
        })
        with seeded_inference(
                seed,
                device=self.device,
                model_type=self.config.model_type,
        ) as effective_seed:
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    **generation_options,
                )
        if not isinstance(generated, tuple) or len(generated) < 2:
            raise RuntimeError("SpeechT5 did not return the waveform lengths requested by "
                               "VoiceHub.")
        waveform = self._normalize_waveform(
            generated[0],
            output_length=generated[1],
        )
        return finish_audio_output(
            waveform,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "backend":
                "transformers",
                "vocoder":
                self.config.vocoder_name_or_path,
                "speaker_embedding": (
                    "provided" if (
                        speaker_embeddings is not None or speaker_embedding_path is not None or
                        self.config.default_speaker_embedding_path is not None) else "zero"),
                "seed":
                effective_seed,
                "requested_seed":
                seed,
            },
        )

    def _save_pretrained(self, save_directory: Path) -> None:
        self._save_native_bundle(save_directory)
        if self.vocoder is not None:
            self.vocoder.save_pretrained(
                save_directory / "vocoder",
                safe_serialization=True,
            )


SpeechT5TTS = SpeechT5ForTextToSpeech

__all__ = [
    "SpeechT5Config",
    "SpeechT5ForTextToSpeech",
    "SpeechT5TTS",
]
