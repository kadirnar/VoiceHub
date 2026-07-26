"""Official Transformers fine-tuning support for Dia.

Dia's released Transformers implementation owns the complete teacher-forced
objective, including the nine-codebook delay layout and label masking.  This
module keeps that contract intact: raw text/audio records are prepared by
``DiaProcessor`` and training consumes the scalar loss returned by
``DiaForConditionalGeneration``.

PyTorch and Transformers remain lazy optional dependencies so importing
VoiceHub does not initialize either framework.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from os import PathLike
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.errors import OptionalDependencyError
from voicehub.training.adapters import Seq2SeqTrainingAdapter


_PREPARED_INPUT_KEYS = frozenset({
    "input_ids",
    "decoder_input_ids",
    "labels",
})
_PROCESSOR_CONTROL_KEYS = frozenset({
    "audio",
    "generation",
    "output_labels",
    "padding",
    "return_tensors",
    "text",
})


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


def _transformers_major_version(transformers: Any) -> int | None:
    version = str(getattr(transformers, "__version__", "")).split(".", 1)[0]
    try:
        return int(version)
    except ValueError:
        return None


def _require_transformers_backend() -> tuple[Any, Any, Any, int | None]:
    torch = import_optional(
        "torch",
        model_type="dia",
        install_extra="dia",
    )
    transformers = import_optional(
        "transformers",
        model_type="dia",
        install_extra="dia",
    )
    try:
        model_class = transformers.DiaForConditionalGeneration
        processor_class = transformers.AutoProcessor
    except AttributeError as exc:
        raise OptionalDependencyError(
            "Dia fine-tuning requires Transformers >= 4.53 with "
            "DiaForConditionalGeneration and AutoProcessor. Upgrade the "
            "'voicehub[dia]' environment and use the "
            "'nari-labs/Dia-1.6B-0626' checkpoint."
        ) from exc
    return (
        torch,
        model_class,
        processor_class,
        _transformers_major_version(transformers),
    )


def resolve_dia_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    """Resolve a configured dtype while keeping CPU execution reliable."""

    if not isinstance(dtype_name, str) or not dtype_name.strip():
        raise ValueError("Dia compute_dtype must be a non-empty torch dtype name.")
    normalized = dtype_name.strip().lower()
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        dtype = getattr(torch, normalized)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported Dia compute_dtype {dtype_name!r}."
        ) from exc
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def _processor_sample_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    return int(getattr(feature_extractor, "sampling_rate", 44_100))


def freeze_dia_audio_tokenizer(processor: Any) -> Any:
    """Freeze the pretrained DAC tokenizer used to construct Dia labels."""

    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is None:
        raise TypeError(
            "DiaProcessor does not expose audio_tokenizer; the official DAC "
            "tokenizer is required to prepare training labels."
        )
    requires_grad = getattr(audio_tokenizer, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
    else:
        parameters = getattr(audio_tokenizer, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad = False
    if hasattr(audio_tokenizer, "eval"):
        audio_tokenizer.eval()
    return audio_tokenizer


def _load_audio_path(path: str | PathLike[str], sample_rate: int) -> Any:
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
        str(path),
        dtype="float32",
        always_2d=False,
    )
    if int(source_rate) != sample_rate:
        raise ValueError(
            "Dia training audio must be resampled to "
            f"{sample_rate} Hz before collation; received {source_rate} Hz "
            f"from {str(path)!r}."
        )
    if audio.ndim > 1:
        audio = numpy.mean(audio, axis=-1)
    return numpy.asarray(audio, dtype=numpy.float32)


def _normalize_audio(audio: Any, *, sample_rate: int) -> Any:
    source_rate = None
    if isinstance(audio, Mapping):
        source_rate = audio.get("sampling_rate")
        if "array" in audio:
            audio = audio["array"]
        elif "path" in audio:
            audio = audio["path"]
        else:
            raise ValueError(
                "Dia audio mappings require an 'array' or 'path' field."
            )
    if source_rate is not None and int(source_rate) != sample_rate:
        raise ValueError(
            "Dia training audio must be resampled to "
            f"{sample_rate} Hz before collation; received {source_rate} Hz."
        )
    if isinstance(audio, (str, PathLike)):
        return _load_audio_path(audio, sample_rate)
    ndim = getattr(audio, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise ValueError(
            "Dia training waveforms must be mono rank-1 arrays. Convert "
            "multi-channel audio to mono before collation."
        )
    return audio


def _normalize_record(
    record: Mapping[str, Any],
    *,
    index: int,
    sample_rate: int,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError("Every Dia training record must be a mapping.")
    missing = [name for name in ("text", "audio") if name not in record]
    if missing:
        raise ValueError(
            f"Dia training record {index} is missing: {', '.join(missing)}."
        )
    text = record["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(
            f"Dia training record {index} requires non-empty text."
        )
    return {
        "text": text,
        "audio": _normalize_audio(
            record["audio"],
            sample_rate=sample_rate,
        ),
    }


@dataclass
class DiaTrainingCollator:
    """Create the official delayed decoder inputs and masked labels."""

    processor: Any
    sample_rate: int | None = None
    processor_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.sample_rate = int(
            self.sample_rate
            if self.sample_rate is not None
            else _processor_sample_rate(self.processor)
        )
        self.processor_kwargs = dict(self.processor_kwargs or {})
        collisions = sorted(
            _PROCESSOR_CONTROL_KEYS.intersection(self.processor_kwargs)
        )
        if collisions:
            raise ValueError(
                "Dia processor_kwargs cannot override training controls: "
                + ", ".join(collisions)
            )

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not _is_sequence(records) or not records:
            raise ValueError(
                "DiaTrainingCollator requires at least one training record."
            )
        normalized = [
            _normalize_record(
                record,
                index=index,
                sample_rate=self.sample_rate,
            )
            for index, record in enumerate(records)
        ]
        prepared = self.processor(
            text=[record["text"] for record in normalized],
            audio=[record["audio"] for record in normalized],
            generation=False,
            output_labels=True,
            padding=True,
            return_tensors="pt",
            **self.processor_kwargs,
        )
        if not isinstance(prepared, Mapping):
            raise TypeError("DiaProcessor must return a mapping.")
        output = dict(prepared)
        required = (
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        )
        missing = [name for name in required if name not in output]
        if missing:
            raise RuntimeError(
                "DiaProcessor did not return required training fields: "
                + ", ".join(missing)
            )
        return output

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "processor_kwargs": dict(self.processor_kwargs),
        }


class DiaSFTDataset:
    """Raw text/audio dataset using the official Dia processor at collation."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        processor: Any,
        sample_rate: int | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
    ):
        if not _is_sequence(records) or not records:
            raise ValueError("DiaSFTDataset requires at least one record.")
        self.records = tuple(records)
        self.collator = DiaTrainingCollator(
            processor,
            sample_rate=sample_rate,
            processor_kwargs=processor_kwargs,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return _normalize_record(
            self.records[index],
            index=index,
            sample_rate=self.collator.sample_rate,
        )

    def collate_fn(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return self.collator(records)

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "collator": self.collator.resume_fingerprint(),
        }


def _columnar_records(inputs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = inputs.get("records")
    if records is not None:
        if not _is_sequence(records):
            raise TypeError("Dia 'records' must be a sequence of mappings.")
        return list(records)

    text = inputs.get("text")
    audio = inputs.get("audio")
    if (
        _is_sequence(text)
        and _is_sequence(audio)
        and len(text) == len(audio)
    ):
        return [
            {
                "text": item_text,
                "audio": item_audio,
            }
            for item_text, item_audio in zip(text, audio)
        ]
    return [inputs]


def prepare_dia_training_inputs(
    processor: Any,
    inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    sample_rate: int | None = None,
    processor_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Pass through model-ready tensors or process raw Dia records."""

    if isinstance(inputs, Mapping):
        if _PREPARED_INPUT_KEYS.issubset(inputs):
            return dict(inputs)
        records = _columnar_records(inputs)
    elif _is_sequence(inputs):
        records = list(inputs)
    else:
        raise TypeError(
            "Dia training inputs must be a mapping or record sequence."
        )
    return DiaTrainingCollator(
        processor,
        sample_rate=sample_rate,
        processor_kwargs=processor_kwargs,
    )(records)


@dataclass
class DiaTrainingBackend:
    """Loaded official Dia model, processor, and native loss helpers."""

    model: Any
    processor: Any
    sample_rate: int
    transformers_major_version: int | None = None

    def prepare_for_training(self) -> "DiaTrainingBackend":
        config = getattr(self.model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        freeze_dia_audio_tokenizer(self.processor)
        return self

    def create_collator(
        self,
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> DiaTrainingCollator:
        return DiaTrainingCollator(
            self.processor,
            sample_rate=self.sample_rate,
            processor_kwargs=processor_kwargs,
        )

    def prepare_inputs(
        self,
        inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        processor_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return prepare_dia_training_inputs(
            self.processor,
            inputs,
            sample_rate=self.sample_rate,
            processor_kwargs=processor_kwargs,
        )

    @staticmethod
    def scalar_loss(outputs: Any) -> Any:
        """Return DiaForConditionalGeneration's native masked-LM loss."""

        if isinstance(outputs, Mapping):
            loss = outputs.get("loss")
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError(
                "DiaForConditionalGeneration returned no loss. Prepare the "
                "batch with generation=False and output_labels=True."
            )
        numel = getattr(loss, "numel", None)
        if callable(numel) and int(numel()) != 1:
            raise ValueError(
                "DiaForConditionalGeneration must return exactly one native "
                "loss value."
            )
        reshape = getattr(loss, "reshape", None)
        return reshape(()) if callable(reshape) else loss

    def forward_loss(
        self,
        inputs: Mapping[str, Any] | None = None,
        **model_inputs: Any,
    ) -> Any:
        if inputs is not None:
            if model_inputs:
                raise ValueError(
                    "Pass Dia model inputs either as a mapping or keywords, "
                    "not both."
                )
            model_inputs = dict(inputs)
        return self.scalar_loss(self.model(**model_inputs))

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Export a directly loadable Transformers safetensors checkpoint."""

        destination = Path(save_directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        save_model = self.model.save_pretrained
        parameters = signature(save_model).parameters.values()
        supports_safe_serialization = (
            self.transformers_major_version is None
            or (
                self.transformers_major_version < 5
                and any(
                    parameter.name == "safe_serialization"
                    or parameter.kind is Parameter.VAR_KEYWORD
                    for parameter in parameters
                )
            )
        )
        save_kwargs = (
            {
                "safe_serialization": True,
            }
            if supports_safe_serialization
            else {}
        )
        save_model(destination, **save_kwargs)
        self.processor.save_pretrained(destination)
        return destination


def load_dia_transformers_backend(
    model_name_or_path: str,
    *,
    device: str,
    compute_dtype: str = "bfloat16",
    for_training: bool = False,
) -> DiaTrainingBackend:
    """Load Dia's official Transformers implementation and processor."""

    (
        torch,
        model_class,
        processor_class,
        major_version,
    ) = _require_transformers_backend()
    dtype = resolve_dia_dtype(torch, compute_dtype, device)
    processor = processor_class.from_pretrained(model_name_or_path)
    dtype_key = (
        "dtype"
        if major_version is not None and major_version >= 5
        else "torch_dtype"
    )
    model = model_class.from_pretrained(
        model_name_or_path,
        use_safetensors=True,
        **{
            dtype_key: dtype,
        },
    )
    model.to(device=device)
    backend = DiaTrainingBackend(
        model=model,
        processor=processor,
        sample_rate=_processor_sample_rate(processor),
        transformers_major_version=major_version,
    )
    if for_training:
        backend.prepare_for_training()
    elif hasattr(model, "eval"):
        model.eval()
    return backend


class _DiaAdapterCollator:
    """Resolve the processor only after the Trainer loads model weights."""

    def __init__(self, adapter: "DiaTrainingAdapter"):
        self.adapter = adapter

    def __call__(self, records):
        self.adapter.setup()
        return self.adapter.backend.create_collator()(records)

    def resume_fingerprint(self) -> dict[str, Any]:
        config = getattr(self.adapter.model, "config", None)
        return {
            "sample_rate": getattr(config, "sample_rate", None),
            "base_model": getattr(config, "name_or_path", None),
        }


class DiaTrainingAdapter(Seq2SeqTrainingAdapter):
    """Train Dia through its official nine-codebook masked-LM loss."""

    supports_custom_recipe = True
    native_export_semantics = "inference-export"

    def __init__(self, model: Any, spec: Any):
        super().__init__(model, spec)
        self.data_collator = _DiaAdapterCollator(self)

    @property
    def backend(self) -> DiaTrainingBackend:
        backend = getattr(self.model, "training_backend", None)
        if backend is None:
            raise RuntimeError(
                "Dia's Transformers training backend is not loaded. Call "
                "adapter.setup() or model.load_for_training() first."
            )
        return backend

    def setup(self):
        super().setup()
        backend = self.backend
        if backend.model is not self.primary_model:
            raise RuntimeError(
                "Dia's training profile did not resolve the official "
                "DiaForConditionalGeneration module."
            )
        backend.prepare_for_training()
        return self

    def create_dataset(self, records, **kwargs):
        self.setup()
        processor_kwargs = dict(kwargs.pop("processor_kwargs", {}) or {})
        duplicate = sorted(set(processor_kwargs).intersection(kwargs))
        if duplicate:
            raise ValueError(
                "Dia processor options were passed twice: "
                + ", ".join(duplicate)
            )
        processor_kwargs.update(kwargs)
        dataset = DiaSFTDataset(
            records,
            processor=self.backend.processor,
            sample_rate=self.backend.sample_rate,
            processor_kwargs=processor_kwargs,
        )
        self.data_collator = dataset.collate_fn
        return dataset

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        self.backend.save_pretrained(save_directory)


__all__ = [
    "DiaSFTDataset",
    "DiaTrainingAdapter",
    "DiaTrainingBackend",
    "DiaTrainingCollator",
    "freeze_dia_audio_tokenizer",
    "load_dia_transformers_backend",
    "prepare_dia_training_inputs",
    "resolve_dia_dtype",
]
