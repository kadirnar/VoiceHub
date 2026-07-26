"""Native Transformers fine-tuning support for Sesame CSM checkpoints.

This module intentionally has no eager PyTorch or Transformers imports.  CSM's
inference wrapper uses the original Sesame runtime, while training uses the
official Transformers conversion and its native backbone/depth-decoder loss.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.errors import OptionalDependencyError
from voicehub.training.adapters import CausalLMTrainingAdapter


_PREPARED_INPUT_KEYS = frozenset({
    "input_ids",
    "inputs_embeds",
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


def _require_transformers_backend() -> tuple[Any, Any, Any]:
    torch = import_optional(
        "torch",
        model_type="csm",
        install_extra="csm",
    )
    transformers = import_optional(
        "transformers",
        model_type="csm",
        install_extra="csm",
    )
    try:
        model_class = transformers.CsmForConditionalGeneration
        processor_class = transformers.CsmProcessor
    except AttributeError as exc:
        raise OptionalDependencyError(
            "CSM fine-tuning requires Transformers >= 4.52.1 with "
            "CsmForConditionalGeneration and CsmProcessor. Upgrade the "
            "'voicehub[csm]' environment and retry."
        ) from exc
    return torch, model_class, processor_class


def _resolve_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    try:
        dtype = getattr(torch, dtype_name)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported CSM torch_dtype {dtype_name!r}."
        ) from exc
    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def freeze_csm_codec(model: Any) -> Any:
    """Freeze Mimi and keep it in evaluation mode during CSM fine-tuning.

    Transformers performs codec encoding under ``no_grad``.  Disabling
    gradients here also keeps codec parameters out of optimizers.  A forward
    pre-hook restores evaluation mode after an outer training loop calls
    ``model.train()`` recursively.
    """

    codec = getattr(model, "codec_model", None)
    if codec is None:
        raise TypeError(
            "The Transformers CSM model does not expose codec_model; "
            "cannot freeze the Mimi tokenizer safely."
        )

    requires_grad = getattr(codec, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
    else:
        parameters = getattr(codec, "parameters", None)
        if not callable(parameters):
            raise TypeError("CSM codec_model does not expose parameters().")
        for parameter in parameters():
            parameter.requires_grad = False
    codec.eval()

    if (
        getattr(model, "_voicehub_csm_codec_eval_hook", None) is None
        and callable(getattr(model, "register_forward_pre_hook", None))
    ):

        def keep_codec_in_eval_mode(_module, _args):
            codec.eval()

        model._voicehub_csm_codec_eval_hook = model.register_forward_pre_hook(
            keep_codec_in_eval_mode,
        )
    return codec


def _processor_sample_rate(processor: Any) -> int:
    feature_extractor = getattr(processor, "feature_extractor", None)
    sample_rate = getattr(feature_extractor, "sampling_rate", 24_000)
    return int(sample_rate)


def _unwrap_audio(audio: Any, *, sample_rate: int) -> Any:
    if not isinstance(audio, Mapping):
        return audio
    source_rate = audio.get("sampling_rate")
    if source_rate is not None and int(source_rate) != sample_rate:
        raise ValueError(
            "CSM training audio must be resampled to "
            f"{sample_rate} Hz before collation; received {source_rate} Hz."
        )
    if "array" in audio:
        return audio["array"]
    if "path" in audio:
        return audio["path"]
    raise ValueError(
        "CSM audio mappings require an 'array' or 'path' field."
    )


def _normalize_content(content: Any, *, sample_rate: int) -> Any:
    if not _is_sequence(content):
        return content
    normalized = []
    for part in content:
        if not isinstance(part, Mapping):
            normalized.append(part)
            continue
        item = dict(part)
        if item.get("type") == "audio":
            if "audio" in item:
                item["audio"] = _unwrap_audio(
                    item["audio"],
                    sample_rate=sample_rate,
                )
            elif "path" in item and isinstance(item["path"], Mapping):
                item["path"] = _unwrap_audio(
                    item["path"],
                    sample_rate=sample_rate,
                )
        normalized.append(item)
    return normalized


def _normalize_conversation(
    conversation: Any,
    *,
    sample_rate: int,
) -> list[dict[str, Any]]:
    if not _is_sequence(conversation) or not conversation:
        raise ValueError(
            "A CSM conversation must be a non-empty sequence of messages."
        )
    normalized = []
    for message in conversation:
        if not isinstance(message, Mapping):
            raise TypeError("Every CSM conversation message must be a mapping.")
        if "role" not in message or "content" not in message:
            raise ValueError(
                "Every CSM conversation message requires 'role' and 'content'."
            )
        item = dict(message)
        item["role"] = str(item["role"])
        item["content"] = _normalize_content(
            item["content"],
            sample_rate=sample_rate,
        )
        normalized.append(item)
    return normalized


def _slice_grouped_audio(
    record: Mapping[str, Any],
    *,
    count: int,
    sample_rate: int,
) -> list[Any]:
    if "audios" in record:
        audios = record["audios"]
        if not _is_sequence(audios) or len(audios) != count:
            raise ValueError(
                "CSM grouped records require one 'audios' entry per text."
            )
        return [
            _unwrap_audio(audio, sample_rate=sample_rate)
            for audio in audios
        ]

    if "audio" not in record:
        raise ValueError(
            "CSM grouped records require 'audio' plus 'audio_cut_idxs', "
            "or an 'audios' sequence."
        )
    audio = _unwrap_audio(record["audio"], sample_rate=sample_rate)
    cut_indices = record.get("audio_cut_idxs")
    if cut_indices is None:
        if count == 1:
            return [audio]
        raise ValueError(
            "CSM grouped records with concatenated audio require "
            "'audio_cut_idxs'."
        )
    if not _is_sequence(cut_indices) or len(cut_indices) != count:
        raise ValueError(
            "CSM audio_cut_idxs must contain one (start, end) pair per text."
        )

    segments = []
    for bounds in cut_indices:
        if not _is_sequence(bounds) or len(bounds) != 2:
            raise ValueError(
                "Each CSM audio_cut_idxs entry must be a (start, end) pair."
            )
        start, end = (int(value) for value in bounds)
        if start < 0 or end <= start:
            raise ValueError(
                f"Invalid CSM audio slice ({start}, {end})."
            )
        segments.append(audio[start:end])
    return segments


def _conversation_from_record(
    record: Mapping[str, Any],
    *,
    sample_rate: int,
) -> list[dict[str, Any]]:
    conversation = record.get("conversation", record.get("messages"))
    if conversation is not None:
        return _normalize_conversation(
            conversation,
            sample_rate=sample_rate,
        )

    if "texts" in record or "speaker_ids" in record:
        texts = record.get("texts")
        speakers = record.get("speaker_ids")
        if not _is_sequence(texts) or not _is_sequence(speakers):
            raise ValueError(
                "CSM grouped records require sequence-valued 'texts' and "
                "'speaker_ids'."
            )
        if not texts or len(texts) != len(speakers):
            raise ValueError(
                "CSM grouped texts and speaker_ids must have equal, "
                "non-zero lengths."
            )
        audios = _slice_grouped_audio(
            record,
            count=len(texts),
            sample_rate=sample_rate,
        )
        return [
            {
                "role": str(speaker),
                "content": [
                    {
                        "type": "text",
                        "text": str(text),
                    },
                    {
                        "type": "audio",
                        "audio": audio,
                    },
                ],
            }
            for speaker, text, audio in zip(speakers, texts, audios)
        ]

    if "text" not in record or "audio" not in record:
        raise ValueError(
            "CSM training records require a conversation, grouped "
            "texts/speaker_ids/audio, or scalar text/audio fields."
        )
    speaker = record.get("speaker_id", record.get("speaker", 0))
    audio = _unwrap_audio(record["audio"], sample_rate=sample_rate)
    return [{
        "role": str(speaker),
        "content": [
            {
                "type": "text",
                "text": str(record["text"]),
            },
            {
                "type": "audio",
                "audio": audio,
            },
        ],
    }]


def _columnar_records(inputs: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = inputs.get("records")
    if records is not None:
        if not _is_sequence(records):
            raise TypeError("CSM 'records' must be a sequence of mappings.")
        return list(records)

    conversations = inputs.get("conversations")
    if conversations is not None:
        if not _is_sequence(conversations):
            raise TypeError(
                "CSM 'conversations' must be a sequence of conversations."
            )
        return [{"conversation": conversation} for conversation in conversations]

    texts = inputs.get("text")
    if not _is_sequence(texts):
        return [inputs]

    length = len(texts)
    records = []
    for index in range(length):
        record = {}
        for key in ("text", "audio", "speaker", "speaker_id"):
            value = inputs.get(key)
            if _is_sequence(value) and len(value) == length:
                record[key] = value[index]
            elif value is not None:
                record[key] = value
        records.append(record)
    return records


@dataclass
class CSMTrainingCollator:
    """Prepare the official CSM audio-frame labels for a batch.

    Records may contain a ready ``conversation``; Sesame's grouped
    ``texts``/``speaker_ids``/concatenated ``audio`` fields; or scalar
    ``text``/``audio``/``speaker_id`` fields.
    """

    processor: Any
    depth_decoder_labels_ratio: float = 1.0

    def __post_init__(self) -> None:
        ratio = float(self.depth_decoder_labels_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "depth_decoder_labels_ratio must be between 0.0 and 1.0."
            )
        self.depth_decoder_labels_ratio = ratio
        self.sample_rate = _processor_sample_rate(self.processor)

    def __call__(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not _is_sequence(records) or not records:
            raise ValueError(
                "CSMTrainingCollator requires at least one training record."
            )
        conversations = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("Every CSM training record must be a mapping.")
            conversations.append(
                _conversation_from_record(
                    record,
                    sample_rate=self.sample_rate,
                )
            )
        prepared = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            depth_decoder_labels_ratio=self.depth_decoder_labels_ratio,
        )
        if not isinstance(prepared, Mapping):
            raise TypeError(
                "CsmProcessor.apply_chat_template() must return a mapping."
            )
        output = dict(prepared)
        if "labels" not in output:
            raise RuntimeError(
                "CsmProcessor did not return labels with output_labels=True."
            )
        return output

    def resume_fingerprint(self) -> dict[str, Any]:
        return {
            "depth_decoder_labels_ratio": (
                self.depth_decoder_labels_ratio
            ),
            "sample_rate": self.sample_rate,
        }


def prepare_csm_training_inputs(
    processor: Any,
    inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    depth_decoder_labels_ratio: float = 1.0,
) -> dict[str, Any]:
    """Pass through prepared tensors or process raw CSM conversations."""

    if isinstance(inputs, Mapping):
        if (
            "labels" in inputs
            and _PREPARED_INPUT_KEYS.intersection(inputs)
        ):
            return dict(inputs)
        records = _columnar_records(inputs)
    elif _is_sequence(inputs):
        records = list(inputs)
    else:
        raise TypeError("CSM training inputs must be a mapping or record sequence.")
    return CSMTrainingCollator(
        processor,
        depth_decoder_labels_ratio=depth_decoder_labels_ratio,
    )(records)


@dataclass
class CSMTrainingBackend:
    """Loaded official CSM model, processor, and native loss helpers."""

    model: Any
    processor: Any
    sample_rate: int
    transformers_major_version: int | None = None

    def create_collator(
        self,
        *,
        depth_decoder_labels_ratio: float = 1.0,
    ) -> CSMTrainingCollator:
        return CSMTrainingCollator(
            self.processor,
            depth_decoder_labels_ratio=depth_decoder_labels_ratio,
        )

    def prepare_inputs(
        self,
        inputs: Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        depth_decoder_labels_ratio: float = 1.0,
    ) -> dict[str, Any]:
        return prepare_csm_training_inputs(
            self.processor,
            inputs,
            depth_decoder_labels_ratio=depth_decoder_labels_ratio,
        )

    @staticmethod
    def scalar_loss(outputs: Any) -> Any:
        """Return the native backbone-plus-depth-decoder loss as a scalar."""

        if isinstance(outputs, Mapping):
            loss = outputs.get("loss")
        else:
            loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError(
                "CsmForConditionalGeneration returned no loss. Prepare the "
                "batch with CsmProcessor output_labels=True."
            )
        numel = getattr(loss, "numel", None)
        if not callable(numel) or int(numel()) != 1:
            raise ValueError(
                "CsmForConditionalGeneration must return exactly one native "
                "loss value."
            )
        reshape = getattr(loss, "reshape", None)
        return reshape(()) if callable(reshape) else loss

    def forward_loss(
        self,
        inputs: Mapping[str, Any] | None = None,
        **model_inputs: Any,
    ) -> Any:
        """Run the official forward and return its exact differentiable loss."""

        if inputs is not None:
            if model_inputs:
                raise ValueError(
                    "Pass CSM model inputs either as a mapping or keywords, "
                    "not both."
                )
            model_inputs = dict(inputs)
        outputs = self.model(**model_inputs)
        return self.scalar_loss(outputs)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Export a native Transformers safetensors training checkpoint."""

        output = Path(save_directory).expanduser()
        output.mkdir(parents=True, exist_ok=True)
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
                "safe_serialization": True
            }
            if supports_safe_serialization
            else {}
        )
        save_model(output, **save_kwargs)
        self.processor.save_pretrained(output)
        return output


class _LazyCSMTrainingCollator:
    """Resolve the processor after the training backend has been loaded."""

    def __init__(self, wrapper: Any):
        self.wrapper = wrapper

    def __call__(self, records):
        backend = getattr(self.wrapper, "training_backend", None)
        if backend is None:
            raise RuntimeError(
                "CSM's training backend must be loaded before its data "
                "collator is invoked."
            )
        return backend.create_collator()(records)

    def resume_fingerprint(self) -> dict[str, Any]:
        config = getattr(self.wrapper, "config", None)
        return {
            "depth_decoder_labels_ratio": 1.0,
            "sample_rate": getattr(config, "sample_rate", None),
            "base_model": getattr(config, "name_or_path", None),
        }


class CSMTrainingAdapter(CausalLMTrainingAdapter):
    """Route VoiceHub training through Transformers' native CSM objective."""

    native_export_semantics = "inference-export"

    def __init__(self, model: Any, spec: Any):
        super().__init__(model, spec)
        self.data_collator = _LazyCSMTrainingCollator(model)

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        backend = getattr(self.model, "training_backend", None)
        if backend is None:
            raise RuntimeError(
                "CSM cannot export a native checkpoint before its "
                "Transformers training backend is loaded."
            )
        backend.save_pretrained(save_directory)


def load_csm_training_backend(
    model_name_or_path: str,
    *,
    device: str,
    torch_dtype: str = "bfloat16",
) -> CSMTrainingBackend:
    """Load the official safetensors model without touching inference source."""

    torch, model_class, processor_class = _require_transformers_backend()
    transformers = import_optional(
        "transformers",
        model_type="csm",
        install_extra="csm",
    )
    major_version = _transformers_major_version(transformers)
    dtype = _resolve_dtype(torch, torch_dtype, device)

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
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    freeze_csm_codec(model)
    return CSMTrainingBackend(
        model=model,
        processor=processor,
        sample_rate=_processor_sample_rate(processor),
        transformers_major_version=major_version,
    )


__all__ = [
    "CSMTrainingBackend",
    "CSMTrainingAdapter",
    "CSMTrainingCollator",
    "freeze_csm_codec",
    "load_csm_training_backend",
    "prepare_csm_training_inputs",
]
