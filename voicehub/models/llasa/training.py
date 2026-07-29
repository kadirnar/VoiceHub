"""Source-faithful native data preparation for LLaSA fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from voicehub.models.llasa.tokenization_llasa import (
    LLASA_SPEECH_CODEBOOK_SIZE,
    SPEECH_GENERATION_END,
    SPEECH_GENERATION_START,
    TEXT_UNDERSTANDING_END,
    TEXT_UNDERSTANDING_START,
)
from voicehub.processing.waveform import load_native_audio
from voicehub.training.data import CausalTokenCollator
from voicehub.training.recipes import CodecCausalLMTrainingAdapter

LLASA_TRAINING_SOURCE_REVISION = ("479acd5277220f78a72093f63755c0892838d0c5")


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than zero.")
    return value


class LlasaSFTDataset:
    """Build the authors' completion-only codec-language-model examples.

    Each record requires ``text`` and either precomputed ``audio_codes`` or
    audio accepted by :func:`voicehub.processing.waveform.load_native_audio`.
    The latter may be a PCM WAVE path, tensor/list plus ``sampling_rate``, or
    a mapping containing both samples and a rate. XCodec2 remains frozen, as
    in the published online fine-tuning recipe.
    """

    TEXT_START = TEXT_UNDERSTANDING_START
    TEXT_END = TEXT_UNDERSTANDING_END
    SPEECH_START = SPEECH_GENERATION_START
    SPEECH_END = SPEECH_GENERATION_END

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        tokenizer,
        codec=None,
        sample_rate: int = 16_000,
        max_length: int = 2_048,
        truncate: bool = True,
    ) -> None:
        is_sequence = isinstance(records, Sequence)
        is_text_value = isinstance(records, (str, bytes))
        if not is_sequence or is_text_value:
            raise TypeError("`records` must be a sequence of mappings.")
        normalized_records = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise TypeError(f"LLaSA record {index} must be a mapping.")
            normalized_records.append(dict(record))
        if not normalized_records:
            raise ValueError("LlasaSFTDataset requires at least one record.")
        if not isinstance(truncate, bool):
            raise TypeError("`truncate` must be a boolean.")

        self.records = tuple(normalized_records)
        self.tokenizer = tokenizer
        self.codec = codec
        self.sample_rate = _positive_integer(
            sample_rate,
            name="sample_rate",
        )
        if self.sample_rate != 16_000:
            raise ValueError("Published LLaSA XCodec2 recipes require 16 kHz audio.")
        self.max_length = _positive_integer(max_length, name="max_length")
        if self.max_length < 2:
            raise ValueError("`max_length` must be at least two.")
        self.truncate = truncate

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("The LLaSA tokenizer must define a pad or EOS ID.")
        self.collate_fn = CausalTokenCollator(pad_token_id=int(pad_token_id), )
        if codec is not None:
            if hasattr(codec, "eval"):
                codec.eval()
            parameters = getattr(codec, "parameters", None)
            if callable(parameters):
                for parameter in parameters():
                    parameter.requires_grad_(False)

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _codec_device(codec) -> torch.device:
        parameters = getattr(codec, "parameters", None)
        if callable(parameters):
            try:
                return next(parameters()).device
            except StopIteration:
                pass
        return torch.device("cpu")

    @staticmethod
    def _normalize_codes(codes: Any) -> list[int]:
        if isinstance(codes, Tensor):
            values = codes.detach()
        else:
            try:
                values = torch.as_tensor(codes)
            except (TypeError, ValueError, RuntimeError) as error:
                raise TypeError("`audio_codes` must be an integer tensor or sequence.") from error
        if values.numel() == 0:
            raise ValueError("`audio_codes` cannot be empty.")
        if (values.dtype == torch.bool or values.is_floating_point() or values.is_complex()):
            raise TypeError("`audio_codes` must use an integer dtype.")
        if values.ndim > 1 and any(size != 1 for size in values.shape[:-1]):
            raise ValueError(
                "Each LLaSA record must contain one code sequence, optionally "
                "shaped [1, frames] or [1, 1, frames].")
        flattened = values.reshape(-1).to(dtype=torch.long, device="cpu")
        if bool((flattened < 0).any()) or bool((flattened >= LLASA_SPEECH_CODEBOOK_SIZE).any()):
            raise ValueError("LLaSA XCodec2 codes must be in [0, 65535].")
        return [int(value) for value in flattened.tolist()]

    def _load_waveform(self, record: Mapping[str, Any]) -> Tensor:
        if "audio_values" in record:
            audio = record["audio_values"]
        elif "waveform" in record:
            audio = record["waveform"]
        elif "audio" in record:
            audio = record["audio"]
        else:
            raise ValueError(
                "LLaSA records require `audio_codes`, `audio`, `waveform`, "
                "or `audio_values`.")
        source_rate = record.get(
            "sampling_rate",
            record.get("sample_rate"),
        )
        if isinstance(audio, (str, Path, Mapping)):
            # WAVE paths and audio mappings carry their own rate. An explicit
            # record-level value is still checked when present.
            native = load_native_audio(
                audio,
                sampling_rate=source_rate,
                target_sampling_rate=self.sample_rate,
            )
        else:
            native = load_native_audio(
                audio,
                sampling_rate=(self.sample_rate if source_rate is None else source_rate),
                target_sampling_rate=self.sample_rate,
            )
        return native.waveform

    def _speech_ids(self, record: Mapping[str, Any]) -> list[int]:
        codes = record.get("audio_codes")
        if codes is not None:
            return self._normalize_codes(codes)
        if self.codec is None:
            raise ValueError(
                "Raw-audio LLaSA records require a loaded XCodec2 codec. "
                "Provide precomputed `audio_codes` for offline preparation.")
        waveform = self._load_waveform(record)
        device = self._codec_device(self.codec)
        self.codec.eval()
        with torch.inference_mode():
            encoded = self.codec.encode_code(
                input_waveform=waveform.to(device).unsqueeze(0),
                sample_rate=self.sample_rate,
            )
        return self._normalize_codes(encoded)

    @staticmethod
    def _assistant_prefix(record: Mapping[str, Any]) -> str:
        prefix = record.get(
            "assistant_prefix",
            record.get("condition"),
        )
        if prefix is None and record.get("speaker") is not None:
            prefix = f"Speaker {record['speaker']}"
        if prefix is None:
            return ""
        if not isinstance(prefix, str):
            raise TypeError(
                "LLaSA `assistant_prefix`, `condition`, and `speaker` "
                "conditioning must resolve to text.")
        return prefix.strip()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "text" not in record:
            raise ValueError(f"LLaSA record {index} is missing `text`.")
        text = record["text"]
        if not isinstance(text, str):
            raise TypeError(f"LLaSA record {index} `text` must be a string.")
        speech_ids = self._speech_ids(record)
        assistant = (
            self._assistant_prefix(record) + self.SPEECH_START +
            "".join(f"<|s_{value}|>" for value in speech_ids) + self.SPEECH_END)
        messages = [
            {
                "role": "user",
                "content": ("Convert the text to speech:" + self.TEXT_START + text + self.TEXT_END),
            },
            {
                "role": "assistant",
                "content": assistant,
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(input_ids, Tensor):
            input_ids = input_ids.detach().cpu().reshape(-1).tolist()
        input_ids = [int(value) for value in input_ids]
        if len(input_ids) > self.max_length:
            if not self.truncate:
                raise ValueError(
                    f"LLaSA record {index} produces {len(input_ids)} tokens, "
                    f"exceeding max_length={self.max_length}.")
            input_ids = input_ids[:self.max_length]
        speech_start_id = self.tokenizer.convert_tokens_to_ids(self.SPEECH_START)
        try:
            completion_start = input_ids.index(speech_start_id)
        except ValueError as error:
            raise ValueError(
                "The LLaSA sequence limit removed the speech-generation "
                "start token; shorten or pre-segment this record.") from error
        labels = [-100] * completion_start + input_ids[completion_start:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class LlasaTrainingAdapter(CodecCausalLMTrainingAdapter):
    """Fine-tune and export the complete native LLaSA runtime."""

    native_export_semantics = "inference-export"

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native LLaSA training requires a wrapper with "
                            "export_native_pretrained().")
        export(save_directory)


__all__ = [
    "LLASA_TRAINING_SOURCE_REVISION",
    "LlasaSFTDataset",
    "LlasaTrainingAdapter",
]
