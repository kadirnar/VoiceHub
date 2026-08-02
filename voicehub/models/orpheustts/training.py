"""Official-style data preparation for Orpheus codec-LM fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.models.orpheustts.protocol import (
    AUDIO_TOKEN_OFFSET,
    END_AI_TOKEN_ID,
    END_HUMAN_TOKEN_ID,
    END_SPEECH_TOKEN_ID,
    END_TEXT_TOKEN_ID,
    PAD_TOKEN_ID,
    START_AI_TOKEN_ID,
    START_HUMAN_TOKEN_ID,
    START_SPEECH_TOKEN_ID,
    interleave_snac_codes,
    normalize_orpheus_audio_tokens,
)
from voicehub.training.data import CausalTokenCollator, load_audio_tensor
from voicehub.training.recipes import CodecCausalLMTrainingAdapter


class OrpheusTrainingAdapter(CodecCausalLMTrainingAdapter):
    """Fine-tune and export the complete VoiceHub-native Orpheus runtime."""

    native_export_semantics = "inference-export"

    def save_pretrained(self, save_directory) -> None:
        self.setup()
        export = getattr(self.model, "export_native_pretrained", None)
        if not callable(export):
            raise TypeError("Native Orpheus training requires a wrapper with "
                            "export_native_pretrained().")
        export(save_directory)


class OrpheusSFTDataset:
    """Build Orpheus control-token and seven-code SNAC sequences."""

    START_HUMAN = START_HUMAN_TOKEN_ID
    END_TEXT = END_TEXT_TOKEN_ID
    END_HUMAN = END_HUMAN_TOKEN_ID
    START_AI = START_AI_TOKEN_ID
    START_SPEECH = START_SPEECH_TOKEN_ID
    END_SPEECH = END_SPEECH_TOKEN_ID
    END_AI = END_AI_TOKEN_ID
    PAD = PAD_TOKEN_ID
    AUDIO_OFFSET = AUDIO_TOKEN_OFFSET

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        tokenizer,
        codec,
        completion_only: bool = False,
        max_length: int | None = None,
    ):
        self.records = tuple(dict(record) for record in records)
        self.tokenizer = tokenizer
        self.codec = codec
        if not isinstance(completion_only, bool):
            raise TypeError("`completion_only` must be a boolean.")
        if max_length is not None:
            if (isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2):
                raise ValueError("`max_length` must be an integer of at least two or None.")
        self.completion_only = completion_only
        self.max_length = max_length
        self.collate_fn = CausalTokenCollator(pad_token_id=self.PAD)
        if not self.records:
            raise ValueError("OrpheusSFTDataset requires at least one record.")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _flatten_snac_codes(layers) -> list[int]:
        return interleave_snac_codes(layers)

    def _audio_tokens(self, record: Mapping[str, Any]) -> list[int]:
        codes = record.get("audio_codes")
        if codes is None:
            audio_path = record.get("audio")
            if not audio_path:
                raise ValueError("Orpheus records require 'audio' or precomputed 'audio_codes'.")
            waveform = load_audio_tensor(
                str(audio_path),
                sample_rate=int(getattr(self.codec, "sampling_rate", 24_000)),
                model_type="orpheustts",
                install_extra="training",
            )
            torch = import_optional(
                "torch",
                model_type="orpheustts",
                install_extra="training",
            )
            device = next(self.codec.parameters()).device
            with torch.inference_mode():
                codes = self.codec.encode(waveform.to(device).unsqueeze(0).unsqueeze(0))
        if isinstance(codes, Mapping):
            codes = (
                codes["layer_1"],
                codes["layer_2"],
                codes["layer_3"],
            )
        if len(codes) == 3 and any(hasattr(item, "shape") or isinstance(item, (tuple, list))
                                   for item in codes):
            return self._flatten_snac_codes(codes)
        flattened = [int(value) for value in codes]
        if flattened and max(flattened) < self.AUDIO_OFFSET:
            raise ValueError(
                "Flat Orpheus audio_codes must already include codebook offsets. "
                "Pass the three raw SNAC hierarchy layers instead.")
        normalize_orpheus_audio_tokens(flattened)
        return flattened

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "text" not in record:
            raise ValueError(f"Orpheus record {index} is missing 'text'.")
        text = str(record["text"])
        voice = record.get("voice")
        if voice:
            text = f"{voice}: {text}"
        text_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
        )
        audio_tokens = self._audio_tokens(record)
        sequence = ([self.START_HUMAN] + list(text_ids) +
                    [self.END_TEXT, self.END_HUMAN, self.START_AI, self.START_SPEECH] + audio_tokens +
                    [self.END_SPEECH, self.END_AI])
        if self.max_length is not None and len(sequence) > self.max_length:
            raise ValueError(
                f"Orpheus record {index} produces {len(sequence)} tokens, "
                f"exceeding max_length={self.max_length}. Pre-segment the "
                "record instead of truncating a SNAC frame.")
        labels = list(sequence)
        if self.completion_only:
            speech_index = sequence.index(self.START_SPEECH)
            labels[:speech_index + 1] = [-100] * (speech_index + 1)
        return {
            "input_ids": sequence,
            "labels": labels,
        }


def build_training_dataset(model, records, **kwargs) -> OrpheusSFTDataset:
    """Build the source-native dataset declared by the training registry."""
    return OrpheusSFTDataset(
        records,
        tokenizer=model.tokenizer,
        codec=model.codec,
        completion_only=bool(kwargs.get("completion_only", False)),
        max_length=kwargs.get("max_length"),
    )


__all__ = [
    "OrpheusTrainingAdapter",
    "OrpheusSFTDataset",
    "build_training_dataset",
]
