"""OuteTTS source-native prompt dataset for full-precision HF fine-tuning."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.training.data import CausalTokenCollator


class OuteTTSSFTDataset:
    """Tokenize V1/V2/V3 training prompts produced by OuteTTS itself."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        interface,
        completion_only: bool = True,
        whisper_model: str = "turbo",
        whisper_device: str | None = None,
    ):
        self.records = tuple(dict(record) for record in records)
        self.interface = interface
        self.prompt_processor = interface.prompt_processor
        self.tokenizer = self.prompt_processor.tokenizer
        self.completion_only = bool(completion_only)
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        self.collate_fn = CausalTokenCollator(pad_token_id=pad_token_id)
        if not self.records:
            raise ValueError("OuteTTSSFTDataset requires at least one record.")

    def __len__(self) -> int:
        return len(self.records)

    def _speaker(self, record: Mapping[str, Any], index: int) -> dict[str, Any]:
        speaker = record.get("speaker")
        if speaker is not None:
            if not isinstance(speaker, Mapping):
                raise TypeError(
                    f"OuteTTS record {index} speaker must be a mapping.")
            return copy.deepcopy(dict(speaker))
        audio_path = record.get("audio")
        if not audio_path:
            raise ValueError(
                f"OuteTTS record {index} requires 'speaker' or 'audio'.")
        return self.interface.create_speaker(
            str(audio_path),
            transcript=record.get("text"),
            whisper_model=self.whisper_model,
            whisper_device=self.whisper_device,
        )

    def _training_prompt(self, speaker: dict[str, Any], text: str) -> str:
        method = self.prompt_processor.get_training_prompt
        parameters = inspect.signature(method).parameters
        if "text" in parameters:
            return method(text, speaker)
        speaker = copy.deepcopy(speaker)
        if text:
            speaker["text"] = text
        return method(speaker)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        speaker = self._speaker(record, index)
        text = str(record.get("text", speaker.get("text", "")))
        prompt = self._training_prompt(speaker, text)
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        labels = list(input_ids)
        if self.completion_only:
            audio_start = self.prompt_processor.special_tokens.audio_start
            audio_start_ids = self.tokenizer.encode(
                audio_start,
                add_special_tokens=False,
            )
            if len(audio_start_ids) != 1:
                raise ValueError(
                    "OuteTTS audio-start marker must encode to one token.")
            try:
                completion_start = labels.index(audio_start_ids[0]) + 1
            except ValueError as exc:
                raise ValueError(
                    "OuteTTS training prompt is missing its audio-start token.") from exc
            labels[:completion_start] = [-100] * completion_start
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

