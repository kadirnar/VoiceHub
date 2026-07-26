"""Completion-only NeuTTS safetensors fine-tuning dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.training.data import CausalTokenCollator


class NeuTTSSFTDataset:
    """Build the prompt and speech-token labels from NeuTTS TRAINING.md."""

    SPEECH_START = "<|SPEECH_GENERATION_START|>"
    SPEECH_END = "<|SPEECH_GENERATION_END|>"

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        runtime,
        max_length: int = 2048,
    ):
        self.records = tuple(dict(record) for record in records)
        self.runtime = runtime
        self.tokenizer = runtime.tokenizer
        self.max_length = int(max_length)
        if self.tokenizer is None:
            raise ValueError("NeuTTS SFT requires a Hugging Face tokenizer.")
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        self.collate_fn = CausalTokenCollator(pad_token_id=pad_token_id)
        if not self.records:
            raise ValueError("NeuTTSSFTDataset requires at least one record.")

    def __len__(self) -> int:
        return len(self.records)

    def _speech_ids(self, record: Mapping[str, Any]) -> list[int]:
        codes = record.get("audio_codes")
        if codes is None:
            audio_path = record.get("audio")
            if not audio_path:
                raise ValueError("NeuTTS records require 'audio' or precomputed 'audio_codes'.")
            codes = self.runtime.encode_reference(str(audio_path))
        if hasattr(codes, "detach"):
            codes = codes.detach().cpu().reshape(-1).tolist()
        return [int(value) for value in codes]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "text" not in record:
            raise ValueError(f"NeuTTS record {index} is missing 'text'.")
        text = str(record["text"])
        if self.runtime.input_format == "phonemes":
            text = self.runtime._to_phones(text)
        prompt = (
            "user: Convert the text to speech:"
            f"<|TEXT_PROMPT_START|>{text}<|TEXT_PROMPT_END|>\n"
            f"assistant:{self.SPEECH_START}")
        completion = ("".join(f"<|speech_{value}|>" for value in self._speech_ids(record)) + self.SPEECH_END)
        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=True,
        )
        completion_ids = self.tokenizer.encode(
            completion,
            add_special_tokens=False,
        )
        input_ids = (list(prompt_ids) + list(completion_ids))[:self.max_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"NeuTTS record {index} has no completion tokens after truncation.")
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
