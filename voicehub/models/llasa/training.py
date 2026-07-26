"""LLaSA completion-only codec-language-model dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.training.data import CausalTokenCollator, load_audio_tensor


class LlasaSFTDataset:
    """Create the chat-template sequence used by published LLaSA recipes."""

    TEXT_START = "<|TEXT_UNDERSTANDING_START|>"
    TEXT_END = "<|TEXT_UNDERSTANDING_END|>"
    SPEECH_START = "<|SPEECH_GENERATION_START|>"
    SPEECH_END = "<|SPEECH_GENERATION_END|>"

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        tokenizer,
        codec,
        sample_rate: int = 16_000,
        max_length: int = 2048,
    ):
        self.records = tuple(dict(record) for record in records)
        self.tokenizer = tokenizer
        self.codec = codec
        self.sample_rate = int(sample_rate)
        self.max_length = int(max_length)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        self.collate_fn = CausalTokenCollator(pad_token_id=pad_token_id, )
        if not self.records:
            raise ValueError("LlasaSFTDataset requires at least one record.")

    def __len__(self) -> int:
        return len(self.records)

    def _speech_ids(self, record: Mapping[str, Any]) -> list[int]:
        codes = record.get("audio_codes")
        if codes is None:
            audio_path = record.get("audio")
            if not audio_path:
                raise ValueError("LLaSA records require 'audio' or precomputed 'audio_codes'.")
            waveform = load_audio_tensor(
                str(audio_path),
                sample_rate=self.sample_rate,
                model_type="llasa",
                install_extra="llasa",
            )
            torch = import_optional(
                "torch",
                model_type="llasa",
                install_extra="llasa",
            )
            device = next(self.codec.parameters()).device
            with torch.inference_mode():
                codes = self.codec.encode_code(
                    input_waveform=waveform.to(device).unsqueeze(0),
                    sample_rate=self.sample_rate,
                )
        if hasattr(codes, "detach"):
            codes = codes.detach().cpu().reshape(-1).tolist()
        return [int(value) for value in codes]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "text" not in record:
            raise ValueError(f"LLaSA record {index} is missing 'text'.")
        text = (self.TEXT_START + str(record["text"]) + self.TEXT_END)
        speech = (
            self.SPEECH_START + "".join(f"<|s_{value}|>"
                                        for value in self._speech_ids(record)) + self.SPEECH_END)
        messages = [
            {
                "role": "user",
                "content": "Convert the text to speech:" + text,
            },
            {
                "role": "assistant",
                "content": speech,
            },
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = [int(value) for value in input_ids[:self.max_length]]
        speech_start_id = self.tokenizer.convert_tokens_to_ids(self.SPEECH_START)
        try:
            completion_start = input_ids.index(speech_start_id)
        except ValueError as exc:
            raise ValueError(
                "The LLaSA tokenizer did not preserve the speech-generation "
                "start token.") from exc
        labels = [-100] * completion_start + input_ids[completion_start:]
        return {
            "input_ids": input_ids,
            "labels": labels,
        }
