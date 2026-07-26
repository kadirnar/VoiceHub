"""Official-style data preparation for Orpheus codec-LM fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from voicehub.dependencies import import_optional
from voicehub.training.data import CausalTokenCollator, load_audio_tensor


class OrpheusSFTDataset:
    """Build Orpheus control-token and seven-code SNAC sequences."""

    START_HUMAN = 128259
    END_TEXT = 128009
    END_HUMAN = 128260
    START_AI = 128261
    START_SPEECH = 128257
    END_SPEECH = 128258
    END_AI = 128262
    PAD = 128263
    AUDIO_OFFSET = 128266

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        tokenizer,
        codec,
        completion_only: bool = False,
    ):
        self.records = tuple(dict(record) for record in records)
        self.tokenizer = tokenizer
        self.codec = codec
        self.completion_only = bool(completion_only)
        self.collate_fn = CausalTokenCollator(pad_token_id=self.PAD)
        if not self.records:
            raise ValueError("OrpheusSFTDataset requires at least one record.")

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _flatten_snac_codes(layers) -> list[int]:
        normalized = []
        for layer in layers:
            values = layer
            if hasattr(values, "detach"):
                values = values.detach().cpu()
            if hasattr(values, "reshape"):
                values = values.reshape(-1).tolist()
            normalized.append([int(value) for value in values])
        if len(normalized) != 3:
            raise ValueError("Orpheus SNAC codes must contain three hierarchy layers.")
        layer_1, layer_2, layer_3 = normalized
        frame_count = min(
            len(layer_1),
            len(layer_2) // 2,
            len(layer_3) // 4,
        )
        frames = []
        for index in range(frame_count):
            frames.append((
                layer_1[index],
                layer_2[2 * index],
                layer_3[4 * index],
                layer_3[4 * index + 1],
                layer_2[2 * index + 1],
                layer_3[4 * index + 2],
                layer_3[4 * index + 3],
            ))
        deduplicated = []
        previous_first = None
        for frame in frames:
            if frame[0] == previous_first:
                continue
            deduplicated.append(frame)
            previous_first = frame[0]
        offsets = (0, 4096, 8192, 12288, 16384, 20480, 24576)
        return [
            value + OrpheusSFTDataset.AUDIO_OFFSET + offsets[channel]
            for frame in deduplicated for channel, value in enumerate(frame)
        ]

    def _audio_tokens(self, record: Mapping[str, Any]) -> list[int]:
        codes = record.get("audio_codes")
        if codes is None:
            audio_path = record.get("audio")
            if not audio_path:
                raise ValueError(
                    "Orpheus records require 'audio' or precomputed 'audio_codes'.")
            waveform = load_audio_tensor(
                str(audio_path),
                sample_rate=24_000,
                model_type="orpheustts",
                install_extra="orpheustts",
            )
            torch = import_optional(
                "torch",
                model_type="orpheustts",
                install_extra="orpheustts",
            )
            device = next(self.codec.parameters()).device
            with torch.inference_mode():
                codes = self.codec.encode(
                    waveform.to(device).unsqueeze(0).unsqueeze(0))
        if isinstance(codes, Mapping):
            codes = (
                codes["layer_1"],
                codes["layer_2"],
                codes["layer_3"],
            )
        if len(codes) == 3 and any(
                hasattr(item, "shape") or isinstance(item, (tuple, list))
                for item in codes):
            return self._flatten_snac_codes(codes)
        flattened = [int(value) for value in codes]
        if flattened and max(flattened) < self.AUDIO_OFFSET:
            raise ValueError(
                "Flat Orpheus audio_codes must already include codebook offsets. "
                "Pass the three raw SNAC hierarchy layers instead.")
        return flattened

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if "text" not in record:
            raise ValueError(f"Orpheus record {index} is missing 'text'.")
        text = str(record["text"])
        voice = record.get("voice") or record.get("source")
        if voice:
            text = f"{voice}: {text}"
        text_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True,
        )
        audio_tokens = self._audio_tokens(record)
        sequence = (
            [self.START_HUMAN] + list(text_ids) +
            [self.END_TEXT, self.END_HUMAN, self.START_AI, self.START_SPEECH] +
            audio_tokens + [self.END_SPEECH, self.END_AI])
        labels = list(sequence)
        if self.completion_only:
            speech_index = sequence.index(self.START_SPEECH)
            labels[:speech_index + 1] = [-100] * (speech_index + 1)
        return {
            "input_ids": sequence,
            "labels": labels,
        }

