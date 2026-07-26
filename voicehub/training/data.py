"""Reusable data primitives for codec-language-model fine-tuning."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from voicehub.dependencies import import_optional


def load_audio_tensor(
    path: str,
    *,
    sample_rate: int,
    model_type: str,
    install_extra: str,
):
    """Load a mono float waveform and resample it when necessary."""
    numpy = import_optional(
        "numpy",
        model_type=model_type,
        install_extra=install_extra,
    )
    soundfile = import_optional(
        "soundfile",
        model_type=model_type,
        install_extra=install_extra,
    )
    torch = import_optional(
        "torch",
        model_type=model_type,
        install_extra=install_extra,
    )
    audio, source_rate = soundfile.read(
        path,
        dtype="float32",
        always_2d=False,
    )
    if audio.ndim > 1:
        audio = numpy.mean(audio, axis=-1)
    waveform = torch.from_numpy(numpy.asarray(audio, dtype=numpy.float32))
    if int(source_rate) != int(sample_rate):
        torchaudio = import_optional(
            "torchaudio",
            model_type=model_type,
            install_extra=install_extra,
        )
        waveform = torchaudio.functional.resample(
            waveform,
            int(source_rate),
            int(sample_rate),
        )
    return waveform.contiguous()


class CausalTokenCollator:
    """Pad token sequences while keeping ignored labels at ``-100``."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        label_pad_token_id: int = -100,
        padding_side: str = "right",
    ):
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'.")
        self.pad_token_id = int(pad_token_id)
        self.label_pad_token_id = int(label_pad_token_id)
        self.padding_side = padding_side

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty token batch.")
        torch = import_optional(
            "torch",
            model_type="Trainer",
            install_extra="training",
        )
        sequences = [
            torch.as_tensor(feature["input_ids"], dtype=torch.long).reshape(-1)
            for feature in features
        ]
        labels = [
            torch.as_tensor(
                feature.get("labels", feature["input_ids"]),
                dtype=torch.long,
            ).reshape(-1)
            for feature in features
        ]
        if any(ids.shape != target.shape for ids, target in zip(sequences, labels)):
            raise ValueError("Every causal token example must align input_ids and labels.")
        max_length = max(int(sequence.numel()) for sequence in sequences)
        input_ids = torch.full(
            (len(features), max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (len(features), max_length),
            dtype=torch.long,
        )
        padded_labels = torch.full(
            (len(features), max_length),
            self.label_pad_token_id,
            dtype=torch.long,
        )
        for row, (sequence, target) in enumerate(zip(sequences, labels)):
            length = int(sequence.numel())
            target_slice = (
                slice(max_length - length, max_length)
                if self.padding_side == "left" else slice(0, length))
            input_ids[row, target_slice] = sequence
            attention_mask[row, target_slice] = 1
            padded_labels[row, target_slice] = target
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }

    def resume_fingerprint(self) -> dict[str, int | str]:
        """Identify padding behavior that changes exact-resume batches."""
        return {
            "pad_token_id": self.pad_token_id,
            "label_pad_token_id": self.label_pad_token_id,
            "padding_side": self.padding_side,
        }
