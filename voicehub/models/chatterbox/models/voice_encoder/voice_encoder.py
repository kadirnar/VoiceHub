"""Native LSTM speaker encoder used by the English Chatterbox checkpoint."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import accumulate

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from voicehub.models.chatterbox.native_audio import as_mono_waveform, resample_batch, trim_silence

from .config import VoiceEncConfig
from .melspec import melspectrogram


def pack(
    arrays: Sequence[Tensor],
    seq_len: int | None = None,
    pad_value: float = 0.0,
) -> Tensor:
    """Right-pad variable-length ``[time, ...]`` tensors into one batch."""
    if not arrays:
        raise ValueError("At least one array is required.")
    tensors = [torch.as_tensor(array) for array in arrays]
    required = max(tensor.shape[0] for tensor in tensors)
    seq_len = required if seq_len is None else seq_len
    if seq_len < required:
        raise ValueError("seq_len is shorter than an input sequence.")
    shape = (len(tensors), seq_len, *tensors[0].shape[1:])
    packed = tensors[0].new_full(shape, pad_value)
    for index, tensor in enumerate(tensors):
        packed[index, :tensor.shape[0]] = tensor.to(packed.device)
    return packed


def get_num_wins(
    n_frames: int,
    step: int,
    min_coverage: float,
    hp: VoiceEncConfig,
) -> tuple[int, int]:
    if n_frames <= 0:
        raise ValueError("A voice encoder utterance must contain at least one mel frame.")
    window = hp.ve_partial_frames
    count, remainder = divmod(max(n_frames - window + step, 0), step)
    if count == 0 or (remainder + (window - step)) / window >= min_coverage:
        count += 1
    return count, window + step * (count - 1)


def get_frame_step(
    overlap: float,
    rate: float | None,
    hp: VoiceEncConfig,
) -> int:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    if rate is None:
        frame_step = round(hp.ve_partial_frames * (1 - overlap))
    else:
        frame_step = round((hp.sample_rate / rate) / hp.ve_partial_frames)
    if not 0 < frame_step <= hp.ve_partial_frames:
        raise ValueError("The requested partial rate produces an invalid frame step.")
    return frame_step


def stride_as_partials(
    mel: Tensor,
    hp: VoiceEncConfig,
    overlap: float = 0.5,
    rate: float | None = None,
    min_coverage: float = 0.8,
) -> Tensor:
    """Create overlapping ``[partials, frames, mels]`` views."""
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1].")
    values = torch.as_tensor(mel, dtype=torch.float32).contiguous()
    if values.ndim != 2 or values.shape[1] != hp.num_mels:
        raise ValueError(f"mel must have shape [frames, {hp.num_mels}].")
    frame_step = get_frame_step(overlap, rate, hp)
    count, target = get_num_wins(values.shape[0], frame_step, min_coverage, hp)
    if target > values.shape[0]:
        values = F.pad(values, (0, 0, 0, target - values.shape[0]))
    else:
        values = values[:target]
    partials = values.unfold(0, hp.ve_partial_frames, frame_step)
    partials = partials.permute(0, 2, 1).contiguous()
    return partials[:count]


class VoiceEncoder(nn.Module):
    """Produce L2-normalized speaker embeddings from waveform or mel inputs."""

    def __init__(self, hp: VoiceEncConfig | None = None):
        super().__init__()
        self.hp = hp or VoiceEncConfig()
        self.lstm = nn.LSTM(
            self.hp.num_mels,
            self.hp.ve_hidden_size,
            num_layers=3,
            batch_first=True,
        )
        if self.hp.flatten_lstm_params:
            self.lstm.flatten_parameters()
        self.proj = nn.Linear(self.hp.ve_hidden_size, self.hp.speaker_embed_size)
        self.similarity_weight = nn.Parameter(torch.tensor([10.0]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.0]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, mels: Tensor) -> Tensor:
        if self.hp.normalized_mels and (mels.min() < 0 or mels.max() > 1):
            raise ValueError(f"Normalized mels must be in [0, 1], got [{mels.min()}, {mels.max()}].")
        _, (hidden, _) = self.lstm(mels)
        embeddings = self.proj(hidden[-1])
        if self.hp.ve_final_relu:
            embeddings = F.relu(embeddings)
        return F.normalize(embeddings, p=2, dim=1)

    def inference(
        self,
        mels: Tensor,
        mel_lens: Sequence[int] | Tensor,
        overlap: float = 0.5,
        rate: float | None = None,
        min_coverage: float = 0.8,
        batch_size: int | None = None,
    ) -> Tensor:
        lengths = ([int(value) for value in mel_lens.tolist()]
                   if torch.is_tensor(mel_lens) else [int(value) for value in mel_lens])
        if len(lengths) != mels.shape[0]:
            raise ValueError("mel_lens must contain one length per batch item.")
        frame_step = get_frame_step(overlap, rate, self.hp)
        windows = [get_num_wins(length, frame_step, min_coverage, self.hp) for length in lengths]
        counts = [item[0] for item in windows]
        target_lengths = [item[1] for item in windows]
        missing = max(target_lengths) - mels.shape[1]
        if missing > 0:
            mels = F.pad(mels, (0, 0, 0, missing))
        partials = torch.stack([
            mel[index * frame_step:index * frame_step + self.hp.ve_partial_frames]
            for mel, count in zip(mels, counts) for index in range(count)
        ])
        chunk_size = batch_size or partials.shape[0]
        chunk_count = math.ceil(partials.shape[0] / chunk_size)
        partial_embeddings = torch.cat(
            [self(chunk) for chunk in partials.chunk(chunk_count)],
            dim=0,
        )
        boundaries = [0, *accumulate(counts)]
        utterance_embeddings = torch.stack([
            partial_embeddings[start:end].mean(dim=0) for start, end in zip(boundaries[:-1], boundaries[1:])
        ])
        return F.normalize(utterance_embeddings, p=2, dim=1)

    @staticmethod
    def utt_to_spk_embed(utterance_embeddings: Tensor) -> Tensor:
        values = torch.as_tensor(utterance_embeddings)
        if values.ndim != 2:
            raise ValueError("Utterance embeddings must have shape [utterances, embedding].")
        return F.normalize(values.mean(dim=0), p=2, dim=0)

    @staticmethod
    def voice_similarity(embeds_x: Tensor, embeds_y: Tensor) -> Tensor:
        left = torch.as_tensor(embeds_x)
        right = torch.as_tensor(embeds_y)
        if left.ndim != 1:
            left = VoiceEncoder.utt_to_spk_embed(left)
        if right.ndim != 1:
            right = VoiceEncoder.utt_to_spk_embed(right)
        return left @ right

    def embeds_from_mels(
        self,
        mels: Tensor | Sequence[Tensor],
        mel_lens: Sequence[int] | Tensor | None = None,
        as_spk: bool = False,
        batch_size: int = 32,
        **kwargs,
    ) -> Tensor:
        if isinstance(mels, (list, tuple)):
            values = [torch.as_tensor(mel, dtype=torch.float32) for mel in mels]
            if any(mel.ndim != 2 or mel.shape[1] != self.hp.num_mels for mel in values):
                raise ValueError(f"Mels must have shape [frames, {self.hp.num_mels}].")
            mel_lens = [mel.shape[0] for mel in values]
            batched = pack(values)
        else:
            batched = torch.as_tensor(mels)
            if mel_lens is None:
                mel_lens = [batched.shape[1]] * batched.shape[0]
        with torch.inference_mode():
            utterance_embeddings = self.inference(
                batched.to(self.device),
                mel_lens,
                batch_size=batch_size,
                **kwargs,
            )
        return (self.utt_to_spk_embed(utterance_embeddings) if as_spk else utterance_embeddings)

    def embeds_from_wavs(
        self,
        wavs: Sequence[Tensor],
        sample_rate: int,
        as_spk: bool = False,
        batch_size: int = 32,
        trim_top_db: float | None = 20,
        **kwargs,
    ) -> Tensor:
        values = [as_mono_waveform(waveform) for waveform in wavs]
        if sample_rate != self.hp.sample_rate:
            values = resample_batch(
                values,
                source_rate=sample_rate,
                target_rate=self.hp.sample_rate,
            )
        if trim_top_db is not None:
            values = [trim_silence(waveform, top_db=trim_top_db) for waveform in values]
        if "rate" not in kwargs:
            kwargs["rate"] = 1.3
        mels = [melspectrogram(waveform, self.hp).transpose(0, 1) for waveform in values]
        return self.embeds_from_mels(
            mels,
            as_spk=as_spk,
            batch_size=batch_size,
            **kwargs,
        )


__all__ = [
    "VoiceEncConfig",
    "VoiceEncoder",
    "get_frame_step",
    "get_num_wins",
    "pack",
    "stride_as_partials",
]
