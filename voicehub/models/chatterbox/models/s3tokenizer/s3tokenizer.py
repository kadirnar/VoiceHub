from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from voicehub.models.chatterbox.models.s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2
from voicehub.models.chatterbox.models.s3tokenizer.native_utils import padding
from voicehub.models.chatterbox.native_audio import s3tokenizer_log_mel, slaney_mel_filter_bank

# Sampling rate of the inputs to S3TokenizerV2
S3_SR = 16_000
S3_HOP = 160  # 100 frames/sec
S3_TOKEN_HOP = 640  # 25 tokens/sec
S3_TOKEN_RATE = 25
SPEECH_VOCAB_SIZE = 6561


class S3Tokenizer(S3TokenizerV2):
    """s3tokenizer.S3TokenizerV2 with the following changes:

    - a more integrated `forward`
    - compute `log_mel_spectrogram` using `_mel_filters` and `window` in `register_buffers`
    """

    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(
        self,
        name: str = "speech_tokenizer_v2_25hz",
        config: ModelConfig | None = None,
    ):
        config = config or ModelConfig()
        super().__init__(name)

        self.n_fft = 400
        mel_filters = slaney_mel_filter_bank(
            sample_rate=S3_SR,
            n_fft=self.n_fft,
            n_mels=config.n_mels,
        )
        self.register_buffer(
            "_mel_filters",
            mel_filters,
        )

        self.register_buffer(
            "window",
            torch.hann_window(self.n_fft),
            persistent=False,
        )

    def pad(self, wavs, sr) -> list[torch.Tensor]:
        """Given a list of wavs with the same `sample_rate`, pad them so that
        the length is multiple of 40ms (S3 runs at 25 token/sec)."""
        processed_wavs = []
        for wav in wavs:
            wav = torch.as_tensor(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.dim() != 2:
                raise ValueError("Each waveform must have shape [samples] or [channels, samples].")

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = math.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav, (0, intended_wav_len - wav.shape[-1]), mode="constant", value=0)
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        """Prepare a list of audios for s3tokenizer processing."""
        processed_wavs = []
        for wav in wavs:
            wav = torch.as_tensor(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if wav.dim() != 2:
                raise ValueError("Each waveform must have shape [samples] or [channels, samples].")

            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(
        self,
        wavs: torch.Tensor | list[torch.Tensor],
        accelerator: Any = None,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.LongTensor]:
        """
        NOTE: mel-spec has a hop size of 160 points (100 frame/sec).
        FIXME: this class inherits `nn.Module` but doesn't accept `torch.Tensor` and handles a list of wavs one by one, which is unexpected.

        Args
        ----
        - `wavs`: 16 kHz speech audio
        - `max_len` max length to truncate the output sequence to (25 token/sec).
        NOTE: please pad the waveform if longer sequence is needed.
        """
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)  # [B=1, F, T]
            if max_len is not None:
                mel = mel[..., :max_len * 4]  # num_mel_frames = 4 * num_tokens
            mels.append(mel.squeeze(0))

        mels, mel_lens = padding(mels)
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        padding: int = 0,
    ):
        """
        Compute the log-Mel spectrogram of.

        Parameters
        ----------
        audio: torch.Tensor, shape = (*)
            The path to audio or either a NumPy array or Tensor containing the
            audio waveform in 16 kHz

        padding: int
            Number of zero samples to pad to the right

        Returns
        -------
        torch.Tensor, shape = (128, n_frames)
            A Tensor that contains the Mel spectrogram
        """
        return s3tokenizer_log_mel(
            torch.as_tensor(audio, device=self.device),
            mel_filters=self._mel_filters,
            window=self.window,
            n_fft=self.n_fft,
            hop_length=S3_HOP,
            padding=padding,
        )
