from __future__ import annotations

import math

import torch
from torch import nn, view_as_complex, view_as_real


class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        padding: str = "same",
    ):
        super().__init__()
        if padding not in {"center", "same"}:
            raise ValueError("Padding must be 'center' or 'same'.")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (n_fft, hop_length, win_length)
        ):
            raise ValueError("FFT, hop, and window lengths must be positive integers.")
        if win_length != n_fft:
            raise ValueError(
                "Vocos' overlap-add ISTFT requires `win_length == n_fft`."
            )
        if hop_length > win_length:
            raise ValueError("`hop_length` cannot exceed `win_length`.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if spec.ndim != 3:
            raise ValueError("ISTFT input must have shape [batch, frequency, frames].")
        if spec.shape[1] != self.n_fft // 2 + 1:
            raise ValueError(
                "ISTFT frequency dimension does not match the configured FFT size."
            )
        if not spec.is_complex():
            raise TypeError("ISTFT input must be a complex tensor.")
        window = self.window.to(device=spec.device, dtype=spec.real.dtype)

        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                window,
                center=True,
            )
        pad = (self.win_length - self.hop_length) // 2
        batch_size, _, frame_count = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * window[None, :, None]

        # Overlap and Add
        output_size = (frame_count - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0]

        # Window envelope
        window_sq = window.square().expand(1, frame_count, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[0, 0, 0]

        if pad:
            y = y[:, pad:-pad]
            window_envelope = window_envelope[pad:-pad]

        # Normalize
        if not bool((window_envelope > 1e-11).all().item()):
            raise RuntimeError("ISTFT window does not satisfy overlap-add.")
        return y.reshape(batch_size, -1) / window_envelope


class MDCT(nn.Module):
    """
    Modified Discrete Cosine Transform (MDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in {"center", "same"}:
            raise ValueError("Padding must be 'center' or 'same'.")
        if (
            isinstance(frame_len, bool)
            or not isinstance(frame_len, int)
            or frame_len <= 0
            or frame_len % 2
        ):
            raise ValueError("`frame_len` must be a positive even integer.")
        self.padding = padding
        self.frame_len = frame_len
        coefficient_count = frame_len // 2
        n0 = (coefficient_count + 1) / 2
        window = torch.sin(
            torch.pi * (torch.arange(frame_len, dtype=torch.float32) + 0.5)
            / frame_len
        )
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(-1j * torch.pi * torch.arange(frame_len) / frame_len)
        post_twiddle = torch.exp(
            -1j
            * torch.pi
            * n0
            * (torch.arange(coefficient_count) + 0.5)
            / coefficient_count
        )
        # view_as_real: NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply the Modified Discrete Cosine Transform (MDCT) to the input audio.

        Args:
            audio (Tensor): Input audio waveform of shape (B, T), where B is the batch size
                and T is the length of the audio.

        Returns:
            Tensor: MDCT coefficients of shape (B, L, N), where L is the number of output frames
                and N is the number of frequency bins.
        """
        if audio.ndim != 2:
            raise ValueError("MDCT input must have shape [batch, samples].")
        if self.padding == "center":
            audio = torch.nn.functional.pad(audio, (self.frame_len // 2, self.frame_len // 2))
        else:
            # hop_length is 1/2 frame_len
            audio = torch.nn.functional.pad(audio, (self.frame_len // 4, self.frame_len // 4))

        x = audio.unfold(-1, self.frame_len, self.frame_len // 2)
        coefficient_count = self.frame_len // 2
        window = self.window.to(device=x.device, dtype=x.dtype)
        pre_twiddle = view_as_complex(
            self.pre_twiddle.to(device=x.device, dtype=x.dtype)
        )
        post_twiddle = view_as_complex(
            self.post_twiddle.to(device=x.device, dtype=x.dtype)
        )
        x = x * window
        transformed = torch.fft.fft(x * pre_twiddle, dim=-1)[
            ..., :coefficient_count
        ]
        transformed = (
            transformed
            * post_twiddle
            * math.sqrt(1.0 / coefficient_count)
        )
        return transformed.real * math.sqrt(2.0)


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in {"center", "same"}:
            raise ValueError("Padding must be 'center' or 'same'.")
        if (
            isinstance(frame_len, bool)
            or not isinstance(frame_len, int)
            or frame_len <= 0
            or frame_len % 2
        ):
            raise ValueError("`frame_len` must be a positive even integer.")
        self.padding = padding
        self.frame_len = frame_len
        coefficient_count = frame_len // 2
        n0 = (coefficient_count + 1) / 2
        window = torch.sin(
            torch.pi * (torch.arange(frame_len, dtype=torch.float32) + 0.5)
            / frame_len
        )
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(
            1j
            * torch.pi
            * n0
            * torch.arange(coefficient_count * 2)
            / coefficient_count
        )
        post_twiddle = torch.exp(
            1j
            * torch.pi
            * (torch.arange(coefficient_count * 2) + n0)
            / (coefficient_count * 2)
        )
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, L, N), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        if X.ndim != 3:
            raise ValueError("IMDCT input must have shape [batch, frames, bins].")
        batch_size, frame_count, coefficient_count = X.shape
        if coefficient_count * 2 != self.frame_len:
            raise ValueError(
                "IMDCT coefficient dimension does not match `frame_len`."
            )
        mirrored = -torch.flip(X, dims=(-1,))
        spectrum = torch.cat((X, mirrored), dim=-1)
        pre_twiddle = view_as_complex(
            self.pre_twiddle.to(device=X.device, dtype=X.dtype)
        )
        post_twiddle = view_as_complex(
            self.post_twiddle.to(device=X.device, dtype=X.dtype)
        )
        y = torch.fft.ifft(spectrum * pre_twiddle, dim=-1)
        y = (
            (y * post_twiddle).real
            * math.sqrt(coefficient_count)
            * math.sqrt(2.0)
        )
        window = self.window.to(device=X.device, dtype=X.dtype)
        result = y * window
        output_size = (1, (frame_count + 1) * coefficient_count)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        else:
            pad = self.frame_len // 4
        if pad:
            audio = audio[:, pad:-pad]
        return audio.reshape(batch_size, -1)
