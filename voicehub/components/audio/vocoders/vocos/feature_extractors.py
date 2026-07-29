from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from voicehub.components.audio.vocoders.vocos.modules import safe_log
from voicehub.processing import htk_mel_filter_bank


class FeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract features from the given audio.

        Args:
            audio (Tensor): Input audio waveform.

        Returns:
            Tensor: Extracted features of shape (B, C, L), where B is the batch size,
                    C denotes output features, and L is the sequence length.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class NativeSpectrogram(nn.Module):
    """State-compatible magnitude spectrogram used by released Vocos models."""

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim < 1:
            raise ValueError("Waveform input must have a sample dimension.")
        if not waveform.is_floating_point():
            waveform = waveform.float()
        window = self.window.to(
            device=waveform.device,
            dtype=waveform.dtype,
        )
        return torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            pad_mode=(
                "reflect"
                if waveform.shape[-1] > self.n_fft // 2
                else "constant"
            ),
            normalized=False,
            onesided=True,
            return_complex=True,
        ).abs()


class NativeMelScale(nn.Module):
    """State-compatible unnormalised HTK mel projection."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "fb",
            htk_mel_filter_bank(
                sample_rate=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
            ),
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        filters = self.fb.to(
            device=spectrogram.device,
            dtype=spectrogram.dtype,
        )
        return torch.matmul(
            spectrogram.transpose(-1, -2),
            filters,
        ).transpose(-1, -2)


class NativeMelSpectrogram(nn.Module):
    """PyTorch-only subset of ``torchaudio.transforms.MelSpectrogram``."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        center: bool,
    ) -> None:
        super().__init__()
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (sample_rate, n_fft, hop_length, n_mels)
        ):
            raise ValueError("Mel-spectrogram dimensions must be positive integers.")
        if hop_length > n_fft:
            raise ValueError("`hop_length` cannot exceed `n_fft`.")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.n_mels = n_mels
        self.center = center
        self.spectrogram = NativeSpectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=center,
        )
        self.mel_scale = NativeMelScale(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mel_scale(self.spectrogram(waveform))


class MelSpectrogramFeatures(FeatureExtractor):
    def __init__(
        self,
        sample_rate: int = 24_000,
        n_fft: int = 1_024,
        hop_length: int = 256,
        n_mels: int = 100,
        padding: str = "center",
    ):
        super().__init__()
        if padding not in {"center", "same"}:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.mel_spec = NativeMelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
        )

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        if self.padding == "same":
            pad = self.mel_spec.win_length - self.mel_spec.hop_length
            audio = torch.nn.functional.pad(audio, (pad // 2, pad // 2), mode="reflect")
        mel = self.mel_spec(audio)
        features = safe_log(mel)
        return features


class EncodecFeatures(FeatureExtractor):
    def __init__(
        self,
        encodec_model: str = "encodec_24khz",
        bandwidths: Sequence[float] = (1.5, 3.0, 6.0, 12.0),
        train_codebooks: bool = False,
        encodec: nn.Module | None = None,
    ):
        super().__init__()
        if encodec_model not in {"encodec_24khz", "encodec_48khz"}:
            raise ValueError(
                f"Unsupported encodec_model: {encodec_model}. Supported options are 'encodec_24khz' and 'encodec_48khz'."
            )
        if not bandwidths:
            raise ValueError("`bandwidths` must contain at least one target.")
        normalized_bandwidths = tuple(float(value) for value in bandwidths)
        if any(value <= 0 for value in normalized_bandwidths):
            raise ValueError("Every Encodec bandwidth must be greater than zero.")
        self.encodec_model_name = encodec_model
        if encodec is None:
            from voicehub.components.audio.codecs.encodec import EncodecModel

            factory = (
                EncodecModel.encodec_model_24khz
                if encodec_model == "encodec_24khz"
                else EncodecModel.encodec_model_48khz
            )
            encodec = factory(pretrained=False)
            self._encodec_weights_available = False
        else:
            self._encodec_weights_available = True
        self.encodec = encodec
        for param in self.encodec.parameters():
            param.requires_grad_(False)
        self.num_q = self.encodec.quantizer.get_num_quantizers_for_bandwidth(
            self.encodec.frame_rate,
            bandwidth=max(normalized_bandwidths),
        )
        layers = self.encodec.quantizer.vq.layers[:self.num_q]
        codebook_weights = torch.cat(
            [vector_quantizer.codebook for vector_quantizer in layers],
            dim=0,
        )
        self.codebook_weights = nn.Parameter(
            codebook_weights,
            requires_grad=train_codebooks,
        )
        self.bandwidths = normalized_bandwidths

    @property
    def encodec_weights_available(self) -> bool:
        """Whether raw-audio encoding has verified codec weights."""
        return self._encodec_weights_available

    def attach_encodec(self, encodec: nn.Module) -> None:
        """Attach an explicitly loaded native codec without changing Vocos embeddings."""
        if not isinstance(encodec, nn.Module):
            raise TypeError("`encodec` must be a PyTorch module.")
        if getattr(encodec, "sample_rate", None) != getattr(
            self.encodec,
            "sample_rate",
            None,
        ):
            raise ValueError("The attached Encodec sample rate does not match Vocos.")
        if getattr(encodec, "channels", None) != getattr(
            self.encodec,
            "channels",
            None,
        ):
            raise ValueError("The attached Encodec channel count does not match Vocos.")
        expected = self.encodec.state_dict()
        actual = encodec.state_dict()
        if expected.keys() != actual.keys() or any(
            expected[name].shape != actual[name].shape
            for name in expected
        ):
            raise ValueError("The attached Encodec graph is not checkpoint-compatible.")
        encodec.eval()
        for parameter in encodec.parameters():
            parameter.requires_grad_(False)
        self.encodec = encodec
        self._encodec_weights_available = True

    def mark_encodec_weights_loaded(self) -> None:
        """Mark codec tensors loaded as part of a validated Vocos checkpoint."""
        self._encodec_weights_available = True

    @torch.no_grad()
    def get_encodec_codes(self, audio: torch.Tensor) -> torch.Tensor:
        if not self._encodec_weights_available:
            raise RuntimeError(
                "Raw-audio Vocos extraction requires verified native Encodec "
                "weights. Pass `load_encodec_weights=True` or an explicit "
                "`encodec_checkpoint` to Vocos.from_pretrained(); discrete "
                "codes can be decoded without the encoder."
            )
        if audio.ndim != 2:
            raise ValueError("Encodec Vocos input must have shape [batch, samples].")
        audio = audio.unsqueeze(1)
        emb = self.encodec.encoder(audio)
        codes = self.encodec.quantizer.encode(emb, self.encodec.frame_rate, self.encodec.bandwidth)
        return codes

    def forward(self, audio: torch.Tensor, **kwargs) -> torch.Tensor:
        bandwidth_id = kwargs.get("bandwidth_id")
        if bandwidth_id is None:
            raise ValueError("The 'bandwidth_id' argument is required")
        if isinstance(bandwidth_id, torch.Tensor):
            if bandwidth_id.numel() != 1:
                raise ValueError(
                    "Raw-audio Encodec extraction requires one shared bandwidth."
                )
            bandwidth_index = int(bandwidth_id.item())
        elif isinstance(bandwidth_id, int) and not isinstance(bandwidth_id, bool):
            bandwidth_index = bandwidth_id
        else:
            raise TypeError("`bandwidth_id` must be an integer or scalar tensor.")
        if not 0 <= bandwidth_index < len(self.bandwidths):
            raise ValueError("`bandwidth_id` is outside the configured bandwidths.")
        self.encodec.eval()  # Force eval mode as Pytorch Lightning automatically sets child modules to training mode
        self.encodec.set_target_bandwidth(self.bandwidths[bandwidth_index])
        codes = self.get_encodec_codes(audio)
        # Instead of summing in the loop, it stores subsequent VQ dictionaries in a single `self.codebook_weights`
        # with offsets given by the number of bins, and finally summed in a vectorized operation.
        offsets = torch.arange(
            0, self.encodec.quantizer.bins * len(codes), self.encodec.quantizer.bins, device=audio.device
        )
        embeddings_idxs = codes + offsets.view(-1, 1, 1)
        features = torch.nn.functional.embedding(embeddings_idxs, self.codebook_weights).sum(dim=0)
        return features.transpose(1, 2)


__all__ = [
    "EncodecFeatures",
    "FeatureExtractor",
    "MelSpectrogramFeatures",
    "NativeMelScale",
    "NativeMelSpectrogram",
    "NativeSpectrogram",
]
