"""Native Encodec comparison adapter used by DAC evaluation tools."""

from __future__ import annotations

from pathlib import Path

import torch

from voicehub.components.audio.codecs._compat import AudioSignal, BaseModel
from voicehub.components.audio.codecs.encodec import EncodecModel


class Encodec(BaseModel):
    """Expose VoiceHub-native Encodec through DAC's comparison contract."""

    def __init__(
        self,
        sample_rate: int = 24_000,
        bandwidth: float = 24.0,
        *,
        pretrained: bool = False,
        repository: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        trust_official_pickle: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate == 24_000:
            factory = EncodecModel.encodec_model_24khz
        elif sample_rate == 48_000:
            factory = EncodecModel.encodec_model_48khz
        else:
            raise ValueError("Encodec comparison supports 24 kHz or 48 kHz.")
        self.model = factory(
            pretrained=pretrained,
            repository=repository,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_official_pickle=trust_official_pickle,
        )
        self.model.set_target_bandwidth(bandwidth)
        self.sample_rate = sample_rate

    def forward(
        self,
        audio_data: torch.Tensor,
        sample_rate: int = 44_100,
        n_quantizers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Round-trip audio while preserving the caller's sample rate."""
        previous_bandwidth = self.model.bandwidth
        if n_quantizers is not None:
            if (
                isinstance(n_quantizers, bool)
                or not isinstance(n_quantizers, int)
                or n_quantizers <= 0
            ):
                raise ValueError("`n_quantizers` must be a positive integer or None.")
            bandwidth = (
                n_quantizers
                * self.model.quantizer.get_bandwidth_per_quantizer(
                    self.model.frame_rate
                )
                / 1_000
            )
            self.model.set_target_bandwidth(bandwidth)

        try:
            signal = AudioSignal(audio_data, sample_rate)
            signal.resample(self.model.sample_rate)
            reconstructed = self.model(signal.audio_data)
            output = AudioSignal(reconstructed, self.model.sample_rate)
            output.resample(sample_rate)
            return {"audio": output.audio_data}
        finally:
            if previous_bandwidth is not None:
                self.model.set_target_bandwidth(previous_bandwidth)


__all__ = ["Encodec"]


if __name__ == "__main__":
    model = Encodec()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(model)
    print("Total parameters:", parameter_count)
