"""PyTorch-native datasets for Vocos fine-tuning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from voicehub.processing.waveform import load_native_audio


@dataclass(frozen=True, slots=True)
class DataConfig:
    """One deterministic Vocos data-loader configuration."""

    filelist_path: str | Path
    sampling_rate: int
    num_samples: int
    batch_size: int
    num_workers: int = 0

    def __post_init__(self) -> None:
        for name in ("sampling_rate", "num_samples", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        if (
            isinstance(self.num_workers, bool)
            or not isinstance(self.num_workers, int)
            or self.num_workers < 0
        ):
            raise ValueError("`num_workers` must be a non-negative integer.")
        path = Path(self.filelist_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Vocos file list was not found: {path}.")
        object.__setattr__(self, "filelist_path", path.resolve())


class VocosDataset(Dataset[torch.Tensor]):
    """Load mono PCM WAVE records without torchaudio or NumPy."""

    def __init__(
        self,
        config: DataConfig,
        *,
        train: bool,
        generator: torch.Generator | None = None,
    ) -> None:
        lines = Path(config.filelist_path).read_text(encoding="utf-8").splitlines()
        root = Path(config.filelist_path).parent
        paths = (
            Path(line.strip()).expanduser()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        )
        self.filelist = tuple(
            path if path.is_absolute() else root / path
            for path in paths
        )
        if not self.filelist:
            raise ValueError("Vocos file list does not contain any audio paths.")
        missing = tuple(path for path in self.filelist if not path.is_file())
        if missing:
            raise FileNotFoundError(
                f"Vocos audio file was not found: {missing[0]}."
            )
        self.sampling_rate = config.sampling_rate
        self.num_samples = config.num_samples
        self.train = bool(train)
        self.generator = generator

    def __len__(self) -> int:
        return len(self.filelist)

    def _random_integer(self, high: int) -> int:
        return int(
            torch.randint(
                high,
                (1,),
                generator=self.generator,
            ).item()
        )

    def _normalize_peak(self, waveform: torch.Tensor) -> torch.Tensor:
        gain_db = (
            -6.0 + 5.0 * float(torch.rand((), generator=self.generator).item())
            if self.train
            else -3.0
        )
        peak = waveform.abs().amax()
        if float(peak.item()) <= 0:
            return waveform
        target_peak = math.pow(10.0, gain_db / 20.0)
        return waveform * (target_peak / peak)

    def __getitem__(self, index: int) -> torch.Tensor:
        native = load_native_audio(
            self.filelist[index],
            target_sampling_rate=self.sampling_rate,
        )
        waveform = native.waveform.float()
        if waveform.ndim != 1:
            raise RuntimeError("Native audio loader did not return mono audio.")
        if waveform.numel() == 0:
            raise ValueError(f"Vocos audio is empty: {self.filelist[index]}.")
        waveform = self._normalize_peak(waveform)

        if waveform.numel() < self.num_samples:
            repetitions = math.ceil(self.num_samples / waveform.numel())
            waveform = waveform.repeat(repetitions)[:self.num_samples]
        elif waveform.numel() > self.num_samples:
            start = (
                self._random_integer(waveform.numel() - self.num_samples + 1)
                if self.train
                else 0
            )
            waveform = waveform[start:start + self.num_samples]
        return waveform.contiguous()


class VocosDataModule:
    """Small framework-independent DataLoader factory."""

    def __init__(
        self,
        train_params: DataConfig,
        val_params: DataConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.train_config = train_params
        self.val_config = val_params
        self.generator = generator

    def _get_dataloader(self, config: DataConfig, *, train: bool) -> DataLoader:
        dataset = VocosDataset(
            config,
            train=train,
            generator=self.generator,
        )
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=train,
            pin_memory=torch.cuda.is_available(),
            generator=self.generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_config, train=False)


__all__ = [
    "DataConfig",
    "VocosDataModule",
    "VocosDataset",
]
