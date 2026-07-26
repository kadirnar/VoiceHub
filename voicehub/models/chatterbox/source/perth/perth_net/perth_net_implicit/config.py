from typing import NamedTuple


class PerthConfig(NamedTuple):
    use_wandb: bool
    batch_size: int
    sample_rate: int
    n_fft: int
    hop_size: int
    window_size: int
    use_lr_scheduler: bool
    stft_magnitude_min: float
    min_lr: float
    max_lr: float
    window_fn: str
    max_wmark_freq: float
    hidden_size: int
    # "simple" or "psychoacoustic"
    loss_type: str


default_hp = PerthConfig(
    use_wandb=True,
    batch_size=16,
    sample_rate=32000,
    n_fft=2048,
    hop_size=320,
    window_size=2048,
    use_lr_scheduler=False,
    stft_magnitude_min=1e-9,
    min_lr=1e-5,
    max_lr=1e-4,
    window_fn="hann",
    max_wmark_freq=2000,
    hidden_size=256,
    # loss_type="simple",
    loss_type="psychoacoustic",
)
