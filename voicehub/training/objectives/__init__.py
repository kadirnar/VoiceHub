"""Model-agnostic objective primitives for specialized TTS recipes.

The package deliberately imports no tensor framework at import time.
Public functions resolve PyTorch lazily when called.
"""

from voicehub.training.objectives.diffusion import (
    DiffusionTrainingPair,
    build_diffusion_training_pair,
    build_flow_matching_training_pair,
    masked_diffusion_regression_loss,
)
from voicehub.training.objectives.token import multi_codebook_cross_entropy
from voicehub.training.objectives.vits import (
    VITSDiscriminatorLoss,
    vits_discriminator_loss,
    vits_feature_matching_loss,
    vits_generator_adversarial_loss,
    vits_kl_loss,
)

__all__ = [
    "DiffusionTrainingPair",
    "VITSDiscriminatorLoss",
    "build_diffusion_training_pair",
    "build_flow_matching_training_pair",
    "masked_diffusion_regression_loss",
    "multi_codebook_cross_entropy",
    "vits_discriminator_loss",
    "vits_feature_matching_loss",
    "vits_generator_adversarial_loss",
    "vits_kl_loss",
]
