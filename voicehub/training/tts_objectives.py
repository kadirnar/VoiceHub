"""Compatibility facade for model-agnostic TTS objective primitives.

Implementations live in :mod:`voicehub.training.objectives`, split by
token, diffusion/flow, and VITS training families.  This module
preserves the original public import path.
"""

from voicehub.training.objectives import (
    DiffusionTrainingPair,
    VITSDiscriminatorLoss,
    build_diffusion_training_pair,
    build_flow_matching_training_pair,
    masked_diffusion_regression_loss,
    multi_codebook_cross_entropy,
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
