"""Adversarial, feature-matching, and KL components used by VITS."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from voicehub.training.objectives._shared import (
    active_selection_mask,
    expand_mask,
    mask_sequence,
    masked_reduction,
    require_floating_tensor,
    require_tensor,
    tensor_sequence,
    torch_module,
)


@dataclass(frozen=True)
class VITSDiscriminatorLoss:
    """Total and per-discriminator least-squares GAN losses."""

    loss: Any
    real_losses: tuple[Any, ...]
    fake_losses: tuple[Any, ...]


def vits_discriminator_loss(
    real_scores: Any,
    fake_scores: Any,
    *,
    masks: Any | None = None,
) -> VITSDiscriminatorLoss:
    """Compute the standard VITS least-squares discriminator objective.

    Each real/fake score pair must have the same shape.  Detach
    generated audio before passing it to the discriminator;
    discriminator scores themselves must remain differentiable so the
    fake branch trains the discriminator.
    """
    torch = torch_module()
    real_scores = tensor_sequence(
        real_scores,
        name="real_scores",
        torch=torch,
    )
    fake_scores = tensor_sequence(
        fake_scores,
        name="fake_scores",
        torch=torch,
    )
    if len(real_scores) != len(fake_scores):
        raise ValueError("Real and fake discriminator output counts must match.")
    masks = mask_sequence(
        masks,
        len(real_scores),
        name="masks",
        torch=torch,
    )
    real_losses = []
    fake_losses = []
    for index, (real, fake, mask) in enumerate(zip(real_scores, fake_scores, masks)):
        if tuple(real.shape) != tuple(fake.shape):
            raise ValueError(
                f"Discriminator pair {index} has mismatched shapes "
                f"{tuple(real.shape)} and {tuple(fake.shape)}.")
        _, _, active = active_selection_mask(
            real,
            mask=mask,
            weights=None,
            torch=torch,
        )
        zero = torch.zeros((), device=real.device, dtype=real.dtype)
        safe_real = torch.where(active, real, zero)
        safe_fake = torch.where(
            active,
            fake.to(device=real.device, dtype=real.dtype),
            zero,
        )
        real_losses.append(
            masked_reduction(
                (1 - safe_real).square(),
                mask=mask,
                weights=None,
                reduction="mean",
                torch=torch,
            ))
        fake_losses.append(
            masked_reduction(
                safe_fake.square(),
                mask=mask,
                weights=None,
                reduction="mean",
                torch=torch,
            ))
    loss = sum(real_losses) + sum(fake_losses)
    return VITSDiscriminatorLoss(
        loss=loss,
        real_losses=tuple(real_losses),
        fake_losses=tuple(fake_losses),
    )


def vits_generator_adversarial_loss(
    fake_scores: Any,
    *,
    masks: Any | None = None,
):
    """Compute the standard VITS least-squares generator objective."""
    torch = torch_module()
    fake_scores = tensor_sequence(
        fake_scores,
        name="fake_scores",
        torch=torch,
    )
    masks = mask_sequence(
        masks,
        len(fake_scores),
        name="masks",
        torch=torch,
    )
    losses = []
    for score, mask in zip(fake_scores, masks):
        _, _, active = active_selection_mask(
            score,
            mask=mask,
            weights=None,
            torch=torch,
        )
        safe_score = torch.where(
            active,
            score,
            torch.zeros((), device=score.device, dtype=score.dtype),
        )
        losses.append(
            masked_reduction(
                (1 - safe_score).square(),
                mask=mask,
                weights=None,
                reduction="mean",
                torch=torch,
            ))
    return sum(losses)


def _feature_pyramid(
    value: Any,
    *,
    name: str,
    torch: Any,
) -> tuple[tuple[Any, ...], ...]:
    if torch.is_tensor(value):
        return ((require_floating_tensor(
            value,
            name=name,
            torch=torch,
        ), ), )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"`{name}` must be a tensor or nested sequence of tensors.")
    outer = tuple(value)
    if not outer:
        raise ValueError(f"`{name}` cannot be empty.")
    if all(torch.is_tensor(item) for item in outer):
        outer = (outer, )
    output = []
    for discriminator_index, layers in enumerate(outer):
        if (not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)) or not layers):
            raise TypeError(f"`{name}[{discriminator_index}]` must be a non-empty "
                            "sequence of tensors.")
        output.append(
            tuple(
                require_floating_tensor(
                    layer,
                    name=(f"{name}[{discriminator_index}]"
                          f"[{layer_index}]"),
                    torch=torch,
                ) for layer_index, layer in enumerate(layers)))
    return tuple(output)


def _feature_masks(
    masks: Any | None,
    features: tuple[tuple[Any, ...], ...],
    *,
    torch: Any,
) -> tuple[tuple[Any | None, ...], ...]:
    """Normalize masks to the same discriminator/layer pyramid as features."""
    if masks is None:
        return tuple((None, ) * len(layers) for layers in features)
    if torch.is_tensor(masks):
        if len(features) != 1 or len(features[0]) != 1:
            raise ValueError("A single feature mask can only mask one feature tensor.")
        return ((masks, ), )
    if not isinstance(masks, Sequence) or isinstance(masks, (str, bytes)):
        raise TypeError("`masks` must mirror the feature pyramid.")

    raw_masks = tuple(masks)
    if len(features) == 1 and len(raw_masks) == len(features[0]):
        raw_masks = (raw_masks, )
    if len(raw_masks) != len(features):
        raise ValueError("`masks` must contain one entry per discriminator.")

    mask_pyramid = []
    for index, (layer_masks, layers) in enumerate(zip(raw_masks, features)):
        if not isinstance(layer_masks, Sequence) or isinstance(layer_masks, (str, bytes)):
            raise TypeError(f"`masks[{index}]` must be a sequence.")
        layer_masks = tuple(layer_masks)
        if len(layer_masks) != len(layers):
            raise ValueError(f"`masks[{index}]` must contain one entry per layer.")
        mask_pyramid.append(layer_masks)
    return tuple(mask_pyramid)


def vits_feature_matching_loss(
    real_features: Any,
    fake_features: Any,
    *,
    masks: Any | None = None,
    scale: float = 2.0,
    detach_real: bool = True,
):
    """Compute VITS feature matching across discriminator feature pyramids.

    Inputs may be one tensor, one sequence of layer tensors, or a nested
    ``[discriminator][layer]`` sequence.  Every real/fake feature pair
    must have the same shape.  Real features are detached by default,
    matching the generator-side feature-matching objective.
    """
    if (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or
            scale < 0):
        raise ValueError("`scale` must be a finite non-negative number.")
    torch = torch_module()
    real = _feature_pyramid(
        real_features,
        name="real_features",
        torch=torch,
    )
    fake = _feature_pyramid(
        fake_features,
        name="fake_features",
        torch=torch,
    )
    if len(real) != len(fake):
        raise ValueError("Real and fake feature-pyramid discriminator counts must match.")
    mask_pyramid = _feature_masks(masks, real, torch=torch)

    losses = []
    for discriminator_index, (real_layers, fake_layers, layer_masks) in enumerate(zip(real, fake,
                                                                                      mask_pyramid)):
        if len(real_layers) != len(fake_layers):
            raise ValueError(
                f"Feature-pyramid discriminator {discriminator_index} has "
                "different real/fake layer counts.")
        for layer_index, (real_layer, fake_layer, mask) in enumerate(zip(real_layers, fake_layers,
                                                                         layer_masks)):
            if tuple(real_layer.shape) != tuple(fake_layer.shape):
                raise ValueError(
                    f"Feature pair [{discriminator_index}][{layer_index}] "
                    f"has mismatched shapes {tuple(real_layer.shape)} and "
                    f"{tuple(fake_layer.shape)}.")
            reference = real_layer.detach() if detach_real else real_layer
            _, _, active = active_selection_mask(
                fake_layer,
                mask=mask,
                weights=None,
                torch=torch,
            )
            zero = torch.zeros(
                (),
                device=fake_layer.device,
                dtype=fake_layer.dtype,
            )
            safe_fake = torch.where(active, fake_layer, zero)
            safe_reference = torch.where(
                active,
                reference.to(
                    device=fake_layer.device,
                    dtype=fake_layer.dtype,
                ),
                zero,
            )
            losses.append(
                masked_reduction(
                    (safe_fake - safe_reference).abs(),
                    mask=mask,
                    weights=None,
                    reduction="mean",
                    torch=torch,
                ))
    return float(scale) * sum(losses)


def vits_kl_loss(
    posterior_latents: Any,
    posterior_log_scale: Any,
    prior_mean: Any,
    prior_log_scale: Any,
    *,
    mask: Any | None = None,
):
    """Compute the diagonal-Gaussian KL component used by standard VITS.

    All four tensors must have exactly the same floating-point shape.  The
    returned value is
    ``log(p_scale) - log(q_scale) - 0.5
    + 0.5 * (z_p - p_mean)^2 * exp(-2 * log(p_scale))``.

    With a mask, the loss is normalized over explicitly selected time/batch
    positions.  As in VITS, broadcast latent-channel dimensions are summed
    rather than included in the mask denominator.  Without a mask, the loss
    is an elementwise mean because no latent-channel axis is inferred.
    """
    torch = torch_module()
    values = (
        require_floating_tensor(
            posterior_latents,
            name="posterior_latents",
            torch=torch,
        ),
        require_floating_tensor(
            posterior_log_scale,
            name="posterior_log_scale",
            torch=torch,
        ),
        require_floating_tensor(
            prior_mean,
            name="prior_mean",
            torch=torch,
        ),
        require_floating_tensor(
            prior_log_scale,
            name="prior_log_scale",
            torch=torch,
        ),
    )
    expected = tuple(values[0].shape)
    for name, value in zip(
        (
            "posterior_log_scale",
            "prior_mean",
            "prior_log_scale",
        ),
            values[1:],
    ):
        if tuple(value.shape) != expected:
            raise ValueError(
                "VITS KL tensors must have identical shapes; "
                f"`{name}` has {tuple(value.shape)}, expected {expected}.")
    z_p, logs_q, m_p, logs_p = (
        value.to(
            device=values[0].device,
            dtype=values[0].dtype,
        ) for value in values)
    if mask is None:
        kl = (logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p).square() * torch.exp(-2.0 * logs_p))
        return kl.mean()
    raw_mask = require_tensor(mask, name="mask", torch=torch)
    expanded_mask = expand_mask(
        raw_mask,
        z_p,
        name="mask",
        torch=torch,
    )
    zero = torch.zeros((), device=z_p.device, dtype=z_p.dtype)
    z_p = torch.where(expanded_mask, z_p, zero)
    logs_q = torch.where(expanded_mask, logs_q, zero)
    m_p = torch.where(expanded_mask, m_p, zero)
    logs_p = torch.where(expanded_mask, logs_p, zero)
    kl = (logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p).square() * torch.exp(-2.0 * logs_p))
    denominator = raw_mask.to(
        device=kl.device,
        dtype=torch.bool,
    ).sum().to(dtype=kl.dtype)
    if not bool((denominator > 0).item()):
        raise ValueError("The VITS KL mask does not select any positions.")
    return torch.where(
        expanded_mask,
        kl,
        torch.zeros((), device=kl.device, dtype=kl.dtype),
    ).sum() / denominator


__all__ = [
    "VITSDiscriminatorLoss",
    "vits_discriminator_loss",
    "vits_feature_matching_loss",
    "vits_generator_adversarial_loss",
    "vits_kl_loss",
]
