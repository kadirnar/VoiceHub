"""Framework-neutral adversarial fine-tuning for native Vocos."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voicehub.checkpointing import save_safetensors
from voicehub.components.audio.vocoders.vocos.discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
)
from voicehub.components.audio.vocoders.vocos.feature_extractors import (
    FeatureExtractor,
)
from voicehub.components.audio.vocoders.vocos.heads import FourierHead
from voicehub.components.audio.vocoders.vocos.loss import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MelSpecReconstructionLoss,
)
from voicehub.components.audio.vocoders.vocos.models import Backbone


@dataclass(frozen=True)
class VocosTrainingStep:
    """One differentiable loss and detached scalar metrics."""

    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor]
    optimizer: str
    skipped: bool = False


@contextmanager
def _temporarily_frozen(*modules: nn.Module):
    states = tuple(
        (parameter, parameter.requires_grad)
        for module in modules
        for parameter in module.parameters()
    )
    try:
        for parameter, _ in states:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)


def _cosine_warmup_lambda(
    current_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if current_step < warmup_steps:
        return float(current_step) / max(1.0, float(warmup_steps))
    progress = float(current_step - warmup_steps) / max(
        1.0,
        float(total_steps - warmup_steps),
    )
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))


class VocosExp(nn.Module):
    """Native Vocos GAN objective independent of a trainer framework.

    ``training_step`` keeps the historical optimizer-index contract, while
    ``generator_step`` and ``discriminator_step`` provide explicit methods for
    VoiceHub and future optimization engines.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int = 0,
        mel_loss_coeff: float = 45.0,
        mrd_loss_coeff: float = 1.0,
        pretrain_mel_steps: int = 0,
        decay_mel_coeff: bool = False,
        evaluate_utmos: bool = False,
        evaluate_pesq: bool = False,
        evaluate_periodicty: bool = False,
        evaluation_metrics: Mapping[
            str,
            Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
        ] | None = None,
        multiperiod_discriminator: MultiPeriodDiscriminator | None = None,
        multiresolution_discriminator: MultiResolutionDiscriminator | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(sample_rate, bool)
            or not isinstance(sample_rate, int)
            or sample_rate <= 0
        ):
            raise ValueError("`sample_rate` must be a positive integer.")
        if (
            not isinstance(initial_learning_rate, (int, float))
            or isinstance(initial_learning_rate, bool)
            or not math.isfinite(float(initial_learning_rate))
            or initial_learning_rate <= 0
        ):
            raise ValueError("`initial_learning_rate` must be finite and positive.")
        for name, value in (
            ("num_warmup_steps", num_warmup_steps),
            ("pretrain_mel_steps", pretrain_mel_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer.")
        for name, value in (
            ("mel_loss_coeff", mel_loss_coeff),
            ("mrd_loss_coeff", mrd_loss_coeff),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"`{name}` must be finite and non-negative.")
        if any((evaluate_utmos, evaluate_pesq, evaluate_periodicty)):
            raise ValueError(
                "Provider-specific UTMOS, PESQ, and periodicity packages are "
                "not part of the native runtime. Supply reviewed metrics "
                "through `evaluation_metrics`."
            )

        self.feature_extractor = feature_extractor
        self.backbone = backbone
        self.head = head
        self.multiperioddisc = (
            multiperiod_discriminator or MultiPeriodDiscriminator()
        )
        self.multiresddisc = (
            multiresolution_discriminator or MultiResolutionDiscriminator()
        )
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.feat_matching_loss = FeatureMatchingLoss()
        self.melspec_loss = MelSpecReconstructionLoss(sample_rate=sample_rate)

        self.sample_rate = sample_rate
        self.initial_learning_rate = float(initial_learning_rate)
        self.num_warmup_steps = num_warmup_steps
        self.base_mel_coeff = float(mel_loss_coeff)
        self.mrd_loss_coeff = float(mrd_loss_coeff)
        self.pretrain_mel_steps = pretrain_mel_steps
        self.decay_mel_coeff = bool(decay_mel_coeff)
        self.evaluation_metrics = dict(evaluation_metrics or {})
        if any(
            not isinstance(name, str) or not name or not callable(metric)
            for name, metric in self.evaluation_metrics.items()
        ):
            raise TypeError(
                "`evaluation_metrics` must map non-empty names to callables."
            )
        self.last_step: VocosTrainingStep | None = None

    def forward(self, audio_input: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        features = self.feature_extractor(audio_input, **kwargs)
        hidden_states = self.backbone(features, **kwargs)
        return self.head(hidden_states)

    def _mel_coefficient(self, global_step: int, total_steps: int | None) -> float:
        if not self.decay_mel_coeff or total_steps is None:
            return self.base_mel_coeff
        if total_steps <= 0:
            raise ValueError("`total_steps` must be positive when supplied.")
        decay = _cosine_warmup_lambda(
            global_step,
            warmup_steps=self.num_warmup_steps,
            total_steps=total_steps,
        )
        return self.base_mel_coeff * decay

    def discriminator_step(
        self,
        audio_input: torch.Tensor,
        *,
        global_step: int,
        **kwargs: Any,
    ) -> VocosTrainingStep:
        if global_step < self.pretrain_mel_steps:
            loss = self.head.out.weight.sum() * 0.0
            return VocosTrainingStep(
                loss=loss,
                metrics={"discriminator/total": loss.detach()},
                optimizer="discriminator",
                skipped=True,
            )
        with torch.no_grad():
            audio_hat = self(audio_input, **kwargs)
        real_mp, generated_mp, _, _ = self.multiperioddisc(
            y=audio_input,
            y_hat=audio_hat.detach(),
            **kwargs,
        )
        real_mrd, generated_mrd, _, _ = self.multiresddisc(
            y=audio_input,
            y_hat=audio_hat.detach(),
            **kwargs,
        )
        loss_mp, real_terms_mp, _ = self.disc_loss(real_mp, generated_mp)
        loss_mrd, real_terms_mrd, _ = self.disc_loss(real_mrd, generated_mrd)
        loss_mp = loss_mp / max(1, len(real_terms_mp))
        loss_mrd = loss_mrd / max(1, len(real_terms_mrd))
        loss = loss_mp + self.mrd_loss_coeff * loss_mrd
        return VocosTrainingStep(
            loss=loss,
            metrics={
                "discriminator/total": loss.detach(),
                "discriminator/multi_period_loss": loss_mp.detach(),
                "discriminator/multi_resolution_loss": loss_mrd.detach(),
            },
            optimizer="discriminator",
        )

    def generator_step(
        self,
        audio_input: torch.Tensor,
        *,
        global_step: int,
        total_steps: int | None = None,
        **kwargs: Any,
    ) -> VocosTrainingStep:
        audio_hat = self(audio_input, **kwargs)
        adversarial = global_step >= self.pretrain_mel_steps
        if adversarial:
            with _temporarily_frozen(
                self.multiperioddisc,
                self.multiresddisc,
            ):
                _, generated_mp, real_features_mp, generated_features_mp = (
                    self.multiperioddisc(
                        y=audio_input,
                        y_hat=audio_hat,
                        **kwargs,
                    )
                )
                _, generated_mrd, real_features_mrd, generated_features_mrd = (
                    self.multiresddisc(
                        y=audio_input,
                        y_hat=audio_hat,
                        **kwargs,
                    )
                )
                loss_gen_mp, terms_gen_mp = self.gen_loss(generated_mp)
                loss_gen_mrd, terms_gen_mrd = self.gen_loss(generated_mrd)
                loss_gen_mp = loss_gen_mp / max(1, len(terms_gen_mp))
                loss_gen_mrd = loss_gen_mrd / max(1, len(terms_gen_mrd))
                loss_fm_mp = self.feat_matching_loss(
                    real_features_mp,
                    generated_features_mp,
                ) / max(1, len(real_features_mp))
                loss_fm_mrd = self.feat_matching_loss(
                    real_features_mrd,
                    generated_features_mrd,
                ) / max(1, len(real_features_mrd))
        else:
            zero = audio_hat.sum() * 0.0
            loss_gen_mp = loss_gen_mrd = loss_fm_mp = loss_fm_mrd = zero

        mel_loss = self.melspec_loss(audio_hat, audio_input)
        mel_coefficient = self._mel_coefficient(global_step, total_steps)
        loss = (
            loss_gen_mp
            + self.mrd_loss_coeff * loss_gen_mrd
            + loss_fm_mp
            + self.mrd_loss_coeff * loss_fm_mrd
            + mel_coefficient * mel_loss
        )
        return VocosTrainingStep(
            loss=loss,
            metrics={
                "generator/total_loss": loss.detach(),
                "generator/multi_period_loss": loss_gen_mp.detach(),
                "generator/multi_resolution_loss": loss_gen_mrd.detach(),
                "generator/feature_matching_mp": loss_fm_mp.detach(),
                "generator/feature_matching_mrd": loss_fm_mrd.detach(),
                "generator/mel_loss": mel_loss.detach(),
                "generator/mel_loss_coeff": torch.tensor(
                    mel_coefficient,
                    device=loss.device,
                    dtype=loss.dtype,
                ),
            },
            optimizer="generator",
        )

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int = 0,
        optimizer_idx: int = 1,
        *,
        global_step: int | None = None,
        total_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del batch_idx
        step = 0 if global_step is None else global_step
        if optimizer_idx == 0:
            result = self.discriminator_step(
                batch,
                global_step=step,
                **kwargs,
            )
        elif optimizer_idx == 1:
            result = self.generator_step(
                batch,
                global_step=step,
                total_steps=total_steps,
                **kwargs,
            )
        else:
            raise ValueError("Vocos `optimizer_idx` must be 0 or 1.")
        self.last_step = result
        return result.loss

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del batch_idx
        prediction = self(batch, **kwargs)
        metrics = {
            "val/mel_loss": self.melspec_loss(prediction, batch),
        }
        for name, metric in self.evaluation_metrics.items():
            value = metric(batch, prediction, self.sample_rate)
            value = torch.as_tensor(value, device=prediction.device)
            if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
                raise ValueError(
                    f"Vocos evaluation metric {name!r} returned an invalid value."
                )
            metrics[f"val/{name}"] = value.reshape(())
        return metrics

    def configure_optimizers(
        self,
        *,
        total_steps: int,
    ) -> tuple[
        tuple[torch.optim.Optimizer, torch.optim.Optimizer],
        tuple[
            torch.optim.lr_scheduler.LambdaLR,
            torch.optim.lr_scheduler.LambdaLR,
        ],
    ]:
        if (
            isinstance(total_steps, bool)
            or not isinstance(total_steps, int)
            or total_steps <= 0
        ):
            raise ValueError("`total_steps` must be a positive integer.")
        discriminator = torch.optim.AdamW(
            (
                {
                    "params": self.multiperioddisc.parameters(),
                },
                {
                    "params": self.multiresddisc.parameters(),
                },
            ),
            lr=self.initial_learning_rate,
            betas=(0.8, 0.9),
        )
        generator = torch.optim.AdamW(
            (
                {"params": self.feature_extractor.parameters()},
                {"params": self.backbone.parameters()},
                {"params": self.head.parameters()},
            ),
            lr=self.initial_learning_rate,
            betas=(0.8, 0.9),
        )
        schedule = lambda step: _cosine_warmup_lambda(
            step,
            warmup_steps=self.num_warmup_steps,
            total_steps=total_steps,
        )
        return (
            (discriminator, generator),
            (
                torch.optim.lr_scheduler.LambdaLR(discriminator, schedule),
                torch.optim.lr_scheduler.LambdaLR(generator, schedule),
            ),
        )

    def export_generator(self, path: str | Path) -> Path:
        """Export a fresh-inference-compatible Vocos generator."""
        state = {
            **{
                f"feature_extractor.{name}": value.detach().cpu().contiguous()
                for name, value in self.feature_extractor.state_dict().items()
            },
            **{
                f"backbone.{name}": value.detach().cpu().contiguous()
                for name, value in self.backbone.state_dict().items()
            },
            **{
                f"head.{name}": value.detach().cpu().contiguous()
                for name, value in self.head.state_dict().items()
            },
        }
        return save_safetensors(
            state,
            path,
            metadata={
                "format": "voicehub-native-vocos-v1",
                "artifact": "generator",
            },
        )


class VocosEncodecExp(VocosExp):
    """Conditional Vocos training over native Encodec bandwidths."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        bandwidths = getattr(self.feature_extractor, "bandwidths", None)
        if not bandwidths:
            raise TypeError("VocosEncodecExp requires an Encodec feature extractor.")
        count = len(bandwidths)
        self.multiperioddisc = MultiPeriodDiscriminator(num_embeddings=count)
        self.multiresddisc = MultiResolutionDiscriminator(num_embeddings=count)

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int = 0,
        optimizer_idx: int = 1,
        **kwargs: Any,
    ) -> torch.Tensor:
        bandwidth_id = kwargs.pop("bandwidth_id", None)
        if bandwidth_id is None:
            bandwidth_id = torch.randint(
                len(self.feature_extractor.bandwidths),
                (1,),
                device=batch.device,
            )
        return super().training_step(
            batch,
            batch_idx,
            optimizer_idx,
            bandwidth_id=bandwidth_id,
            **kwargs,
        )

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        bandwidth_id = kwargs.pop(
            "bandwidth_id",
            torch.zeros(1, dtype=torch.long, device=batch.device),
        )
        return super().validation_step(
            batch,
            batch_idx,
            bandwidth_id=bandwidth_id,
            **kwargs,
        )


__all__ = [
    "VocosEncodecExp",
    "VocosExp",
    "VocosTrainingStep",
]
