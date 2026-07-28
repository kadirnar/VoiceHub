"""Inflect Micro/Nano v2 inference backed by vendored architecture source."""

from __future__ import annotations

import math
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, seeded_inference


class InflectTTSConfig(VoiceHubConfig):
    """Configuration for the compact Inflect v2 checkpoint family."""

    model_type = "inflecttts"

    def __init__(self, *, sample_rate: int = 22050, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)


class InflectTTSForTextToSpeech(PreTrainedTTSModel):
    """Source-integrated Inflect Micro/Nano v2 speech synthesis."""

    config_class = InflectTTSConfig
    default_model_name_or_path = "owensong/Inflect-Micro-v2"

    def __init__(
        self,
        config: InflectTTSConfig | str | None = None,
        *,
        model_path: str | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **config_overrides,
    ):
        config = self._coerce_config(
            config,
            model_path=model_path,
            **config_overrides,
        )
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="inflecttts",
        )
        runtime = import_optional(
            "voicehub.models.inflecttts.source.inflect.inference",
            model_type="inflecttts",
            install_extra=None,
        )
        self.model = runtime.InflectTTS(
            model_directory,
            device=self.device,
        )
        self.config.sample_rate = int(self.model.sample_rate)

    def _validate_training_runtime(self) -> None:
        raise RuntimeError(
            "Inflect v2 is published as an inference-first VITS artifact. Its "
            "configuration sets `inference_only=true`, and the checkpoint "
            "omits the posterior encoder required by the native VITS "
            "alignment, KL, and adversarial objectives. Fine-tuning requires "
            "the author's full generator/discriminator training checkpoint, "
            "not the deployable model.pth file.")

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        speed = model_inputs.get("speed", 1.0)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or
                not 0.5 <= speed <= 2.0):
            raise ValueError("`speed` must be a finite number between 0.5 and 2.0.")
        variation = model_inputs.get("variation", 0.667)
        if (not isinstance(variation, (int, float)) or isinstance(variation, bool) or
                not math.isfinite(variation) or not 0 <= variation <= 1):
            raise ValueError("`variation` must be a finite number between 0 and 1.")
        seed = model_inputs.get("seed", 0)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("`seed` must be an integer.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
    ) -> TTSOutput:
        with seeded_inference(
                seed,
                device=self.device,
                model_type="inflecttts",
        ):
            sample_rate, audio = self.model.synthesize(
                text,
                speed=speed,
                variation=variation,
                seed=seed,
            )
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError(f"Inflect returned an invalid sample rate: {sample_rate}.")
        if audio is None:
            sample_count = 0
        elif hasattr(audio, "numel"):
            sample_count = audio.numel()
        elif hasattr(audio, "size"):
            sample_count = audio.size
        else:
            sample_count = len(audio)
        if sample_count == 0:
            raise RuntimeError("Inflect returned an empty audio waveform.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "speed": speed,
                "variation": variation,
                "seed": seed,
            },
        )


InflectTTSModel = InflectTTSForTextToSpeech
