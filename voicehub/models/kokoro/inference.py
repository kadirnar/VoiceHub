"""Kokoro integration backed by the source included in VoiceHub."""

from __future__ import annotations

import math
import os
import re
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference

KOKORO_SAMPLE_RATE = 24_000


class KokoroConfig(VoiceHubConfig):
    """Configuration for Kokoro model and G2P pipeline."""

    model_type = "kokoro"

    def __init__(
        self,
        *,
        language_code: str = "a",
        lang_code: str | None = None,
        sample_rate: int = KOKORO_SAMPLE_RATE,
        **kwargs,
    ):
        # Kokoro's released decoder is fixed at 24 kHz. Keep serialized config
        # overrides from mislabelling generated audio.
        super().__init__(sample_rate=KOKORO_SAMPLE_RATE, **kwargs)
        self.language_code = language_code if lang_code is None else lang_code


class KokoroForTextToSpeech(PreTrainedTTSModel):
    """Multilingual Kokoro synthesis without the ``kokoro`` package."""

    config_class = KokoroConfig
    default_model_name_or_path = "hexgrad/Kokoro-82M"

    def __init__(
        self,
        config: KokoroConfig | str | None = None,
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
        self._requested_device = device
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _runtime_device(self) -> str:
        if (self._requested_device == "auto" and str(self.device).split(":", 1)[0].lower() == "mps" and
                os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"):
            # The vendored pipeline requires PyTorch's MPS CPU fallback for
            # operations which do not have Metal kernels. ``auto`` must remain
            # usable on Macs where that opt-in variable was not set.
            self.device = "cpu"
        return self.device

    def _load_pretrained_model(self) -> None:
        pipeline_module = import_optional(
            "voicehub.models.kokoro.pipeline",
            model_type="kokoro",
            install_extra="kokoro",
        )
        self.model = pipeline_module.KPipeline(
            lang_code=self.config.language_code,
            repo_id=self.config.name_or_path,
            device=self._runtime_device(),
        )
        self.config.sample_rate = KOKORO_SAMPLE_RATE

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        language_code = self.config.language_code
        if not isinstance(language_code, str) or not language_code.strip():
            raise ValueError("`language_code` must be a non-empty Kokoro language code.")
        voice = model_inputs.get("voice", "af_heart")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("`voice` must be a non-empty Kokoro voice name.")

        speed = model_inputs.get("speed", 1.0)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or
                speed <= 0):
            raise ValueError("`speed` must be a finite positive number.")

        split_pattern = model_inputs.get("split_pattern", r"\n+")
        if not isinstance(split_pattern, str) or not split_pattern:
            raise ValueError("`split_pattern` must be a non-empty regular expression.")
        try:
            re.compile(split_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid `split_pattern`: {exc}.") from exc

    def _generate(
        self,
        text: str,
        *,
        voice: str = "af_heart",
        speed: float = 1.0,
        split_pattern: str = r"\n+",
        output_file: str | None = None,
        seed: int | None = None,
    ) -> TTSOutput:
        chunks: list[Any] = []
        segments: list[str] = []
        with seeded_inference(
                seed,
                device=self.device,
                model_type="kokoro",
        ) as effective_seed:
            for result in self.model(
                    text,
                    voice=voice,
                    speed=speed,
                    split_pattern=split_pattern,
            ):
                if result.audio is not None:
                    chunks.append(result.audio.reshape(-1))
                    segments.append(result.graphemes)
        if not chunks:
            raise RuntimeError("Kokoro returned no audio.")

        torch = import_optional(
            "torch",
            model_type="kokoro",
            install_extra="kokoro",
        )
        return finish_audio_output(
            torch.cat(chunks),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "segments": tuple(segments),
                "voice": voice,
                "speed": speed,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


KokoroTTS = KokoroForTextToSpeech
