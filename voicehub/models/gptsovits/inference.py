"""GPT-SoVITS integration backed by vendored upstream source."""

from __future__ import annotations

import math
import secrets
from pathlib import Path
from typing import Any, Mapping

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, seeded_inference


class GPTSoVITSConfig(VoiceHubConfig):
    """Configuration for the GPT-SoVITS inference pipeline."""

    model_type = "gptsovits"

    def __init__(
        self,
        *,
        runtime_config: dict | None = None,
        sample_rate: int = 32000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.runtime_config = runtime_config


class GPTSoVITSForTextToSpeech(PreTrainedTTSModel):
    """Few-shot multilingual TTS without an external GPT-SoVITS checkout."""

    config_class = GPTSoVITSConfig
    default_model_name_or_path = ""
    _VERSION_CONFIG_KEYS = frozenset({
        "custom",
        "v1",
        "v2",
        "v2Pro",
        "v2ProPlus",
        "v3",
        "v4",
    })

    def __init__(
        self,
        config: GPTSoVITSConfig | str | None = None,
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

    def _resolve_runtime_config(self) -> dict[str, Any] | str:
        config_source = self.config.runtime_config
        if config_source is not None:
            if not isinstance(config_source, Mapping):
                raise TypeError("`runtime_config` must be a mapping of GPT-SoVITS options.")
            return self._normalize_runtime_config(config_source)

        if not self.config.name_or_path:
            raise FileNotFoundError(
                "GPT-SoVITS requires `model_path` pointing to an inference YAML "
                "or `runtime_config` containing checkpoint paths.")
        config_path = Path(self.config.name_or_path).expanduser()
        if not config_path.is_file():
            raise FileNotFoundError(f"GPT-SoVITS inference configuration was not found: {config_path}.")
        return str(config_path.resolve())

    @classmethod
    def _normalize_runtime_config(
        cls,
        config_source: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Accept both upstream version maps and VoiceHub's flat options.

        Upstream ``TTS_Config`` selects ``custom`` (or ``v2``) from a
        version-keyed mapping. Passing checkpoint paths at the top level
        would otherwise be accepted but silently ignored in favour of
        upstream defaults.
        """
        normalized = dict(config_source)
        if cls._VERSION_CONFIG_KEYS.intersection(normalized):
            return normalized
        custom = dict(normalized)
        custom.setdefault("version", "v2")
        return {"custom": custom}

    def _load_pretrained_model(self) -> None:
        config_source = self._resolve_runtime_config()
        runtime = import_optional(
            "voicehub.models.gptsovits.source.GPT_SoVITS.TTS_infer_pack.TTS",
            model_type="gptsovits",
            install_extra=None,
        )
        runtime_config = runtime.TTS_Config(config_source)
        runtime_config.device = self.device
        if str(self.device).split(":", 1)[0].lower() == "cpu":
            # ``TTS_Config`` validates precision before VoiceHub overrides the
            # configured device. Keep those fields synchronized so a config
            # authored for CUDA cannot leave CPU inference in half precision.
            runtime_config.is_half = False
        runtime_config.update_configs()
        self.model = runtime.TTS(runtime_config)

    def _prepare_for_inference(self) -> None:
        """Restore eval mode on modules hidden by the upstream TTS shell."""
        if self.model is None:
            return
        for component_name in (
                "t2s_model",
                "vits_model",
                "cnhuhbert_model",
                "bert_model",
                "vocoder",
        ):
            component = getattr(self.model, component_name, None)
            evaluate = getattr(component, "eval", None)
            if callable(evaluate):
                evaluate()

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        required_text = {
            "text_language": "the language of the synthesis text",
            "prompt_language": "the language of the reference transcript",
        }
        for name, description in required_text.items():
            value = model_inputs.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must specify {description}.")

        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip():
            raise ValueError("`speaker_audio_path` must point to a local reference-audio file.")
        reference_path = Path(speaker_audio_path).expanduser()
        if not reference_path.is_file():
            raise FileNotFoundError(f"GPT-SoVITS reference audio was not found: {reference_path}.")

        speed = model_inputs.get("speed", 1.0)
        if (not isinstance(speed, (int, float)) or isinstance(speed, bool) or not math.isfinite(speed) or
                speed <= 0):
            raise ValueError("`speed` must be a finite positive number.")
        temperature = model_inputs.get("temperature", 1.0)
        if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("`temperature` must be a finite non-negative number.")
        for name, default in (("batch_size", 1), ("top_k", 15)):
            value = model_inputs.get(name, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"`{name}` must be a positive integer.")
        top_p = model_inputs.get("top_p", 1.0)
        if (not isinstance(top_p, (int, float)) or isinstance(top_p, bool) or not math.isfinite(top_p) or
                not 0 <= top_p <= 1):
            raise ValueError("`top_p` must be in the interval [0, 1].")
        seed = model_inputs.get("seed", -1)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("`seed` must be an integer.")
        if seed != -1 and not 0 <= seed < 2**32:
            raise ValueError("`seed` must be -1 for a random take or an integer in "
                             "[0, 2**32).")

    @staticmethod
    def _build_request(
        text: str,
        *,
        text_language: str,
        speaker_audio_path: str,
        prompt_language: str,
        prompt_text: str,
        speed: float,
        seed: int,
        batch_size: int,
        text_split_method: str,
        parallel_inference: bool,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "text": text,
            "text_lang": text_language,
            "ref_audio_path": str(Path(speaker_audio_path).expanduser()),
            "prompt_text": prompt_text,
            "prompt_lang": prompt_language,
            "speed_factor": speed,
            "seed": seed,
            "batch_size": batch_size,
            "text_split_method": text_split_method,
            "parallel_infer": parallel_inference,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "streaming_mode": False,
            "return_fragment": False,
        }

    def _generate(
        self,
        text: str,
        *,
        text_language: str,
        speaker_audio_path: str,
        prompt_language: str,
        prompt_text: str = "",
        output_file: str | None = None,
        speed: float = 1.0,
        seed: int = -1,
        batch_size: int = 1,
        text_split_method: str = "cut5",
        parallel_inference: bool = True,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> TTSOutput:
        effective_seed = (secrets.randbelow(2**32) if seed == -1 else seed)
        request = self._build_request(
            text,
            text_language=text_language,
            speaker_audio_path=speaker_audio_path,
            prompt_language=prompt_language,
            prompt_text=prompt_text,
            speed=speed,
            seed=effective_seed,
            batch_size=batch_size,
            text_split_method=text_split_method,
            parallel_inference=parallel_inference,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        with seeded_inference(
                effective_seed,
                device=self.device,
                model_type="gptsovits",
        ):
            results = list(self.model.run(request))
        if not results:
            raise RuntimeError("GPT-SoVITS returned no audio.")

        np = import_optional(
            "numpy",
            model_type="gptsovits",
            install_extra=None,
        )
        malformed = [result for result in results if not isinstance(result, tuple) or len(result) != 2]
        if malformed:
            raise RuntimeError(
                "GPT-SoVITS returned a malformed chunk; expected "
                "(sample_rate, waveform) pairs.")
        sample_rates = {int(sample_rate) for sample_rate, _ in results}
        if len(sample_rates) != 1:
            raise RuntimeError("GPT-SoVITS returned chunks with different sample rates.")
        self.config.sample_rate = sample_rates.pop()
        chunks = [np.asarray(chunk).reshape(-1) for _, chunk in results]
        if any(chunk.size == 0 for chunk in chunks):
            raise RuntimeError("GPT-SoVITS returned an empty audio chunk.")
        return finish_audio_output(
            np.concatenate(chunks),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "seed": effective_seed,
                "text_language": text_language,
                "prompt_language": prompt_language,
            },
        )


GPTSoVITSTTS = GPTSoVITSForTextToSpeech
