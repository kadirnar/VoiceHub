"""ZONOS2 offline inference backed by vendored source."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


class Zonos2Config(VoiceHubConfig):
    """Configuration for the ZONOS2 MoE TTS runtime."""

    model_type = "zonos2"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        decode_audio: bool = True,
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.decode_audio = decode_audio


class Zonos2ForTextToSpeech(PreTrainedTTSModel):
    """Batched ZONOS2 synthesis with cloning and conditioning controls."""

    config_class = Zonos2Config
    default_model_name_or_path = "Zyphra/ZONOS2"

    def __init__(
        self,
        config: Zonos2Config | str | None = None,
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
        self._sampling_class = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _validate_training_runtime(self) -> None:
        raise RuntimeError(
            "ZONOS2's vendored TTSLLM is a fused inference engine rather than "
            "a differentiable nn.Module. Fine-tuning requires a custom "
            "training adapter built around the unfused training graph.")

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="zonos2",
            install_extra="zonos2",
        )
        if str(self.device).split(":", 1)[0].lower() != "cuda":
            raise RuntimeError(
                "ZONOS2's fused runtime requires CUDA. Construct the model "
                "with `device='cuda'` or a specific device such as `cuda:1`.")
        runtime = import_optional(
            "voicehub.models.zonos2.source.zonos2.tts",
            model_type="zonos2",
            install_extra="zonos2",
        )
        message = import_optional(
            "voicehub.models.zonos2.source.zonos2.message",
            model_type="zonos2",
            install_extra="zonos2",
        )
        self.model = runtime.TTSLLM(
            model_path=self.config.name_or_path,
            dtype=resolve_torch_dtype(
                torch,
                self.config.torch_dtype,
                self.device,
            ),
            device=self.device,
            decode_audio=self.config.decode_audio,
        )
        self._sampling_class = message.TTSSamplingParams

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        language = model_inputs.get("language", "en_us")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty ZONOS2 language code.")

        speaker_audio_path = model_inputs.get("speaker_audio_path")
        if speaker_audio_path is not None:
            if not isinstance(speaker_audio_path, (str, Path)) or not str(speaker_audio_path).strip():
                raise ValueError("`speaker_audio_path` must be a local audio path or None.")
            reference_path = Path(speaker_audio_path).expanduser()
            if not reference_path.is_file():
                raise FileNotFoundError(f"ZONOS2 reference audio was not found: {reference_path}.")

        repetition_penalty = model_inputs.get("repetition_penalty", 1.2)
        if (not isinstance(repetition_penalty, (int, float)) or isinstance(repetition_penalty, bool) or
                not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
            raise ValueError("`repetition_penalty` must be a finite positive number.")
        temperature = model_inputs.get("temperature", 1.15)
        if (not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or
                not math.isfinite(temperature) or temperature < 0):
            raise ValueError("`temperature` must be a finite non-negative number.")

        for name, default in (("top_p", 0.0), ("min_p", 0.18)):
            value = model_inputs.get(name, default)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or
                    not 0 <= value <= 1):
                raise ValueError(f"`{name}` must be in the interval [0, 1].")

        top_k = model_inputs.get("top_k", 106)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("`top_k` must be a non-negative integer.")
        max_new_tokens = model_inputs.get("max_new_tokens", 1024)
        if (not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer.")
        speed = model_inputs.get("speed")
        if speed is not None and (not isinstance(speed, (int, float)) or isinstance(speed, bool) or
                                  not math.isfinite(speed) or speed <= 0):
            raise ValueError("`speed` must be a finite positive number or None.")
        seed = model_inputs.get("seed")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("`seed` must be an integer or None.")
        if not self.config.decode_audio or model_inputs.get("decode_audio") is False:
            raise ValueError(
                "VoiceHub TTS output requires decoded audio. Set "
                "`decode_audio=True` for ZONOS2 generation.")

    def _sampling_params(
        self,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        min_p: float,
        max_new_tokens: int,
        repetition_penalty: float,
        seed: int | None,
    ):
        return self._sampling_class(
            temperature=temperature,
            topk=top_k,
            top_p=top_p,
            min_p=min_p,
            max_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )

    @staticmethod
    def _first_result(results: Any) -> Mapping[str, Any]:
        if (not isinstance(results, Sequence) or isinstance(results, (str, bytes, bytearray)) or not results):
            raise RuntimeError("ZONOS2 returned no generation results.")
        result = results[0]
        if not isinstance(result, Mapping):
            raise RuntimeError("ZONOS2 returned a malformed generation result.")
        return result

    @staticmethod
    def _decode_audio(audio: Any):
        if audio is None:
            raise RuntimeError(
                "ZONOS2 returned audio tokens without a decoded waveform. "
                "Enable `decode_audio` in the model configuration.")
        if isinstance(audio, (bytes, bytearray, memoryview)):
            if len(audio) == 0:
                raise RuntimeError("ZONOS2 returned an empty audio waveform.")
            numpy = import_optional(
                "numpy",
                model_type="zonos2",
                install_extra="zonos2",
            )
            audio = numpy.frombuffer(audio, dtype=numpy.float32).copy()
        size = getattr(audio, "numel", None)
        sample_count = size() if callable(size) else getattr(audio, "size", None)
        if sample_count == 0:
            raise RuntimeError("ZONOS2 returned an empty audio waveform.")
        return audio

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        speaker_audio_path: str | None = None,
        language: str = "en_us",
        speed: float | None = None,
        temperature: float = 1.15,
        top_k: int = 106,
        top_p: float = 0.0,
        min_p: float = 0.18,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.2,
        seed: int | None = None,
        accurate_mode: bool = True,
        text_normalization: bool = True,
        **generation_options,
    ) -> TTSOutput:
        normalized_language = language.strip().lower().replace("-", "_")
        with seeded_inference(
                seed,
                device=self.device,
                model_type="zonos2",
        ) as effective_seed:
            speaker_embedding = None
            if speaker_audio_path:
                speaker_embedding = self.model.embed_speaker_file(str(Path(speaker_audio_path).expanduser()))
            params = self._sampling_params(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                seed=effective_seed,
            )
            results = self.model.generate(
                [text],
                params,
                language=normalized_language,
                speed=speed,
                speaker_embedding=speaker_embedding,
                accurate_mode=accurate_mode,
                text_normalization=text_normalization,
                **generation_options,
            )
        result = self._first_result(results)
        audio = self._decode_audio(result.get("audio"))
        sample_rate = int(result.get("sample_rate", self.sample_rate))
        if sample_rate <= 0:
            raise RuntimeError(f"ZONOS2 returned an invalid sample rate: {sample_rate}.")
        self.config.sample_rate = sample_rate
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "language": normalized_language,
                "eos_frame": result.get("eos_frame"),
                "voice_cloned": speaker_audio_path is not None,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


Zonos2TTS = Zonos2ForTextToSpeech
