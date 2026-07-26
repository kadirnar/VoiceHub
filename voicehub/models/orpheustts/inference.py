"""Orpheus TTS integration using Transformers and vendored SNAC source."""

from __future__ import annotations

import math
from numbers import Real

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_torch_dtype, seeded_inference


class OrpheusTTSConfig(VoiceHubConfig):
    """Configuration for Orpheus language model and SNAC decoder."""

    model_type = "orpheustts"

    def __init__(
        self,
        *,
        codec_name_or_path: str = "hubertsiuzdak/snac_24khz",
        torch_dtype: str = "bfloat16",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.codec_name_or_path = codec_name_or_path
        self.torch_dtype = torch_dtype


class OrpheusTTSForTextToSpeech(PreTrainedTTSModel):
    """Expressive speech generation without a TTS pip package."""

    config_class = OrpheusTTSConfig
    default_model_name_or_path = "canopylabs/orpheus-3b-0.1-ft"
    _START_HUMAN_TOKEN_ID = 128259
    _END_TEXT_TOKEN_ID = 128009
    _END_HUMAN_TOKEN_ID = 128260
    _START_SPEECH_TOKEN_ID = 128257
    _END_SPEECH_TOKEN_ID = 128258
    _AUDIO_TOKEN_OFFSET = 128266
    _SNAC_CODEBOOK_SIZE = 4096
    _SNAC_FRAME_WIDTH = 7

    def __init__(
        self,
        config: OrpheusTTSConfig | str | None = None,
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
        self.tokenizer = None
        self.codec = None
        self._torch = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="orpheustts",
            install_extra="orpheustts",
        )
        transformers = import_optional(
            "transformers",
            model_type="orpheustts",
            install_extra="orpheustts",
        )
        snac = import_optional(
            "voicehub.models.orpheustts.source.snac",
            model_type="orpheustts",
            install_extra="orpheustts",
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        self.codec = (snac.SNAC.from_pretrained(self.config.codec_name_or_path).eval().to("cpu"))
        sample_rate = int(getattr(self.codec, "sampling_rate", 0))
        if sample_rate <= 0:
            raise RuntimeError("Orpheus SNAC codec reported an invalid sample rate: "
                               f"{sample_rate}.")
        self.config.sample_rate = sample_rate
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                self.config.name_or_path,
                torch_dtype=dtype,
            ).eval().to(self.device))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.name_or_path)
        self._torch = torch

    def _prepare_for_inference(self) -> None:
        """Restore deterministic serving state without replacing trained
        weights."""
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        if self.codec is not None and hasattr(self.codec, "eval"):
            self.codec.eval()
        model_config = getattr(self.model, "config", None)
        if model_config is not None and hasattr(model_config, "use_cache"):
            model_config.use_cache = True

    def _validate_generation_inputs(self, model_inputs: dict) -> None:
        voice = model_inputs.get("voice")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("Orpheus generation requires a non-empty `voice`.")
        temperature = model_inputs.get("temperature", 0.6)
        if (isinstance(temperature, bool) or not isinstance(temperature, Real) or
                not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Orpheus sampling requires `temperature` greater than zero.")
        top_p = model_inputs.get("top_p", 0.95)
        if (isinstance(top_p, bool) or not isinstance(top_p, Real) or not math.isfinite(top_p) or
                not 0 < top_p <= 1):
            raise ValueError("Orpheus sampling requires `top_p` in the interval (0, 1].")
        repetition_penalty = model_inputs.get("repetition_penalty", 1.1)
        if (isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, Real) or
                not math.isfinite(repetition_penalty) or repetition_penalty <= 0):
            raise ValueError("`repetition_penalty` must be a finite positive number.")

    def _prepare_inputs(self, text: str, voice: str):
        torch = self._torch
        input_ids = self.tokenizer(
            f"{voice}: {text}",
            return_tensors="pt",
        ).input_ids
        formatted = torch.cat(
            [
                torch.tensor(
                    [[self._START_HUMAN_TOKEN_ID]],
                    dtype=torch.int64,
                ),
                input_ids,
                torch.tensor(
                    [[self._END_TEXT_TOKEN_ID, self._END_HUMAN_TOKEN_ID]],
                    dtype=torch.int64,
                ),
            ],
            dim=1,
        )
        return (
            formatted.to(self.device),
            torch.ones_like(formatted).to(self.device),
        )

    @classmethod
    def _extract_audio_codes(cls, generated_ids) -> list[int]:
        """Extract one complete Orpheus speech span from generated tokens."""
        tokens = [int(token) for token in generated_ids[0].detach().cpu().tolist()]
        try:
            speech_start = len(tokens) - 1 - tokens[::-1].index(cls._START_SPEECH_TOKEN_ID)
        except ValueError as exc:
            raise RuntimeError(
                "Orpheus returned no speech-start token; generation did not "
                "produce a decodable audio span.") from exc

        speech_tokens = tokens[speech_start + 1:]
        if cls._END_SPEECH_TOKEN_ID in speech_tokens:
            speech_tokens = speech_tokens[:speech_tokens.index(cls._END_SPEECH_TOKEN_ID)]
        codes = [
            token - cls._AUDIO_TOKEN_OFFSET for token in speech_tokens if token >= cls._AUDIO_TOKEN_OFFSET
        ]
        complete_length = len(codes) - len(codes) % cls._SNAC_FRAME_WIDTH
        if complete_length == 0:
            raise RuntimeError("Orpheus returned no complete SNAC frames.")
        codes = codes[:complete_length]
        for position, code in enumerate(codes):
            channel = position % cls._SNAC_FRAME_WIDTH
            lower_bound = channel * cls._SNAC_CODEBOOK_SIZE
            upper_bound = lower_bound + cls._SNAC_CODEBOOK_SIZE
            if not lower_bound <= code < upper_bound:
                raise RuntimeError(
                    "Orpheus generated an invalid SNAC token for channel "
                    f"{channel}: {code + cls._AUDIO_TOKEN_OFFSET}.")
        return codes

    def _decode_codes(self, codes: list[int]):
        torch = self._torch
        groups = len(codes) // self._SNAC_FRAME_WIDTH
        if groups == 0:
            raise RuntimeError("Orpheus returned no complete SNAC frames.")
        layer_1 = [codes[self._SNAC_FRAME_WIDTH * index] for index in range(groups)]
        layer_2 = [
            value for index in range(groups) for value in (
                codes[self._SNAC_FRAME_WIDTH * index + 1] - self._SNAC_CODEBOOK_SIZE,
                codes[self._SNAC_FRAME_WIDTH * index + 4] - 4 * self._SNAC_CODEBOOK_SIZE,
            )
        ]
        layer_3 = [
            value for index in range(groups) for value in (
                codes[self._SNAC_FRAME_WIDTH * index + 2] - 2 * self._SNAC_CODEBOOK_SIZE,
                codes[self._SNAC_FRAME_WIDTH * index + 3] - 3 * self._SNAC_CODEBOOK_SIZE,
                codes[self._SNAC_FRAME_WIDTH * index + 5] - 5 * self._SNAC_CODEBOOK_SIZE,
                codes[self._SNAC_FRAME_WIDTH * index + 6] - 6 * self._SNAC_CODEBOOK_SIZE,
            )
        ]
        return self.codec.decode([
            torch.tensor(layer_1, dtype=torch.long).unsqueeze(0),
            torch.tensor(layer_2, dtype=torch.long).unsqueeze(0),
            torch.tensor(layer_3, dtype=torch.long).unsqueeze(0),
        ])

    def _generate(
        self,
        text: str,
        *,
        voice: str,
        output_file: str | None = None,
        max_new_tokens: int = 1200,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        seed: int | None = None,
    ) -> TTSOutput:
        input_ids, attention_mask = self._prepare_inputs(text, voice)
        with seeded_inference(
                seed,
                device=self.device,
                model_type="orpheustts",
        ) as effective_seed:
            with self._torch.inference_mode():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=1,
                    eos_token_id=self._END_SPEECH_TOKEN_ID,
                )
                codes = self._extract_audio_codes(generated_ids)
                audio = self._decode_codes(codes)
        return finish_audio_output(
            audio,
            self.sample_rate,
            output_file=output_file,
            metadata={
                "voice": voice,
                "audio_tokens": len(codes),
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


OrpheusTTS = OrpheusTTSForTextToSpeech
