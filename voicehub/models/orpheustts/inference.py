"""Orpheus TTS integration using Transformers and vendored SNAC source."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


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
        self.codec = snac.SNAC.from_pretrained(self.config.codec_name_or_path).to("cpu")
        self.model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                self.config.name_or_path,
                torch_dtype=getattr(torch, self.config.torch_dtype),
            ).to(self.device))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.name_or_path)
        self._torch = torch

    def _prepare_inputs(self, text: str, voice: str):
        torch = self._torch
        input_ids = self.tokenizer(
            f"{voice}: {text}",
            return_tensors="pt",
        ).input_ids
        formatted = torch.cat(
            [
                torch.tensor([[128259]], dtype=torch.int64),
                input_ids,
                torch.tensor(
                    [[128009, 128260]],
                    dtype=torch.int64,
                ),
            ],
            dim=1,
        )
        return (
            formatted.to(self.device),
            torch.ones_like(formatted).to(self.device),
        )

    def _decode_codes(self, codes: list[int]):
        torch = self._torch
        groups = len(codes) // 7
        codes = codes[:groups * 7]
        if not codes:
            raise RuntimeError("Orpheus returned no complete SNAC frames.")
        layer_1 = [codes[7 * index] for index in range(groups)]
        layer_2 = [
            value for index in range(groups) for value in (
                codes[7 * index + 1] - 4096,
                codes[7 * index + 4] - 4 * 4096,
            )
        ]
        layer_3 = [
            value for index in range(groups) for value in (
                codes[7 * index + 2] - 2 * 4096,
                codes[7 * index + 3] - 3 * 4096,
                codes[7 * index + 5] - 5 * 4096,
                codes[7 * index + 6] - 6 * 4096,
            )
        ]
        return self.codec.decode([
            torch.tensor(layer_1).unsqueeze(0),
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0),
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
    ) -> TTSOutput:
        self.load()
        input_ids, attention_mask = self._prepare_inputs(text, voice)
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
                eos_token_id=128258,
            )
        start_positions = (generated_ids == 128257).nonzero(as_tuple=True)
        if len(start_positions[1]) > 0:
            generated_ids = generated_ids[
                :,
                start_positions[1][-1].item() + 1:,
            ]
        clean = generated_ids[0][generated_ids[0] != 128258]
        codes = [token.item() - 128266 for token in clean]
        output = TTSOutput(
            audio=self._decode_codes(codes),
            sample_rate=self.sample_rate,
            metadata={
                "voice": voice,
                "audio_tokens": len(codes)
            },
        )
        if output_file:
            output.save(output_file)
        return output


OrpheusTTS = OrpheusTTSForTextToSpeech
