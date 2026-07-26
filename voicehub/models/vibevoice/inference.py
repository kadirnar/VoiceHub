"""VibeVoice realtime inference backed by vendored Microsoft source."""

from __future__ import annotations

import copy

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, resolve_torch_dtype


class VibeVoiceConfig(VoiceHubConfig):
    """Configuration for the VibeVoice realtime streaming architecture."""

    model_type = "vibevoice"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        diffusion_steps: int = 5,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.diffusion_steps = diffusion_steps


class VibeVoiceForTextToSpeech(PreTrainedTTSModel):
    """Single-speaker realtime VibeVoice generation with cached voice prompts."""

    config_class = VibeVoiceConfig
    default_model_name_or_path = "microsoft/VibeVoice-Realtime-0.5B"

    def __init__(
        self,
        config: VibeVoiceConfig | str | None = None,
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
        self._torch = None
        self._processor = None
        self._safe_globals = ()
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="vibevoice",
            install_extra="vibevoice",
        )
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="vibevoice",
        )
        model_module = import_optional(
            "voicehub.models.vibevoice.source.vibevoice.modular."
            "modeling_vibevoice_streaming_inference",
            model_type="vibevoice",
            install_extra="vibevoice",
        )
        processor_module = import_optional(
            "voicehub.models.vibevoice.source.vibevoice.processor."
            "vibevoice_streaming_processor",
            model_type="vibevoice",
            install_extra="vibevoice",
        )
        modeling_outputs = import_optional(
            "transformers.modeling_outputs",
            model_type="vibevoice",
            install_extra="vibevoice",
        )
        cache_utils = import_optional(
            "transformers.cache_utils",
            model_type="vibevoice",
            install_extra="vibevoice",
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        model = (
            model_module.VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                str(model_directory),
                torch_dtype=dtype,
                device_map=None,
                attn_implementation=self.config.attention_implementation,
            ))
        model.to(self.device).eval()
        model.set_ddpm_inference_steps(num_steps=self.config.diffusion_steps)
        self._processor = (processor_module.VibeVoiceStreamingProcessor.from_pretrained(str(model_directory)))
        self._safe_globals = (
            modeling_outputs.BaseModelOutputWithPast,
            cache_utils.DynamicCache,
        )
        self._torch = torch
        self.model = model

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        voice_prompt_path: str | None = None,
        cfg_scale: float = 1.5,
        max_new_tokens: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        self.load()
        if not voice_prompt_path:
            raise ValueError(
                "VibeVoice requires voice_prompt_path pointing to a cached "
                "realtime .pt voice prompt.")
        with self._torch.serialization.safe_globals(list(self._safe_globals)):
            cached_prompt = self._torch.load(
                voice_prompt_path,
                map_location=self.device,
                weights_only=True,
            )
        inputs = self._processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            key: value.to(self.device) if self._torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            tokenizer=self._processor.tokenizer,
            generation_config={"do_sample": False},
            all_prefilled_outputs=copy.deepcopy(cached_prompt),
            **generation_options,
        )
        if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
            raise RuntimeError("VibeVoice did not return an audio waveform.")
        return finish_audio_output(
            outputs.speech_outputs[0].detach().float().cpu(),
            self.sample_rate,
            output_file=output_file,
            metadata={"cfg_scale": cfg_scale},
        )


VibeVoiceTTS = VibeVoiceForTextToSpeech
