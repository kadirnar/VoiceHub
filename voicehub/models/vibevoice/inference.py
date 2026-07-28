"""VibeVoice realtime inference backed by vendored Microsoft source."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel
from voicehub.models._shared import finish_audio_output, resolve_model_directory, resolve_torch_dtype, seeded_inference


class VibeVoiceConfig(VoiceHubConfig):
    """Configuration for the VibeVoice realtime streaming architecture."""

    model_type = "vibevoice"

    def __init__(
        self,
        *,
        torch_dtype: str = "bfloat16",
        attention_implementation: str = "sdpa",
        diffusion_steps: int = 5,
        training_ce_loss_weight: float = 1.0,
        training_diffusion_loss_weight: float = 1.0,
        training_ddpm_batch_mul: int = 1,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.diffusion_steps = diffusion_steps
        self.training_ce_loss_weight = training_ce_loss_weight
        self.training_diffusion_loss_weight = training_diffusion_loss_weight
        self.training_ddpm_batch_mul = training_ddpm_batch_mul


class VibeVoiceForTextToSpeech(PreTrainedTTSModel):
    """Single-speaker realtime VibeVoice generation with cached voice
    prompts."""

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
        self._runtime_kind = None
        super().__init__(config, device=device, lazy_load=lazy_load)

    def _load_pretrained_model(self) -> None:
        torch = import_optional(
            "torch",
            model_type="vibevoice",
            install_extra=None,
        )
        model_directory = resolve_model_directory(
            self.config.name_or_path,
            model_type="vibevoice",
        )
        dtype = resolve_torch_dtype(
            torch,
            self.config.torch_dtype,
            self.device,
        )
        if self.is_training_load:
            self._load_non_streaming_training_runtime(
                torch,
                model_directory,
                dtype,
            )
            return
        model_module = import_optional(
            "voicehub.models.vibevoice.source.vibevoice.modular."
            "modeling_vibevoice_streaming_inference",
            model_type="vibevoice",
            install_extra=None,
        )
        processor_module = import_optional(
            "voicehub.models.vibevoice.source.vibevoice.processor."
            "vibevoice_streaming_processor",
            model_type="vibevoice",
            install_extra=None,
        )
        modeling_outputs = import_optional(
            "transformers.modeling_outputs",
            model_type="vibevoice",
            install_extra=None,
        )
        cache_utils = import_optional(
            "transformers.cache_utils",
            model_type="vibevoice",
            install_extra=None,
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
        self.config.sample_rate = self._checkpoint_sample_rate(self._processor)
        self._safe_globals = (
            modeling_outputs.BaseModelOutputWithPast,
            cache_utils.DynamicCache,
        )
        self._torch = torch
        self._runtime_kind = "streaming"
        self.model = model

    def _load_non_streaming_training_runtime(
        self,
        torch,
        model_directory: Path,
        dtype,
    ) -> None:
        """Load the verified 1.5B graph instead of the realtime runtime."""
        configuration_path = model_directory / "config.json"
        if not configuration_path.is_file():
            raise FileNotFoundError(f"VibeVoice training config was not found: {configuration_path}.")
        configuration = json.loads(configuration_path.read_text(encoding="utf-8"), )
        if str(configuration.get("model_type", "")).lower() != "vibevoice":
            raise ValueError(
                "VibeVoice fine-tuning supports only the non-streaming "
                "`microsoft/VibeVoice-1.5B` architecture; the selected "
                f"checkpoint declares model_type={configuration.get('model_type')!r}.")
        model_module = import_optional(
            "voicehub.models.vibevoice.source.vibevoice.modular."
            "modeling_vibevoice",
            model_type="vibevoice",
            install_extra="training",
        )
        processor_module = import_optional(
            "voicehub.models.vibevoice.source.vibevoice.processor."
            "vibevoice_processor",
            model_type="vibevoice",
            install_extra="training",
        )
        model = model_module.VibeVoiceForConditionalGeneration.from_pretrained(
            str(model_directory),
            torch_dtype=dtype,
            device_map=None,
            attn_implementation=self.config.attention_implementation,
        )
        model.to(self.device)
        self._processor = processor_module.VibeVoiceProcessor.from_pretrained(str(model_directory), )
        self._processor.acoustic_tokenizer = model.model.acoustic_tokenizer
        self._processor.semantic_tokenizer = model.model.semantic_tokenizer
        self.config.sample_rate = self._checkpoint_sample_rate(self._processor)
        self._torch = torch
        self._runtime_kind = "non-streaming-training"
        self.model = model
        self._prepare_for_training()

    def _validate_training_runtime(self) -> None:
        """Reject the streaming checkpoint before allocating its fused
        graph."""
        identifier = str(self.config.name_or_path)
        source = Path(identifier).expanduser()
        if source.is_dir():
            configuration_path = source / "config.json"
            if not configuration_path.is_file():
                raise FileNotFoundError(
                    "A local VibeVoice training directory must contain "
                    f"`config.json`: {configuration_path}.")
            configuration = json.loads(configuration_path.read_text(encoding="utf-8"), )
            if str(configuration.get("model_type", "")).lower() != "vibevoice":
                raise ValueError(
                    "VibeVoice fine-tuning requires the non-streaming 1.5B "
                    "architecture (`model_type=\"vibevoice\"`).")
            return
        if source.exists():
            raise NotADirectoryError(
                "VibeVoice fine-tuning expects a Hub ID or checkpoint directory, "
                f"not a file: {source}.")
        if identifier.strip().lower() != "microsoft/vibevoice-1.5b":
            raise ValueError(
                "VibeVoice fine-tuning is verified only for the non-streaming "
                "`microsoft/VibeVoice-1.5B` checkpoint. The default "
                "`VibeVoice-Realtime-0.5B` runtime has no unified training "
                "forward graph.")

    def _prepare_for_training(self) -> None:
        if self._runtime_kind != "non-streaming-training":
            raise ValueError("VibeVoice fine-tuning requires the non-streaming 1.5B runtime.")
        self.model.train()
        for name in ("acoustic_tokenizer", "semantic_tokenizer"):
            tokenizer = getattr(self.model.model, name, None)
            if tokenizer is None:
                raise RuntimeError(f"VibeVoice training runtime is missing `{name}`.")
            tokenizer.eval()
            for parameter in tokenizer.parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _checkpoint_sample_rate(processor: Any) -> int:
        audio_processor = getattr(processor, "audio_processor", None)
        sample_rate = getattr(audio_processor, "sampling_rate", None)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise RuntimeError(
                "The VibeVoice checkpoint processor does not define an "
                "integer audio sampling rate.")
        if sample_rate <= 0:
            raise RuntimeError(
                "The VibeVoice checkpoint processor defines an invalid "
                f"audio sampling rate: {sample_rate}.")
        return sample_rate

    def _validate_generation_inputs(self, model_inputs: dict[str, Any]) -> None:
        diffusion_steps = self.config.diffusion_steps
        if (not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool) or
                diffusion_steps <= 0):
            raise ValueError("`diffusion_steps` must be a positive integer.")
        voice_prompt_path = model_inputs.get("voice_prompt_path")
        if not isinstance(voice_prompt_path, (str, Path)) or not str(voice_prompt_path).strip():
            raise ValueError("`voice_prompt_path` must point to a cached VibeVoice `.pt` prompt.")
        prompt_path = Path(voice_prompt_path).expanduser()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"VibeVoice cached voice prompt was not found: {prompt_path}.")

        cfg_scale = model_inputs.get("cfg_scale", 1.5)
        if (not isinstance(cfg_scale, (int, float)) or isinstance(cfg_scale, bool) or
                not math.isfinite(cfg_scale) or cfg_scale <= 0):
            raise ValueError("`cfg_scale` must be a finite positive number.")
        max_new_tokens = model_inputs.get("max_new_tokens")
        if max_new_tokens is not None and (not isinstance(max_new_tokens, int) or
                                           isinstance(max_new_tokens, bool) or max_new_tokens <= 0):
            raise ValueError("`max_new_tokens` must be a positive integer or None.")

    def _load_cached_prompt(self, voice_prompt_path: str) -> Mapping[str, Any]:
        with self._torch.serialization.safe_globals(list(self._safe_globals)):
            cached_prompt = self._torch.load(
                str(Path(voice_prompt_path).expanduser()),
                map_location=self.device,
                weights_only=True,
            )
        if not isinstance(cached_prompt, Mapping):
            raise TypeError("VibeVoice cached prompt must contain a mapping.")
        required_sections = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
        missing = [name for name in required_sections if name not in cached_prompt]
        if missing:
            raise ValueError(
                "VibeVoice cached prompt is missing required section(s): " + ", ".join(missing) + ".")
        return cached_prompt

    def _move_inputs_to_device(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if self._torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

    @staticmethod
    def _generation_kwargs(
        generation_options: Mapping[str, Any],
        *,
        max_new_tokens: int | None,
        cfg_scale: float,
        tokenizer: Any,
        cached_prompt: Mapping[str, Any],
    ) -> dict[str, Any]:
        reserved = {
            "all_prefilled_outputs",
            "cfg_scale",
            "max_new_tokens",
            "tokenizer",
        }
        conflicts = sorted(reserved & set(generation_options))
        if conflicts:
            raise ValueError(
                "VibeVoice generation option(s) are managed by the wrapper: " + ", ".join(conflicts) + ".")
        options = dict(generation_options)
        options.setdefault("generation_config", {"do_sample": False})
        options.update({
            "cfg_scale": cfg_scale,
            "tokenizer": tokenizer,
            "all_prefilled_outputs": copy.deepcopy(cached_prompt),
        })
        if max_new_tokens is not None:
            options["max_new_tokens"] = max_new_tokens
        return options

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        voice_prompt_path: str | None = None,
        cfg_scale: float = 1.5,
        max_new_tokens: int | None = None,
        seed: int | None = None,
        **generation_options,
    ) -> TTSOutput:
        cached_prompt = self._load_cached_prompt(voice_prompt_path)
        inputs = self._processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = self._move_inputs_to_device(inputs)
        options = self._generation_kwargs(
            generation_options,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            tokenizer=self._processor.tokenizer,
            cached_prompt=cached_prompt,
        )
        with seeded_inference(
                seed,
                device=self.device,
                model_type="vibevoice",
        ) as effective_seed:
            outputs = self.model.generate(
                **inputs,
                **options,
            )
        speech_outputs = getattr(outputs, "speech_outputs", None)
        if not speech_outputs or speech_outputs[0] is None:
            raise RuntimeError("VibeVoice did not return an audio waveform.")
        waveform = speech_outputs[0]
        if not hasattr(waveform, "detach"):
            raise RuntimeError("VibeVoice returned a non-tensor audio waveform.")
        if hasattr(waveform, "numel") and waveform.numel() == 0:
            raise RuntimeError("VibeVoice returned an empty audio waveform.")
        return finish_audio_output(
            waveform.detach().float().cpu(),
            self.sample_rate,
            output_file=output_file,
            metadata={
                "cfg_scale": cfg_scale,
                "seed": effective_seed,
                "requested_seed": seed,
            },
        )


VibeVoiceTTS = VibeVoiceForTextToSpeech
