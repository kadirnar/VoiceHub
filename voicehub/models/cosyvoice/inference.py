"""CosyVoice 1/2/3 integration backed by vendored upstream source."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class CosyVoiceConfig(VoiceHubConfig):
    """Configuration shared by CosyVoice checkpoint generations."""

    model_type = "cosyvoice"

    def __init__(
        self,
        *,
        load_jit: bool = False,
        load_trt: bool = False,
        load_vllm: bool = False,
        fp16: bool = False,
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.load_jit = load_jit
        self.load_trt = load_trt
        self.load_vllm = load_vllm
        self.fp16 = fp16


class CosyVoiceForTextToSpeech(PreTrainedTTSModel):
    """Unified source-native CosyVoice synthesis interface."""

    config_class = CosyVoiceConfig
    default_model_name_or_path = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"

    def __init__(
        self,
        config: CosyVoiceConfig | str | None = None,
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
        runtime = import_optional(
            "voicehub.models.cosyvoice.source.cosyvoice.cli.cosyvoice",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        model_directory = Path(self.config.name_or_path).expanduser()
        if not model_directory.is_dir():
            model_directory = Path(runtime.snapshot_download(self.config.name_or_path))

        common = {
            "model_dir": str(model_directory),
            "load_trt": self.config.load_trt,
            "fp16": self.config.fp16,
        }
        if (model_directory / "cosyvoice3.yaml").is_file():
            model_class = runtime.CosyVoice3
            common["load_vllm"] = self.config.load_vllm
        elif (model_directory / "cosyvoice2.yaml").is_file():
            model_class = runtime.CosyVoice2
            common["load_jit"] = self.config.load_jit
            common["load_vllm"] = self.config.load_vllm
        elif (model_directory / "cosyvoice.yaml").is_file():
            model_class = runtime.CosyVoice
            common["load_jit"] = self.config.load_jit
        else:
            raise ValueError(f"{model_directory} does not contain a CosyVoice model config.")

        self.model = model_class(**common)
        self.config.name_or_path = str(model_directory)
        self.config.sample_rate = self.model.sample_rate

    def _select_inference(
        self,
        *,
        mode: str,
        text: str,
        speaker: str | None,
        prompt_text: str,
        speaker_audio_path: str | None,
        instruct_text: str,
        stream: bool,
        speed: float,
    ):
        if mode == "auto":
            mode = "zero_shot" if speaker_audio_path else "sft"

        common = {"stream": stream, "speed": speed}
        if mode == "sft":
            if speaker is None:
                speakers = self.model.list_available_spks()
                if not speakers:
                    raise ValueError(
                        "This checkpoint has no built-in speaker; provide "
                        "speaker_audio_path for zero-shot synthesis.")
                speaker = speakers[0]
            return self.model.inference_sft(text, speaker, **common)
        if mode == "zero_shot":
            if not speaker_audio_path or not prompt_text:
                raise ValueError("zero_shot requires speaker_audio_path and prompt_text.")
            return self.model.inference_zero_shot(
                text,
                prompt_text,
                speaker_audio_path,
                **common,
            )
        if mode == "cross_lingual":
            if not speaker_audio_path:
                raise ValueError("cross_lingual requires speaker_audio_path.")
            return self.model.inference_cross_lingual(
                text,
                speaker_audio_path,
                **common,
            )
        if mode == "instruct":
            if hasattr(self.model, "inference_instruct2"):
                if not speaker_audio_path:
                    raise ValueError("CosyVoice 2/3 instruct mode requires "
                                     "speaker_audio_path.")
                return self.model.inference_instruct2(
                    text,
                    instruct_text,
                    speaker_audio_path,
                    **common,
                )
            if speaker is None:
                raise ValueError("CosyVoice 1 instruct mode requires speaker.")
            return self.model.inference_instruct(
                text,
                speaker,
                instruct_text,
                **common,
            )
        raise ValueError("mode must be one of: auto, sft, zero_shot, cross_lingual, "
                         "instruct.")

    def _generate(
        self,
        text: str,
        *,
        output_file: str | None = None,
        mode: str = "auto",
        speaker: str | None = None,
        prompt_text: str = "",
        speaker_audio_path: str | None = None,
        instruct_text: str = "",
        stream: bool = False,
        speed: float = 1.0,
    ) -> TTSOutput:
        self.load()
        generator = self._select_inference(
            mode=mode,
            text=text,
            speaker=speaker,
            prompt_text=prompt_text,
            speaker_audio_path=speaker_audio_path,
            instruct_text=instruct_text,
            stream=stream,
            speed=speed,
        )
        chunks = [item["tts_speech"].reshape(-1) for item in generator]
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks.")

        torch = import_optional(
            "torch",
            model_type="cosyvoice",
            install_extra="cosyvoice",
        )
        output = TTSOutput(
            audio=torch.cat(chunks),
            sample_rate=self.sample_rate,
            metadata={"mode": mode},
        )
        if output_file:
            output.save(output_file)
        return output


CosyVoiceTTS = CosyVoiceForTextToSpeech
