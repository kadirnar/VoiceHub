"""GPT-SoVITS integration backed by vendored upstream source."""

from __future__ import annotations

from pathlib import Path

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


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

    def _load_pretrained_model(self) -> None:
        runtime = import_optional(
            (
                "voicehub.models.gptsovits.source.GPT_SoVITS."
                "TTS_infer_pack.TTS"
            ),
            model_type="gptsovits",
            install_extra="gptsovits",
        )
        config_source = self.config.runtime_config
        if config_source is None:
            config_path = (
                Path(self.config.name_or_path).expanduser()
                if self.config.name_or_path
                else (
                    Path(__file__).parent
                    / "source"
                    / "GPT_SoVITS"
                    / "configs"
                    / "tts_infer.yaml"
                )
            )
            if not config_path.is_file():
                raise FileNotFoundError(
                    "GPT-SoVITS requires a local inference YAML or "
                    "`runtime_config` containing checkpoint paths."
                )
            config_source = str(config_path)

        runtime_config = runtime.TTS_Config(config_source)
        runtime_config.device = self.device
        self.model = runtime.TTS(runtime_config)

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
        self.load()
        request = {
            "text": text,
            "text_lang": text_language,
            "ref_audio_path": speaker_audio_path,
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
        results = list(self.model.run(request))
        if not results:
            raise RuntimeError("GPT-SoVITS returned no audio.")

        np = import_optional(
            "numpy",
            model_type="gptsovits",
            install_extra="gptsovits",
        )
        sample_rates = {sample_rate for sample_rate, _ in results}
        if len(sample_rates) != 1:
            raise RuntimeError(
                "GPT-SoVITS returned chunks with different sample rates."
            )
        self.config.sample_rate = sample_rates.pop()
        output = TTSOutput(
            audio=np.concatenate(
                [np.asarray(chunk).reshape(-1) for _, chunk in results]
            ),
            sample_rate=self.sample_rate,
            metadata={"seed": seed},
        )
        if output_file:
            output.save(output_file)
        return output


GPTSoVITSTTS = GPTSoVITSForTextToSpeech
