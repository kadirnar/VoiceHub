"""ESPnet ASR inference provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from voicehub.audio import load_audio
from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput
from voicehub.models.asr_native._shared import reject_unsupported_options, resolve_cpu_cuda_device
from voicehub.models.asr_native.configuration import ESPnetASRConfig


class ESPnetASRForSpeechRecognition(PreTrainedASRModel):
    """ESPnet Speech2Text provider with upstream ASRTask training boundary."""

    config_class = ESPnetASRConfig
    default_model_name_or_path = "espnet/kan-bayashi_librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best"
    training_support = "upstream-custom"

    def __init__(
        self,
        config: ESPnetASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        **kwargs,
    ):
        config = self._coerce_config(config, model_path=model_path, **kwargs)
        super().__init__(config, device=device, lazy_load=lazy_load)

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_cpu_cuda_device(device, provider="ESPnet")

    def _load_pretrained_model(self) -> None:
        inference = import_optional(
            "espnet2.bin.asr_inference",
            model_type=self.config.model_type,
            install_extra=None,
        )
        source = self.config.name_or_path or self.default_model_name_or_path
        speech_to_text = getattr(inference, "Speech2Text", None)
        loader = getattr(speech_to_text, "from_pretrained", None)
        if not callable(loader):
            raise RuntimeError(
                "The installed ESPnet package does not expose "
                "Speech2Text.from_pretrained().")
        self.model = loader(
            model_tag=source,
            device=self.device,
            beam_size=self.config.beam_size,
            ctc_weight=self.config.ctc_weight,
            **self.config.model_kwargs,
        )
        if self.model is None or not callable(self.model):
            raise RuntimeError(f"ESPnet could not load an ASR runtime from {source!r}.")

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s=None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords=None,
    ) -> ASROutput:
        reject_unsupported_options(
            "ESPnet Speech2Text",
            task=task,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
        )
        if return_timestamps:
            raise ValueError(
                "The generic ESPnet Speech2Text API does not expose stable "
                "segment timestamps across all recipes.")
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=self.sample_rate,
        )
        hypotheses = self.model(materialized.waveform)
        if not hypotheses:
            text = ""
        else:
            first = hypotheses[0]
            text = first[0] if isinstance(first, (tuple, list)) else str(first)
        return ASROutput(
            text=str(text),
            language=language,
            duration=materialized.duration,
            metadata={"backend": "espnet"},
        )

    def _validate_training_runtime(self) -> None:
        raise ValueError(
            "ESPnet ASR fine-tuning requires its upstream ASRTask recipe so "
            "tokenizer, language model, augmentation, and exact resume state "
            "remain consistent.")
