"""ConversationTTS registration and source-licensing guard."""

from __future__ import annotations

from voicehub.configuration_utils import VoiceHubConfig
from voicehub.errors import SourceLicenseError
from voicehub.modeling_outputs import TTSOutput
from voicehub.modeling_utils import PreTrainedTTSModel


class ConversationTTSConfig(VoiceHubConfig):
    """Configuration placeholder for the ConversationTTS architecture."""

    model_type = "conversationtts"

    def __init__(
        self,
        *,
        text_tokenizer_path: str = "",
        audio_tokenizer_path: str = "",
        experiment_directory: str = "",
        model_args: dict | None = None,
        dtype: str = "bfloat16",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.text_tokenizer_path = text_tokenizer_path
        self.audio_tokenizer_path = audio_tokenizer_path
        self.experiment_directory = experiment_directory
        self.model_args = model_args or {}
        self.dtype = dtype


class ConversationTTSForTextToSpeech(PreTrainedTTSModel):
    """Reserved integration blocked until upstream grants a source license."""

    config_class = ConversationTTSConfig

    def __init__(
        self,
        config: ConversationTTSConfig | str | None = None,
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
        raise SourceLicenseError(
            "Audio-Foundation-Models/ConversationTTS does not publish a "
            "source-code license. VoiceHub cannot legally vendor or "
            "redistribute that implementation until upstream adds one."
        )

    def _generate(self, text: str, **kwargs) -> TTSOutput:
        self.load()
        raise AssertionError("ConversationTTS loading must raise.")


ConversationTTS = ConversationTTSForTextToSpeech
