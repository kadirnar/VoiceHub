from voicehub.base_model import BaseTTSModel
from voicehub.models.vui.model import Vui
from voicehub.models.vui.tts import render


class VuiTTS(BaseTTSModel):
    """High-level TTS interface for the Vui model.

    Lazily loads the checkpoint on first call so that instantiation is cheap.
    """

    def __init__(self, model_path: str = "", device: str = "cuda"):
        """Create a VuiTTS wrapper.

        Args:
            model_path: HuggingFace repo filename or local checkpoint path.
            device: Torch device string (``"cuda"`` or ``"cpu"``).
        """
        super().__init__(model_path, device)
        self.model = None

    @property
    def sample_rate(self) -> int:
        return 22050

    def load_model(self):
        """Load the Vui checkpoint and move the model to the configured device."""
        model = Vui.from_pretrained(checkpoint_path=self.model_path).to(self.device)
        self.model = model

    def __call__(self, text: str, output_file: str = "output.wav"):
        """Synthesise speech from *text* and write it to *output_file* (22 050 Hz WAV)."""
        if self.model is None:
            self.load_model()
        waveform = render(self.model, text)
        self.save_audio(output_file, waveform[0], self.sample_rate)
