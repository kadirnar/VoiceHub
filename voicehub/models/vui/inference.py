import torchaudio

from voicehub.models.vui.model import Vui
from voicehub.models.vui.tts import render


class VuiTTS:

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None

    def load_model(self):
        model = Vui.from_pretrained(checkpoint_path=self.model_path).to(self.device)
        self.model = model

    def __call__(self, text: str, output_file: str = "output.wav"):
        if self.model is None:
            self.load_model()
        waveform = render(self.model, text)
        torchaudio.save(output_file, waveform[0], 22050)
