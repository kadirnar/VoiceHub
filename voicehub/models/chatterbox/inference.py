import torchaudio

from voicehub.models.chatterbox.tts import ChatterboxTTS


class ChatterboxInference:
    """A class for running text-to-speech inference with the Chatterbox model."""

    def __init__(self):
        """Initializes the ChatterboxInference class."""
        self.model = None
        self.sample_rate = None
        self.device = None

    def load_model(self, device="cuda"):
        """
        Loads the ChatterboxTTS model onto the specified device.

        Args:
            device (str): The device to run the model on, e.g., "cuda" or "cpu".
        """
        self.device = device
        print(f"Loading ChatterboxTTS model on {self.device}...")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.sample_rate = self.model.sr
        print("Model loaded successfully.")

    def __call__(self, text, output_path, audio_prompt_path=None):
        """
        Synthesizes speech from text and saves it to a file.

        Args:
            text (str): The text to synthesize.
            output_path (str): The path to save the generated audio file.
            audio_prompt_path (str, optional): Path to an audio file for voice prompt. Defaults to None.
        """
        if not self.model:
            raise RuntimeError("Model is not loaded. Please call load_model() first.")

        print(f"Synthesizing text: '{text}'")
        if audio_prompt_path:
            print(f"Using audio prompt: {audio_prompt_path}")

        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        torchaudio.save(output_path, wav, self.sample_rate)
        print(f"Audio saved to {output_path}")
