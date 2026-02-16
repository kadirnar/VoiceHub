from voicehub.base_model import BaseTTSModel
from voicehub.models.chatterbox.tts import ChatterboxTTS


class ChatterboxInference(BaseTTSModel):
    """A class for running text-to-speech inference with the Chatterbox model."""

    def __init__(self, model_path: str = "", device: str = "cuda"):
        """Initializes the ChatterboxInference class.

        Args:
            model_path: Unused, kept for interface compatibility.
            device: The device to run the model on, e.g., "cuda" or "cpu".
        """
        super().__init__(model_path, device)
        print(f"Loading ChatterboxTTS model on {self.device}...")
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        print("Model loaded successfully.")

    @property
    def sample_rate(self) -> int:
        return self.model.sr

    def __call__(self, text, output_file="output.wav", audio_prompt_path=None):
        """
        Synthesizes speech from text and saves it to a file.

        Args:
            text (str): The text to synthesize.
            output_file (str): The path to save the generated audio file.
            audio_prompt_path (str, optional): Path to an audio file for voice prompt. Defaults to None.
        """
        print(f"Synthesizing text: '{text}'")
        if audio_prompt_path:
            print(f"Using audio prompt: {audio_prompt_path}")

        wav = self.model.generate(text, audio_prompt_path=audio_prompt_path)
        self.save_audio(output_file, wav, self.sample_rate)
        print(f"Audio saved to {output_file}")
