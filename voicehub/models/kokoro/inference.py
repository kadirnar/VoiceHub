from kokoro import KPipeline

from voicehub.base_model import BaseTTSModel


class KokoroTTS(BaseTTSModel):
    """
    KokoroTTS class for text-to-speech generation using the Kokoro model.

    This class provides a simple interface for loading and using the Kokoro model
    to generate speech from text prompts.

    Example:
        ```python
        # Initialize the KokoroTTS model
        # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English, 🇪🇸 'e' => Spanish, etc.
        tts = KokoroTTS(lang_code="a")

        # Generate speech from text
        text = "The sky above the port was the color of television, tuned to a dead channel."
        audios = tts(text=text, voice="af_heart", output_prefix="output")

        # To listen in a notebook:
        # from IPython.display import Audio, display
        # display(Audio(audios[0], rate=24000))
        ```
    """

    def __init__(self, model_path: str = "", device: str = "cuda", lang_code: str = "a"):
        """
        Initialize the KokoroTTS model.

        Args:
            model_path (str): Unused, kept for interface compatibility.
            device (str): Unused, kept for interface compatibility.
            lang_code (str): Language code for the model. Default is "a".
                - 'a': American English
                - 'b': British English
                - 'e': Spanish
                - 'f': French
                - 'h': Hindi
                - 'i': Italian
                - 'j': Japanese (requires `pip install misaki[ja]`)
                - 'p': Brazilian Portuguese
                - 'z': Mandarin Chinese (requires `pip install misaki[zh]`)
        """
        super().__init__(model_path, device)
        self.pipeline = KPipeline(lang_code=lang_code)

    @property
    def sample_rate(self) -> int:
        return 24000

    def __call__(
            self,
            text: str,
            voice: str = "af_heart",
            speed: float = 1.0,
            output_prefix: str = "output",
            split_pattern: str = r'\n+'):
        """
        Generate speech from text and save it to files.

        Args:
            text (str): Text to convert to speech.
            voice (str): The voice to use for generation. Default is "af_heart".
                         Can also be a path to a voice tensor.
            speed (float): Speaking speed. Default is 1.0.
            output_prefix (str): Prefix for the output audio files.
                                 Files will be saved as {output_prefix}_0.wav, etc.
            split_pattern (str): Regex pattern to split the input text into segments.
                                 Default is r'\n+'.

        Returns:
            list: A list of audio data numpy arrays.
        """
        generator = self.pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)

        generated_audios = []
        print("Generating audio...")
        for i, (graphemes, phonemes, audio) in enumerate(generator):
            print(f"  - Segment {i}: {repr(graphemes)}")
            output_file = f"{output_prefix}_{i}.wav"
            self.save_audio(output_file, audio, self.sample_rate)
            print(f"    Saved to {output_file}")
            generated_audios.append(audio)

        print("Audio generation complete.")
        return generated_audios
