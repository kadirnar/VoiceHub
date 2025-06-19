from voicehub.models.dia.model import Dia


class DiaTTS:
    """
    DiaTTS class for text-to-speech generation using the Dia model.

    This class provides a simple interface for loading and using the Dia model
    to generate speech from text prompts.

    Example:
        ```python
        # Initialize the DiaTTS model
        tts = DiaTTS(model_path="nari-labs/Dia-1.6B", device="cuda")

        # Generate speech from text
        audio = tts(prompt="Hello, how are you?", output_file="output.wav")

        ```
    """

    def __init__(
        self,
        model_path: str = "nari-labs/Dia-1.6B",
        device: str = "cuda",
        compute_dtype: str = "bfloat16",
        use_torch_compile=False,
    ):
        """
        Initialize the DiaTTS model.

        Args:
            model_path (str): Path or name of the pretrained model to load.
                Default is "nari-labs/Dia-1.6B".
            device (str): Device to run the model on (e.g., "cuda", "cpu").
                Default is "cuda".
            compute_dtype (str): Data type for computation (e.g., "bfloat16", "float32").
                Default is "bfloat16".
            use_torch_compile (bool): Whether to use torch.compile for potential speedup.
                Default is False.
        """
        # Store configuration parameters
        self.device = device
        self.compute_dtype = compute_dtype
        self.use_torch_compile = use_torch_compile
        # Load the model with the specified parameters
        self._load_models(model_path)

    def _load_models(self, model_path: str):
        """
        Load the Dia model from the specified path.

        Args:
            model_path (str): Path or name of the pretrained model to load.
        """
        # Initialize the model using the from_pretrained method with the specified compute dtype
        self.model = Dia.from_pretrained(model_path, compute_dtype=self.compute_dtype)

    def __call__(self, prompt: str = "Hello, how are you?", output_file: str = "output.wav"):
        """
        Generate speech from text and save it to a file.

        Args:
            prompt (str): Text to convert to speech.
                Default is "Hello, how are you?".
            output_file (str): Path to save the generated audio.
                Default is "output.wav".

        Returns:
            Audio: The generated audio data.
        """
        # Generate audio from the text prompt
        audio = self.model.generate(prompt, use_torch_compile=self.use_torch_compile, verbose=True)
        # Save the generated audio to the specified output file
        self.model.save_audio(output_file, audio)

        return audio
