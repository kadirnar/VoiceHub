MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "orpheustts": "OrpheusTTS",
    "dia": "DiaTTS",
    "vui": "VuiTTS",
"chatterbox": "ChatterboxInference",
    "kokoro": "KokoroTTS",
    "echo": "EchoTTS",
}


class AutoInferenceModel:
    """Factory class that dynamically loads and instantiates TTS model backends.

    Uses a registry mapping (``MODEL_TYPE_TO_MODEL_CLASS_NAME``) to resolve
    short model-type strings to their concrete inference classes, importing the
    appropriate module on demand so that unused backends are never loaded.
    """

    @staticmethod
    def from_pretrained(
            model_type: str = "orpheustts",
            model_path: str = "canopylabs/orpheus-3b-0.1-ft",
            device: str = "cuda",
            **kwargs):
        """Dynamically load and instantiate the appropriate model class.

        Args:
            model_type: Key into ``MODEL_TYPE_TO_MODEL_CLASS_NAME``
                (e.g. ``"orpheustts"``, ``"dia"``, ``"kokoro"``).
            model_path: HuggingFace repo id or local path to the checkpoint.
            device: Target device (``"cuda"`` or ``"cpu"``).
            **kwargs: Additional keyword arguments passed to the model class
                (e.g. ``lang_code`` for Kokoro).

        Returns:
            An instance of the resolved inference model class, ready for use.

        Raises:
            KeyError: If *model_type* is not found in the registry.
        """
        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]

        module = __import__(f"voicehub.models.{model_type}.inference", fromlist=[model_class_name])

        InferenceModel = getattr(module, model_class_name)

        return InferenceModel(
            model_path=model_path,
            device=device,
            **kwargs,
        )
