MODEL_TYPE_TO_MODEL_CLASS_NAME = {"orpheustts": "OrpheusTTS", "dia": "DiaTTS", "vui": "VuiTTS"}


class AutoInferenceModel:

    def from_pretrained(
            model_type: str = "orpheustts",
            model_path: str = "canopylabs/orpheus-3b-0.1-ft",
            device: str = "cuda"):
        """
        Dynamically load and instantiate the appropriate model class based on model_type.

        Args:
            model_type: Type of model to load (e.g., "orpheustts")
            model_path: Path or name of the pre-trained model
            device: Device to load the model on ("cuda" or "cpu")
            **kwargs: Additional arguments to pass to the model constructor

        Returns:
            An instance of the requested model class
        """
        # Get the model class name from the mapping
        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]

        # Import the module dynamically
        module = __import__(f"voicehub.models.{model_type}.inference", fromlist=[model_class_name])

        # Get the model class from the module
        InferenceModel = getattr(module, model_class_name)

        # Instantiate and return the model
        return InferenceModel(
            model_path=model_path,
            device=device,
        )
