MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "orpheustts": "OrpheusTTS",
}


class AutoInferenceModel:
    def from_pretrained(model_type: str = "orpheustts", model_path: str = "canopylabs/orpheus-3b-0.1-ft", device: str = "cuda", **kwargs):

        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        InferenceModel = getattr(__import__(f"voicehub.models.{model_type}.orpheus", fromlist=[model_class_name]), model_class_name)

        return InferenceModel(
            model_path=model_path,
            device=device,
            **kwargs
        )
