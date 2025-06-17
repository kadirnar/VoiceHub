from voicehub.automodel import AutoInferenceModel

model = AutoInferenceModel.from_pretrained(model_path="canopylabs/orpheus-3b-0.1-ft", device="cuda")
output = model("Hello, how are you?", chosen_voice="tara")
print(output)