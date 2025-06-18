# VoiceHub

## Usage Example

```python
from voicehub.automodel import AutoInferenceModel

# Create model using the static from_pretrained method
model = AutoInferenceModel.from_pretrained(
    model_type="orpheustts",
    model_path="canopylabs/orpheus-3b-0.1-ft",
    device="cuda"
)

# Generate speech with the model
output = model(["Hello, how are you today?"], voice="tara", output_prefix="test_output")

print("Speech generation completed. Audio saved as test_output_0.wav")
```