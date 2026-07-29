"""Fine-tuning adapter for VoiceHub's native HuBERT CTC runtime."""

from voicehub.models.asr_wav2vec2.training_asr_wav2vec2 import NativeWav2Vec2TrainingAdapter


class NativeHubertTrainingAdapter(NativeWav2Vec2TrainingAdapter):
    """Train and export HuBERT without an upstream model runtime."""

    native_export_semantics = ("voicehub-native-hubert-ctc-safetensors-and-processor")
    runtime_name = "HuBERT"
    checkpoint_format = "native-hubert-ctc-v1"
    native_architecture_family = "hubert"


__all__ = ["NativeHubertTrainingAdapter"]
