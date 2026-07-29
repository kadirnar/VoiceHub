"""Fine-tuning adapter for VoiceHub's native WavLM CTC runtime."""

from voicehub.models.asr_wav2vec2.training_asr_wav2vec2 import NativeWav2Vec2TrainingAdapter


class NativeWavLMTrainingAdapter(NativeWav2Vec2TrainingAdapter):
    """Train and export WavLM without an upstream architecture runtime."""

    native_export_semantics = ("voicehub-native-wavlm-ctc-safetensors-and-processor")
    runtime_name = "WavLM"
    checkpoint_format = "native-wavlm-ctc-v1"
    native_architecture_family = "wavlm"


__all__ = ["NativeWavLMTrainingAdapter"]
