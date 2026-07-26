import os
from loguru import logger

def create_speaker(
        audio_processor,
        audio_path: str,
        whisper_model: str = "turbo",
        whisper_device = None,
    ):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    return audio_processor.create_speaker_from_whisper(
        audio=audio_path,
        whisper_model=whisper_model,
        device=whisper_device
    )