import torch
from loguru import logger

from ...whisper import transcribe
from .alignment import CTCForcedAlignment

def create_speaker(
        device,
        audio_codec,
        audio_path: str,
        transcript: str = None,
        whisper_model: str = "turbo",
        whisper_device = None
    ):

    if transcript is None:
        logger.info("Transcription not provided, transcribing audio with whisper.")
        transcript = transcribe.transcribe_once(
            audio_path=audio_path,
            model=whisper_model,
            device=whisper_device
        )

    if not transcript:
        raise ValueError("Transcript text is empty")

    ctc = CTCForcedAlignment(device)
    words = ctc.align(audio_path, transcript)
    ctc.free()

    full_codes = audio_codec.encode(
        audio_codec.convert_audio_tensor(
            audio=torch.cat([i["audio"] for i in words], dim=1),
            sr=ctc.sample_rate
        ).to(audio_codec.device)
    ).tolist()

    data = []
    start = 0
    for i in words:
        end = int(round((i["x1"] / ctc.sample_rate) * 75))
        word_tokens = full_codes[0][0][start:end]
        start = end
        if not word_tokens:
            word_tokens = [1]

        data.append({
            "word": i["word"],
            "duration": round(len(word_tokens) / 75, 2),
            "codes": word_tokens
        })

    return {
        "text": transcript,
        "words": data,
    }
