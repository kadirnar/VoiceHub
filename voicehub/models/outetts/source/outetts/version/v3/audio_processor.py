import io
import torch
import torch.nn.functional as F
from loguru import logger

from ...dac.interface import DacInterface
from ...whisper.transcribe import transcribe_once_word_level
from ...utils.preprocessing import text_normalizations

def calculate_pitch(
    audio_tensor: torch.Tensor,
    sr: int,
    min_freq: float = 75.0,
    max_freq: float = 600.0,
    frame_length: int = 400,
    hop_length: int = 160,
    threshold: float = 0.3
) -> torch.Tensor:
    """
    Calculate pitch frequencies for short audio clips using autocorrelation.
    
    Args:
        audio_tensor: Input audio tensor (1D or 2D [channels, samples])
        sr: Sampling rate
        min_freq: Minimum detectable frequency (Hz)
        max_freq: Maximum detectable frequency (Hz)
        frame_length: Analysis frame length in samples
        hop_length: Hop size in samples
        threshold: Voicing threshold (0.0-1.0)
    
    Returns:
        Tensor of pitch values (Hz) per frame
    """
    
    # Convert to mono and ensure 1D
    if audio_tensor.dim() > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0)
    audio_tensor = audio_tensor.squeeze()
    
    # Pad audio to ensure full frame coverage
    num_samples = audio_tensor.shape[-1]
    pad_len = (frame_length - (num_samples % hop_length)) % hop_length
    audio_tensor = F.pad(audio_tensor, (0, pad_len))
    
    # Create frames using unfold
    frames = audio_tensor.unfold(0, frame_length, hop_length)
    num_frames = frames.shape[0]
    
    # Precompute window and move to device
    window = torch.hann_window(frame_length, device=audio_tensor.device)
    frames_windowed = frames * window
    
    # Compute autocorrelation using FFT
    f = torch.fft.rfft(frames_windowed, n=2*frame_length, dim=1)
    power_spectrum = f.real.pow(2) + f.imag.pow(2)
    autocorr = torch.fft.irfft(power_spectrum, dim=1)[:, :frame_length]
    
    # Find valid frequency range indices
    min_idx = max(1, int(sr / max_freq))
    max_idx = min(frame_length, int(sr / min_freq))
    
    # Find peak indices in valid range
    relevant_autocorr = autocorr[:, min_idx:max_idx]
    peak_values, peak_indices = torch.max(relevant_autocorr, dim=1)
    peak_indices += min_idx  # Adjust to original indices
    
    # Parabolic interpolation for sub-sample accuracy
    indices = torch.clamp(peak_indices, 1, frame_length-2)
    alpha = autocorr[torch.arange(num_frames), indices-1]
    beta = autocorr[torch.arange(num_frames), indices]
    gamma = autocorr[torch.arange(num_frames), indices+1]
    
    delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma + 1e-8)
    delta = torch.where((peak_indices > 0) & (peak_indices < frame_length-1), delta, 0.0)
    
    # Calculate final periods and pitches
    best_period = (peak_indices + delta) / sr
    pitch = torch.where(best_period > 0, 1.0 / best_period, 0.0)
    
    # Apply voicing threshold
    autocorr_0 = autocorr[:, 0]
    voiced = (peak_values / (autocorr_0 + 1e-8)) > threshold
    pitch = torch.where(voiced, pitch, 0.0)
    
    # Clamp valid frequencies
    pitch = torch.clamp(pitch, min_freq, max_freq)
    
    return pitch

def extract_single_pitch_value(
    audio_tensor: torch.Tensor,
    sr: int,
    min_freq: float = 75.0,
    max_freq: float = 600.0,
    frame_length: int = 400,
    hop_length: int = 160,
    threshold: float = 0.3
) -> float:
    """
    Calculates the average pitch of an audio tensor and normalizes it to 0-1 range.

    Args:
        audio_tensor: Input audio tensor (1D or 2D [channels, samples])
        sr: Sampling rate
        min_freq: Minimum detectable frequency (Hz)
        max_freq: Maximum detectable frequency (Hz)
        frame_length: Analysis frame length in samples
        hop_length: Hop size in samples
        threshold: Voicing threshold (0.0-1.0)

    Returns:
        A single float value representing the normalized average pitch (0.0-1.0).
    """
    pitch_tensor = calculate_pitch(
        audio_tensor, sr, min_freq, max_freq, frame_length, hop_length, threshold
    )

    # Calculate the average pitch across frames
    average_pitch = torch.mean(pitch_tensor)

    # Normalize to 0-1 range
    normalized_pitch = (average_pitch - min_freq) / (max_freq - min_freq)

    # Clamp to ensure it's strictly within 0-1 
    normalized_pitch = torch.clamp(normalized_pitch, 0.0, 1.0)

    return normalized_pitch.item() 

class Features:
    def __init__(self, device):
        self.eps = 1e-10
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")

    def scale_values(self, value: float) -> int:
        """
        Scale a value from [0,1] to [0,100] and round to nearest integer
        """
        return round(value * 100)

    def features_to_tokens(self, features: dict) -> list:
        """
        Convert features to token strings in format <|feature_value|>
        """
        return [f"<|{name}_{value}|>" for name, value in features.items()]
    
    def validate_audio(self, audio: torch.Tensor) -> bool:
        """
        Validate audio tensor before processing
        """
        if audio is None or not isinstance(audio, torch.Tensor):
            return False
        if audio.numel() == 0:  # Check if tensor is empty
            return False
        if torch.isnan(audio).any() or torch.isinf(audio).any():
            return False
        return True
    
    def get_default_features(self) -> dict:
        """
        Return default feature values when audio is invalid
        """
        return {
            'energy': 0,
            'spectral_centroid': 0,
            'pitch': 0
        }
    
    def extract_audio_features(self, audio: torch.Tensor, sr: int) -> dict:
        """
        Extract fast-to-compute features from audio segments.
        Each feature is normalized to [0, 1] range.
        
        Args:
            audio: Audio tensor of shape [channels, samples]
            sr: Sample rate
        
        Returns:
            Dictionary of features, each as a single float value
        """

        if not self.validate_audio(audio):
            return self.get_default_features()

        # Convert to mono if stereo
        if audio.dim() == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        # Get absolute values for amplitude calculations
        features = {}
        
        # RMS Energy (loudness) - normalized to [0, 1]
        features['energy'] = float(torch.sqrt(torch.mean(audio ** 2)))
        
        # Spectral Centroid - normalized to [0, 1]
        spec = torch.abs(torch.fft.rfft(audio))
        freqs = torch.linspace(0, sr/2, spec.shape[-1], device=self.device)
        spec_sum = torch.sum(spec) + self.eps
        centroid = torch.sum(freqs * spec.squeeze()) / spec_sum
        features['spectral_centroid'] = float(centroid / (sr/2)) 

        # Pitch - normalized to [0, 1]
        features['pitch'] = extract_single_pitch_value(audio, sr)

        for name, value in features.items():
            features[name] = self.scale_values(value)

        return features

class AudioProcessor:
    def __init__(self, config):
        self.device = torch.device(config.device if config.device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        self.features = Features(config.device)
        self.audio_codec = DacInterface(config.device, config.audio_codec_path)

    def create_speaker_from_whisper(self, audio, whisper_model: str = "turbo", device = None):
        seconds = self.audio_codec.load_audio(audio).flatten().size(0) / self.audio_codec.sr
        if seconds > 20:
            raise ValueError("Speaker audio is longer than 20 seconds. Use a shorter clip for best results.")
        if seconds > 15:
            logger.warning("Speaker audio is longer than 15 seconds. For best results, consider using an audio clip up to 15 seconds.")
        
        data = transcribe_once_word_level(audio, whisper_model, device)
        text = text_normalizations(data['text'])
        words = []
        for s in data['segments']:
            words.extend([{'word': i['word'].strip(), 'start': float(i['start']), 'end': float(i['end'])} for i in s['words']])

        with open(audio, "rb") as f:
            audio_bytes = f.read()

        return self.create_speaker_from_dict({"audio": {"bytes": audio_bytes}, "text": text, "words": words})
    
    def create_speaker_from_dict(self, data: dict):
        audio = io.BytesIO(data['audio']['bytes'])
        audio = self.audio_codec.load_audio(audio).to(self.device)
        full_codes = self.audio_codec.encode(audio, verbose=True).tolist()[0]

        c1 = full_codes[0]
        c2 = full_codes[1]

        sr = self.audio_codec.sr
        text = data['text']
        words = data['words']

        tps = 75

        audio = audio.squeeze(0)
        global_features = self.features.extract_audio_features(audio, sr)

        start = None
        word_codes = []
        max_extension = 20

        for idx, i in enumerate(words):
            if start is None:
                start = max(0, int(i['start'] * tps) - max_extension)
            word = i['word'].strip()
            if idx == len(words) - 1:
                end = min(len(c1), int(i['end'] * tps) + max_extension)
            else:
                end = int(i['end'] * tps)

            word_c1 = c1[start:end]
            word_c2 = c2[start:end]

            word_audio = audio[:, int(i['start']*sr):int(i['end']*sr)]
            features = self.features.extract_audio_features(word_audio, sr)

            start = end

            word_codes.append({
                "word": word,
                "duration": round(len(word_c1) / tps, 2),
                "c1": word_c1,
                "c2": word_c2,
                "features": features
            })

        return {
            "text": text,
            "words": word_codes,
            "global_features": global_features
        }
