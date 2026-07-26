import voicehub.third_party.dac as dac
import torch
import tqdm 
import torch.nn as nn
import torchaudio
import math
import pyloudnorm as pyln
import numpy as np
from huggingface_hub import hf_hub_download
import os

from ..utils import helpers

def process_audio_tensor(
    audio: torch.Tensor,
    sample_rate: int = 24000,
    target_loudness: float = -18.0,
    peak_limit: float = -1,
    block_size: float = 0.400,
) -> torch.Tensor:
    device = audio.device
    dtype = audio.dtype
    audio_np = audio.detach().cpu().numpy()

    if audio_np.ndim > 1:
        if audio_np.shape[1] > 1:
            audio_np = np.mean(audio_np, axis=1)
        else:
            audio_np = np.squeeze(audio_np)

    original_length = len(audio_np)
    min_samples = int(block_size * sample_rate)
    
    if original_length < min_samples:
        pad_length = min_samples - original_length
        audio_padded = np.pad(audio_np, (0, pad_length), mode='constant')
    else:
        audio_padded = audio_np

    meter = pyln.Meter(sample_rate, block_size=block_size)
    measured_loudness = meter.integrated_loudness(audio_padded)
    normalized = pyln.normalize.loudness(audio_padded, measured_loudness, target_loudness)

    peak_value = np.max(np.abs(normalized))
    threshold_value = 10 ** (peak_limit / 20)
    if peak_value > threshold_value:
        normalized = pyln.normalize.peak(normalized, peak_limit)

    if original_length < min_samples:
        normalized = normalized[:original_length]

    normalized_tensor = torch.from_numpy(normalized).to(dtype=dtype).to(device).unsqueeze(0).unsqueeze(0)
    return normalized_tensor

class DacInterface:
    def __init__(self, device: str = None, model_path: str = None):
        
        if model_path is None:
            model_path = self._get_model()
            
        self.device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = dac.DAC.load(model_path).to(self.device).eval()
        self.sr = 24000

    def _get_model(self):
        cache = helpers.get_cache_dir()
        model_path = hf_hub_download(
            repo_id="ibm-research/DAC.speech.v1.0",
            filename="weights_24khz_1.5kbps_v1.0.pth",
            local_dir=os.path.join(cache, "dac"),
            local_files_only=False
        )
        return model_path

    def convert_audio(self, wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
        assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
        assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
        *shape, channels, length = wav.shape
        if target_channels == 1:
            wav = wav.mean(-2, keepdim=True)
        elif target_channels == 2:
            wav = wav.expand(*shape, target_channels, length)
        elif channels == 1:
            wav = wav.expand(target_channels, -1)
        else:
            raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        return wav
    
    def convert_audio_tensor(self, audio: torch.Tensor, sr):
        return self.convert_audio(audio, sr, self.sr, 1)
    
    def load_audio(self, path):
        wav, sr = torchaudio.load(path)
        return self.convert_audio_tensor(wav, sr).unsqueeze(0)
    
    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.model.hop_length) * self.model.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor, win_duration: int = 5.0, verbose: bool = False):
        x = process_audio_tensor(x)
        device = x.device
        nb, nac, nt = x.shape
        x = x.reshape(nb * nac, 1, nt)
        n_samples = int(win_duration * self.sr)
        n_samples = int(math.ceil(n_samples / self.model.hop_length) * self.model.hop_length)
        hop = n_samples 
        codes = []
        range_fn = range if not verbose else tqdm.trange
        for i in range_fn(0, nt, hop):
            chunk = x[..., i:i + n_samples]
            audio_data = chunk.to(self.model.device)
            audio_data = self.preprocess(audio_data)
            _, c, _, _, _ = self.model.encode(audio_data, None)
            codes.append(c.to(device))
        codes = torch.cat(codes, dim=-1)
        return codes
    
    def apply_fade(self, audio):
        fade_in_sec  = 0.015
        fade_out_sec = 0.015

        total_len = audio.shape[-1]

        fade_in_len  = int(self.sr * fade_in_sec)
        fade_out_len = int(self.sr * fade_out_sec)

        max_fade = total_len // 2
        fade_in_len  = min(fade_in_len, max_fade)
        fade_out_len = min(fade_out_len, max_fade)

        fade_in  = torch.linspace(0., 1., fade_in_len,  device=audio.device)
        fade_out = torch.linspace(1., 0., fade_out_len, device=audio.device)

        audio[..., :fade_in_len]                         *= fade_in
        audio[..., total_len - fade_out_len : total_len] *= fade_out

        return audio

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        verbose: bool = False,
        chunk_length: int = 2048
    ) -> torch.Tensor:
        model = self.model
        range_fn = range if not verbose else tqdm.trange
        original_device = codes.device
        recons = []
        for i in range_fn(0, codes.shape[-1], chunk_length):
            c = codes[..., i : i + chunk_length].to(model.device)
            z = model.quantizer.from_codes(c)[0]
            r = model.decode(z)
            recons.append(self.apply_fade(r.to(original_device)))
        recons = torch.cat(recons, dim=-1)
        return process_audio_tensor(recons)
