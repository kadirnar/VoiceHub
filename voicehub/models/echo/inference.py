from functools import partial

import torch

from voicehub.base_model import BaseTTSModel
from voicehub.models.echo.sampling import (
    ae_decode,
    crop_audio_to_flattening_point,
    load_audio,
    load_fish_ae_from_hf,
    load_model_from_hf,
    load_pca_state_from_hf,
    sample_euler_cfg_independent_guidances,
    sample_pipeline,
)


class EchoTTS(BaseTTSModel):
    """VoiceHub wrapper for Echo-TTS, a multi-speaker text-to-speech model
    with speaker reference conditioning via flow matching."""

    def __init__(
        self,
        model_path: str = "jordand/echo-tts-base",
        device: str = "cuda",
        compile: bool = False,
    ):
        """Initialize the EchoTTS model.

        Args:
            model_path: HuggingFace repo id for the Echo-TTS checkpoint.
            device: Torch device string.
            compile: Whether to torch.compile the model for faster inference.
        """
        super().__init__(model_path, device)
        self.model = load_model_from_hf(
            repo_id=model_path,
            device=device,
            compile=compile,
            delete_blockwise_modules=True,
        )
        self.fish_ae = load_fish_ae_from_hf(device=device)
        self.pca_state = load_pca_state_from_hf(repo_id=model_path, device=device)

    @property
    def sample_rate(self) -> int:
        return 44100

    def __call__(
        self,
        text: str,
        output_file: str = "output.wav",
        speaker_audio_path: str | None = None,
        num_steps: int = 40,
        cfg_scale_text: float = 3.0,
        cfg_scale_speaker: float = 8.0,
        cfg_min_t: float = 0.5,
        cfg_max_t: float = 1.0,
        sequence_length: int = 640,
        rng_seed: int = 0,
        truncation_factor: float | None = None,
    ):
        """Generate speech from text and save to a file.

        Args:
            text: Text to synthesize. Will be prefixed with ``[S1]`` if needed.
            output_file: Path to save the generated audio (44.1 kHz WAV).
            speaker_audio_path: Optional path to a speaker reference audio file.
            num_steps: Number of Euler sampling steps.
            cfg_scale_text: Classifier-free guidance scale for text.
            cfg_scale_speaker: Classifier-free guidance scale for speaker.
            cfg_min_t: Minimum timestep for applying CFG.
            cfg_max_t: Maximum timestep for applying CFG.
            sequence_length: Max latent sequence length (640 = ~30 s).
            rng_seed: Random seed for reproducibility.
            truncation_factor: Optional noise truncation factor.
        """
        speaker_audio = None
        if speaker_audio_path is not None:
            speaker_audio = load_audio(speaker_audio_path).to(self.device)

        sample_fn = partial(
            sample_euler_cfg_independent_guidances,
            num_steps=num_steps,
            cfg_scale_text=cfg_scale_text,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_min_t=cfg_min_t,
            cfg_max_t=cfg_max_t,
            truncation_factor=truncation_factor,
            rescale_k=None,
            rescale_sigma=None,
            speaker_kv_scale=None,
            speaker_kv_max_layers=None,
            speaker_kv_min_t=None,
            sequence_length=sequence_length,
        )

        audio_out, _ = sample_pipeline(
            model=self.model,
            fish_ae=self.fish_ae,
            pca_state=self.pca_state,
            sample_fn=sample_fn,
            text_prompt=text,
            speaker_audio=speaker_audio,
            rng_seed=rng_seed,
        )

        self.save_audio(output_file, audio_out[0], self.sample_rate)
        return audio_out
