from pathlib import Path

import torch

from voicehub.hub_transport import download_hugging_face_snapshot
from voicehub.models.chatterbox.checkpoint import (
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
    export_module_safetensors,
    load_module_safetensors,
)
from voicehub.models.chatterbox.models.s3gen import S3GEN_SR, S3Gen
from voicehub.models.chatterbox.models.s3tokenizer import S3_SR
from voicehub.models.chatterbox.native_audio import load_waveform
from voicehub.models.chatterbox.watermark import NativePerthWatermarker

REPO_ID = CHECKPOINT_REPOSITORY


class ChatterboxVC:
    """Voice conversion model that re-synthesises speech with a target
    speaker's voice.

    Tokenises the source audio with S3Tokenizer, then decodes the tokens
    through S3Gen conditioned on a reference speaker embedding.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict | None = None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = NativePerthWatermarker(device=device)
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {k: v.to(device) if torch.is_tensor(v) else v for k, v in ref_dict.items()}

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxVC':
        """Load the S3Gen model from a local checkpoint directory."""
        ckpt_dir = Path(ckpt_dir).expanduser().resolve()
        checkpoint_path = ckpt_dir / "s3gen.safetensors"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Chatterbox checkpoint was not found: {checkpoint_path}")

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(
                builtin_voice,
                map_location=map_location,
                weights_only=True,
            )
            ref_dict = states['gen']

        s3gen = S3Gen()
        load_module_safetensors(s3gen, checkpoint_path)
        s3gen.to(device).eval()

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(
        cls,
        device,
        repo_id: str = REPO_ID,
        revision: str = CHECKPOINT_REVISION,
    ) -> 'ChatterboxVC':
        """Download weights from HuggingFace Hub and initialise the model."""
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
            device = "cpu"

        snapshot = download_hugging_face_snapshot(
            repo_id,
            revision=revision,
            allow_patterns=("s3gen.safetensors", "conds.pt"),
        )
        return cls.from_local(snapshot, device)

    def set_target_voice(self, wav_fpath):
        """Extract a reference speaker embedding from a target voice audio
        file."""
        # Load reference wav
        s3gen_ref_wav = load_waveform(
            wav_fpath,
            target_sample_rate=S3GEN_SR,
            device=self.device,
        )

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        """Convert source audio to the target speaker's voice and return a
        watermarked waveform."""
        if target_voice_path:
            self.set_target_voice(target_voice_path)
        elif self.ref_dict is None:
            raise ValueError("Call set_target_voice() or provide target_voice_path.")

        with torch.inference_mode():
            audio_16 = load_waveform(
                audio,
                target_sample_rate=S3_SR,
                device=self.device,
            ).unsqueeze(0)

            s3_tokens, _ = self.s3gen.tokenizer(audio_16)
            wav, _ = self.s3gen.inference(
                speech_tokens=s3_tokens,
                ref_dict=self.ref_dict,
            )
            wav = wav.squeeze(0).detach()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return watermarked_wav.unsqueeze(0)

    def save_pretrained(self, directory: str | Path) -> Path:
        """Export the voice-conversion runtime and optional built-in voice."""
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        export_module_safetensors(
            self.s3gen,
            destination / "s3gen.safetensors",
            component="s3gen",
        )
        if self.ref_dict is not None:
            portable = {
                name: (value.detach().cpu() if torch.is_tensor(value) else value)
                for name, value in self.ref_dict.items()
            }
            torch.save({"gen": portable}, destination / "conds.pt")
        else:
            (destination / "conds.pt").unlink(missing_ok=True)
        return destination
