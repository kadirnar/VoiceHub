from __future__ import annotations

import argparse
import contextlib
import io
import logging
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))

import voicehub.models.inflecttts.source.inflect.runtime.commons as commons  # noqa: E402
import voicehub.models.inflecttts.source.inflect.runtime.utils as utils  # noqa: E402
from voicehub.models.inflecttts.source.inflect.inflect_vits_frontend import run_vits_frontend  # noqa: E402
from voicehub.models.inflecttts.source.inflect.runtime.models import SynthesizerTrn  # noqa: E402
from voicehub.models.inflecttts.source.inflect.runtime.text import cleaned_text_to_sequence  # noqa: E402
from voicehub.models.inflecttts.source.inflect.runtime.text.symbols import symbols  # noqa: E402


def split_text(text: str, limit: int = 280) -> list[str]:
    normalized = " ".join(text.split())
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?;:])\s+", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    for sentence in sentences or [normalized]:
        while len(sentence) > limit:
            search = sentence[: limit + 1]
            punctuation = max(search.rfind(mark) for mark in (",", ";", ":"))
            split_at = (
                punctuation + 1
                if punctuation >= limit // 2
                else sentence.rfind(" ", 0, limit + 1)
            )
            if split_at < limit // 2:
                split_at = limit
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


def boundary_pause_seconds(chunk: str) -> float:
    ending = chunk.rstrip()[-1:] if chunk.strip() else ""
    return {
        "?": 0.28,
        "!": 0.24,
        ".": 0.22,
        ";": 0.16,
        ":": 0.13,
        ",": 0.09,
    }.get(ending, 0.08)


def edge_fade(waveform: np.ndarray, sample_rate: int, milliseconds: float = 5.0) -> np.ndarray:
    frames = min(round(sample_rate * milliseconds / 1000.0), waveform.size // 2)
    if frames <= 0:
        return waveform
    output = waveform.copy()
    ramp = np.linspace(0.0, 1.0, frames, endpoint=True, dtype=np.float32)
    output[:frames] *= ramp
    output[-frames:] *= ramp[::-1]
    return output


def optimize_for_inference(model: SynthesizerTrn) -> None:
    """Collapse training-time weight normalization without changing outputs."""
    with contextlib.redirect_stdout(io.StringIO()):
        model.dec.remove_weight_norm()
    for flow in model.flow.flows:
        encoder = getattr(flow, "enc", None)
        if encoder is not None and hasattr(encoder, "remove_weight_norm"):
            encoder.remove_weight_norm()


class InflectTTS:
    def __init__(self, model_dir: str | Path = PACKAGE_ROOT, device: str = "cpu") -> None:
        self.root = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.hps = utils.get_hparams_from_file(str(self.root / "config.json"))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`torch.nn.utils.weight_norm` is deprecated",
                category=FutureWarning,
            )
            self.model = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                **self.hps.model,
            ).to(self.device).eval()
        root_logger = logging.getLogger()
        previous_level = root_logger.level
        try:
            root_logger.setLevel(logging.WARNING)
            utils.load_checkpoint(str(self.root / "model.pth"), self.model, None)
        finally:
            root_logger.setLevel(previous_level)
        self.checkpoint_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        optimize_for_inference(self.model)
        self.deployed_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.sample_rate = int(self.hps.data.sampling_rate)

    def _tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        phonemes = run_vits_frontend(text).phoneme_text
        sequence = cleaned_text_to_sequence(phonemes)
        if self.hps.data.add_blank:
            sequence = commons.intersperse(sequence, 0)
        if not sequence:
            raise ValueError("The text frontend produced no speakable tokens.")
        tokens = torch.LongTensor(sequence).to(self.device).unsqueeze(0)
        lengths = torch.LongTensor([tokens.size(1)]).to(self.device)
        return tokens, lengths

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        *,
        speed: float = 1.0,
        variation: float = 0.667,
        seed: int = 0,
    ) -> tuple[int, np.ndarray]:
        normalized = " ".join(text.split())
        if not normalized:
            raise ValueError("Text must not be empty.")
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        if not 0.0 <= variation <= 1.0:
            raise ValueError("variation must be between 0.0 and 1.0")
        chunks = split_text(normalized)
        pieces: list[np.ndarray] = []
        for index, chunk in enumerate(chunks):
            if index:
                pieces.append(
                    np.zeros(
                        round(self.sample_rate * boundary_pause_seconds(chunks[index - 1])),
                        dtype=np.float32,
                    )
                )
            tokens, lengths = self._tokens(chunk)
            torch.manual_seed(seed + index)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed + index)
            waveform = self.model.infer(
                tokens,
                lengths,
                noise_scale=variation,
                noise_scale_w=0.8,
                length_scale=1.0 / speed,
                max_len=4000,
            )[0][0, 0].float().cpu().numpy()
            pieces.append(edge_fade(waveform, self.sample_rate))
        waveform = np.clip(np.concatenate(pieces), -1.0, 1.0)
        return self.sample_rate, waveform

    def save(self, text: str, output: str | Path, **kwargs: object) -> Path:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sample_rate, waveform = self.synthesize(text, **kwargs)
        sf.write(destination, waveform, sample_rate)
        return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone Inflect v2 synthesis.")
    parser.add_argument("--model-dir", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--variation", type=float, default=0.667)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    engine = InflectTTS(args.model_dir, args.device)
    engine.save(
        args.text,
        args.output,
        speed=args.speed,
        variation=args.variation,
        seed=args.seed,
    )
    print(f"wrote {args.output} at {engine.sample_rate} Hz")


if __name__ == "__main__":
    main()
