# Copyright (c) 2023 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by VoiceHub in 2026: vendored and namespaced for inference-only use.

from pathlib import Path

import torch

from .context_graph import ContextGraph
from .ctc_utils import force_align, gen_ctc_peak_time, gen_timestamps_from_peak
from .file_utils import read_symbol_table
from .hub import Hub
from .search import DecodeResult, attention_rescoring, ctc_prefix_beam_search
from voicehub.processing.kaldi import KaldiFbankConfig, kaldi_fbank
from voicehub.processing.waveform import load_native_audio


class Model:

    def __init__(
        self,
        model_dir: str | Path,
        *,
        device: str = "cpu",
        beam: int = 5,
        context_path: str | None = None,
        context_score: float = 6.0,
        resample_rate: int = 16000,
    ):
        model_dir = Path(model_dir).expanduser()
        model_path = model_dir / "final.zip"
        units_path = model_dir / "units.txt"
        missing = [path.name for path in (model_path, units_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"WeNet runtime directory {model_dir} is missing required "
                f"file(s): {', '.join(missing)}.")

        self.device = torch.device(device)
        self.model = torch.jit.load(
            str(model_path),
            map_location=self.device,
        )
        self.resample_rate = resample_rate
        self.model.eval()
        self.model.to(self.device)
        self.symbol_table = read_symbol_table(str(units_path))
        self.char_dict = {v: k for k, v in self.symbol_table.items()}
        self.beam = beam
        if context_path is not None:
            self.context_graph = ContextGraph(
                context_path,
                self.symbol_table,
                context_score=context_score,
            )
        else:
            self.context_graph = None

    def compute_feats(self, audio_file: str) -> torch.Tensor:
        audio = load_native_audio(
            audio_file,
            target_sampling_rate=self.resample_rate,
        )
        waveform = audio.waveform.to(self.device)
        # The published WeNet recipes multiply normalized PCM by 2**15
        # before Kaldi feature extraction.
        feats = kaldi_fbank(
            waveform * float(1 << 15),
            KaldiFbankConfig(
                sample_frequency=float(self.resample_rate),
                num_mel_bins=80,
                frame_length=25.0,
                frame_shift=10.0,
                dither=0.0,
                energy_floor=0.0,
            ),
        )
        feats = feats.unsqueeze(0)
        return feats

    @torch.no_grad()
    def _decode(
        self,
        audio_file: str,
        tokens_info: bool = False,
        label: str | None = None,
    ) -> dict:
        feats = self.compute_feats(audio_file)
        encoder_out, _, _ = self.model.forward_encoder_chunk(feats, 0, -1)
        encoder_lens = torch.tensor([encoder_out.size(1)], dtype=torch.long, device=encoder_out.device)
        ctc_probs = self.model.ctc_activation(encoder_out)
        if label is None:
            ctc_prefix_results = ctc_prefix_beam_search(
                ctc_probs,
                encoder_lens,
                self.beam,
                context_graph=self.context_graph,
            )
        else:  # force align mode, construct ctc prefix result from alignment
            label_t = self.tokenize(label)
            alignment = force_align(
                ctc_probs.squeeze(0),
                torch.tensor(label_t, dtype=torch.long),
            )
            peaks = gen_ctc_peak_time(alignment)
            ctc_prefix_results = [
                DecodeResult(
                    tokens=label_t,
                    score=0.0,
                    times=peaks,
                    nbest=[label_t],
                    nbest_scores=[0.0],
                    nbest_times=[peaks],
                )
            ]
        rescoring_results = attention_rescoring(
            self.model,
            ctc_prefix_results,
            encoder_out,
            encoder_lens,
            0.3,
            0.5,
        )
        res = rescoring_results[0]
        result = {
            "text": "".join(self.char_dict[token] for token in res.tokens),
            "confidence": res.confidence,
        }

        if tokens_info:
            frame_rate = self.model.subsampling_rate() * 0.01
            max_duration = encoder_out.size(1) * frame_rate
            times = gen_timestamps_from_peak(
                res.times,
                max_duration,
                frame_rate,
                1.0,
            )
            result["tokens"] = [{
                "token": self.char_dict[token],
                "start": times[index][0],
                "end": times[index][1],
                "confidence": res.tokens_confidence[index],
            } for index, token in enumerate(res.tokens)]
        return result

    def transcribe(self, audio_file: str, tokens_info: bool = False) -> dict:
        return self._decode(audio_file, tokens_info)

    def tokenize(self, label: str):
        # TODO(Binbin Zhang): Support BPE
        tokens = []
        for c in label:
            if c == " ":
                c = "▁"
            tokens.append(c)
        token_list = []
        for c in tokens:
            if c in self.symbol_table:
                token_list.append(self.symbol_table[c])
            elif "<unk>" in self.symbol_table:
                token_list.append(self.symbol_table["<unk>"])
        return token_list

    def align(self, audio_file: str, label: str) -> dict:
        return self._decode(audio_file, True, label)


def load_model(
    model_name_or_path: str | Path | None = None,
    *,
    language: str | None = None,
    model_dir: str | Path | None = None,
    gpu: int = -1,
    beam: int = 5,
    context_path: str | None = None,
    context_score: float = 6.0,
    resample_rate: int = 16000,
    device: str = "cpu",
) -> Model:
    """Load a WeNet TorchScript runtime directory or a known Hub asset."""
    sources = [source for source in (model_name_or_path, language, model_dir) if source is not None]
    if len(sources) != 1:
        raise ValueError("Pass exactly one of `model_name_or_path`, `language`, or "
                         "`model_dir`.")
    source = sources[0]
    candidate = Path(source).expanduser()
    if candidate.exists():
        resolved_model_dir = candidate
    elif str(source) in Hub.Assets:
        resolved_model_dir = Path(Hub.get_model_by_lang(str(source)))
    else:
        available = ", ".join(sorted(Hub.Assets))
        raise FileNotFoundError(
            f"WeNet model path was not found: {candidate}. Known downloadable "
            f"runtime names: {available}.")

    if gpu >= 0:
        device = f"cuda:{gpu}"
    return Model(
        resolved_model_dir,
        device=device,
        beam=beam,
        context_path=context_path,
        context_score=context_score,
        resample_rate=resample_rate,
    )
