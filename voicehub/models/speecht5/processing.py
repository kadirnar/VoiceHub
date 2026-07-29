"""VoiceHub-native text and audio processing for SpeechT5.

The implementation intentionally mirrors the public SpeechT5 processor
contract without importing Transformers, NumPy, SentencePiece,
torchaudio, or librosa.  Text tokenization is backed by VoiceHub's
audited SentencePiece unigram reader.  Target features reproduce the
published frontend: 16 kHz audio, a periodic 1024-sample Hann window,
256-sample hop, amplitude spectrogram, Slaney-normalized 80-bin mel
projection, and base-10 logarithm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil, isfinite
from pathlib import Path
from shutil import copyfile
from typing import Any

from voicehub.hub import read_json_file, write_json_file
from voicehub.tokenization import SentencePieceUnigramTokenizer

_TOKENIZER_MODEL_NAME = "spm_char.model"
_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_SPECIAL_TOKENS_NAME = "special_tokens_map.json"
_PREPROCESSOR_CONFIG_NAME = "preprocessor_config.json"


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - package invariant
        raise RuntimeError(
            "Native SpeechT5 processing requires PyTorch, VoiceHub's compute "
            "runtime.") from error
    return torch


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")
    return value


def _finite_real(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number.")
    value = float(value)
    if not isfinite(value) or (minimum is not None and value < minimum):
        qualifier = "finite" if minimum is None else f"finite and at least {minimum}"
        raise ValueError(f"`{name}` must be {qualifier}.")
    return value


def _text_batch(text: str | Sequence[str]) -> tuple[list[str], bool]:
    if isinstance(text, str):
        return [text], False
    if isinstance(text, (bytes, bytearray)) or not isinstance(text, Sequence):
        raise TypeError("SpeechT5 text must be a string or a sequence of strings.")
    rows = list(text)
    if not rows:
        raise ValueError("SpeechT5 text batches cannot be empty.")
    if any(not isinstance(row, str) for row in rows):
        raise TypeError("Every SpeechT5 text item must be a string.")
    return rows, True


def _audio_batch(value: Any) -> tuple[list[Any], bool]:
    torch = _torch()
    if isinstance(value, torch.Tensor):
        if value.ndim == 1:
            return [value], False
        if value.ndim == 2:
            return list(value.unbind(0)), True
        raise ValueError("SpeechT5 audio tensors must have shape [time] or [batch, time].")
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError("SpeechT5 processor audio must be an in-memory waveform.")
    if not isinstance(value, Sequence):
        return [value], False
    if not value:
        raise ValueError("SpeechT5 audio cannot be empty.")
    first = value[0]
    batched = isinstance(first, (Sequence, torch.Tensor)) and not isinstance(
        first,
        (str, bytes, bytearray),
    )
    return (list(value), True) if batched else ([value], False)


@dataclass(frozen=True, slots=True)
class SpeechT5FeatureConfig:
    """Serializable SpeechT5 frontend parameters."""

    feature_size: int = 1
    sampling_rate: int = 16_000
    padding_value: float = 0.0
    do_normalize: bool = False
    num_mel_bins: int = 80
    hop_length: int = 16
    win_length: int = 64
    win_function: str = "hann_window"
    frame_signal_scale: float = 1.0
    fmin: float = 80.0
    fmax: float = 7_600.0
    mel_floor: float = 1e-10
    reduction_factor: int = 2
    return_attention_mask: bool = True

    def __post_init__(self) -> None:
        for name in (
                "feature_size",
                "sampling_rate",
                "num_mel_bins",
                "hop_length",
                "win_length",
                "reduction_factor",
        ):
            _positive_integer(getattr(self, name), name=name)
        for name in (
                "padding_value",
                "frame_signal_scale",
                "fmin",
                "fmax",
                "mel_floor",
        ):
            minimum = 0.0 if name in {"fmin", "mel_floor"} else None
            object.__setattr__(
                self,
                name,
                _finite_real(getattr(self, name), name=name, minimum=minimum),
            )
        if self.fmax <= self.fmin or self.fmax > self.sampling_rate / 2:
            raise ValueError("SpeechT5 mel bounds must satisfy 0 <= fmin < fmax <= Nyquist.")
        if self.mel_floor <= 0.0:
            raise ValueError("`mel_floor` must be greater than zero.")
        if self.win_function != "hann_window":
            raise ValueError("Native SpeechT5 supports the published Hann window only.")
        if self.frame_signal_scale != 1.0:
            raise ValueError("Native SpeechT5 rejects deprecated non-unit frame scaling.")
        if not isinstance(self.do_normalize, bool):
            raise TypeError("`do_normalize` must be a boolean.")
        if not isinstance(self.return_attention_mask, bool):
            raise TypeError("`return_attention_mask` must be a boolean.")

    @property
    def sample_size(self) -> int:
        return self.win_length * self.sampling_rate // 1_000

    @property
    def sample_stride(self) -> int:
        return self.hop_length * self.sampling_rate // 1_000

    @property
    def n_fft(self) -> int:
        return 1 << (self.sample_size - 1).bit_length()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> SpeechT5FeatureConfig:
        if not isinstance(values, Mapping):
            raise TypeError("SpeechT5 feature configuration must be a mapping.")
        allowed = set(cls.__dataclass_fields__)
        return cls(**{name: values[name] for name in allowed if name in values})

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values.update({
            "feature_extractor_type": "SpeechT5FeatureExtractor",
            "processor_class": "SpeechT5Processor",
        })
        return values


class SpeechT5Tokenizer:
    """SpeechT5's character SentencePiece tokenizer with EOS framing."""

    def __init__(
        self,
        model_file: str | Path,
        *,
        model_max_length: int = 600,
    ) -> None:
        model_path = Path(model_file).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"SpeechT5 SentencePiece model was not found: {model_path}.")
        self.sentencepiece = SentencePieceUnigramTokenizer.from_model_file(model_path)
        self.model_file = model_path
        self.model_max_length = _positive_integer(
            model_max_length,
            name="model_max_length",
        )
        if self.sentencepiece.vocabulary_size > 81:
            raise ValueError(
                "SpeechT5 tokenizer vocabulary exceeds the published model "
                "capacity of 81 tokens.")
        self.bos_token_id = self.sentencepiece.bos_token_id
        self.pad_token_id = self.sentencepiece.pad_token_id
        self.eos_token_id = self.sentencepiece.eos_token_id
        self.unk_token_id = self.sentencepiece.unk_token_id
        expected = {
            "bos_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "unk_token_id": 3,
        }
        actual = {
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "unk_token_id": self.unk_token_id,
        }
        if actual != expected:
            raise ValueError(
                "SpeechT5 tokenizer special-token IDs do not match the "
                f"published contract: expected {expected}, found {actual}.")

    @property
    def vocab_size(self) -> int:
        return self.sentencepiece.vocabulary_size

    def encode(self, text: str) -> list[int]:
        ids = self.sentencepiece.encode_as_ids(text)
        ids.append(self.eos_token_id)
        if len(ids) > self.model_max_length:
            raise ValueError(
                "SpeechT5 token sequence has "
                f"{len(ids)} positions; the model limit is "
                f"{self.model_max_length}.")
        return ids

    def __call__(
        self,
        text: str | Sequence[str],
        *,
        padding: bool | str = False,
        return_tensors: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        rows, _ = _text_batch(text)
        encoded = [self.encode(row) for row in rows]
        do_pad = padding is True or padding == "longest"
        if padding not in (False, True, "longest", "do_not_pad"):
            raise ValueError("Native SpeechT5 tokenizer supports no padding or longest padding.")
        maximum = max(map(len, encoded)) if do_pad else None
        ids: list[list[int]] = []
        masks: list[list[int]] = []
        for row in encoded:
            amount = 0 if maximum is None else maximum - len(row)
            ids.append(row + [self.pad_token_id] * amount)
            masks.append([1] * len(row) + [0] * amount)
        if return_tensors is None:
            return {"input_ids": ids, "attention_mask": masks}
        if return_tensors != "pt":
            raise ValueError("Native SpeechT5 processing supports `return_tensors='pt'`.")
        if not do_pad and len({len(row) for row in ids}) > 1:
            raise ValueError("A variable-length SpeechT5 text batch requires padding=True.")
        torch = _torch()
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        model_path = destination / _TOKENIZER_MODEL_NAME
        if self.model_file != model_path.resolve():
            copyfile(self.model_file, model_path)
        write_json_file(
            destination / _TOKENIZER_CONFIG_NAME,
            {
                "model_max_length": self.model_max_length,
                "processor_class": "SpeechT5Processor",
                "tokenizer_class": "SpeechT5Tokenizer",
            },
        )
        write_json_file(
            destination / _SPECIAL_TOKENS_NAME,
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "unk_token": "<unk>",
            },
        )
        return model_path


class SpeechT5FeatureExtractor:
    """PyTorch implementation of the published SpeechT5 audio frontend."""

    def __init__(
        self,
        config: SpeechT5FeatureConfig | None = None,
        **overrides: Any,
    ) -> None:
        if config is not None and overrides:
            raise TypeError("Pass a feature config or keyword values, not both.")
        self.config = config or SpeechT5FeatureConfig(**overrides)

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    @property
    def num_mel_bins(self) -> int:
        return self.config.num_mel_bins

    @staticmethod
    def _reflect_pad(waveform, amount: int):
        """NumPy-compatible reflect padding, including very short inputs."""
        torch = _torch()
        length = waveform.shape[0]
        if length < 2:
            raise ValueError(
                "SpeechT5 target audio must contain at least two samples for "
                "reflect padding.")
        positions = torch.arange(
            -amount,
            length + amount,
            dtype=torch.long,
            device=waveform.device,
        )
        period = 2 * (length - 1)
        indices = torch.abs(torch.remainder(positions + length - 1, period) - (length - 1))
        return waveform.index_select(0, indices)

    def _waveform(self, value: Any, *, sampling_rate: int):
        from voicehub.processing.waveform import normalize_waveform, resample_waveform

        waveform = normalize_waveform(value)
        if sampling_rate != self.config.sampling_rate:
            waveform = resample_waveform(
                waveform,
                sampling_rate,
                self.config.sampling_rate,
            )
        return waveform

    def extract_mel(self, value: Any, *, sampling_rate: int):
        """Return one ``[frames, mel_bins]`` float32 target."""
        torch = _torch()
        from voicehub.processing.audio import mel_filter_bank

        waveform = self._waveform(value, sampling_rate=sampling_rate).to(dtype=torch.float64)
        frame_length = self.config.sample_size
        hop_length = self.config.sample_stride
        waveform = self._reflect_pad(waveform, frame_length // 2)
        frames = waveform.unfold(0, frame_length, hop_length)
        window = torch.hann_window(
            frame_length,
            periodic=True,
            dtype=torch.float64,
            device=waveform.device,
        )
        spectrum = torch.fft.rfft(
            frames * window,
            n=self.config.n_fft,
            dim=-1,
        ).to(torch.complex64)
        # The reference explicitly promotes complex64 components to float64
        # while computing magnitude.
        magnitude = torch.sqrt(
            spectrum.real.to(torch.float64).square() + spectrum.imag.to(torch.float64).square()).transpose(
                0, 1)
        filters = mel_filter_bank(
            sample_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.num_mel_bins,
            minimum_frequency=self.config.fmin,
            maximum_frequency=self.config.fmax,
            dtype=torch.float64,
            device=waveform.device,
        )
        mel = filters @ magnitude
        return mel.clamp_min(self.config.mel_floor).log10().transpose(0, 1).float()

    def __call__(
        self,
        audio: Any | None = None,
        *,
        audio_target: Any | None = None,
        sampling_rate: int | None = None,
        padding: bool | str = False,
        return_tensors: str | None = None,
        pad_to_multiple_of: int | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if (audio is None) == (audio_target is None):
            raise ValueError("Pass exactly one of `audio` or `audio_target`.")
        if sampling_rate is None:
            raise ValueError(
                "Native SpeechT5 requires the source `sampling_rate`; this "
                "prevents silent feature corruption.")
        sampling_rate = _positive_integer(sampling_rate, name="sampling_rate")
        do_pad = padding is True or padding == "longest"
        if padding not in (False, True, "longest", "do_not_pad"):
            raise ValueError("Native SpeechT5 feature extraction supports no padding or "
                             "longest padding.")

        values, _ = _audio_batch(audio_target if audio_target is not None else audio)
        target = audio_target is not None
        rows = [(
            self.extract_mel(value, sampling_rate=sampling_rate) if target else self._waveform(
                value, sampling_rate=sampling_rate)) for value in values]
        multiple = pad_to_multiple_of
        if target and multiple is None:
            multiple = self.config.reduction_factor
        if multiple is not None:
            multiple = _positive_integer(multiple, name="pad_to_multiple_of")
        maximum = max(row.shape[0] for row in rows) if do_pad else None
        if maximum is not None and multiple is not None:
            maximum = ceil(maximum / multiple) * multiple
        if not do_pad and return_tensors == "pt" and len({row.shape[0] for row in rows}) > 1:
            raise ValueError("A variable-length SpeechT5 audio batch requires padding=True.")

        padded = []
        masks = []
        torch = _torch()
        for row in rows:
            length = row.shape[0]
            requested = length if maximum is None else maximum
            amount = requested - length
            if amount:
                shape = (amount, self.config.num_mel_bins) if target else (amount, )
                padding_values = torch.full(
                    shape,
                    (-100.0 if target else self.config.padding_value),
                    dtype=row.dtype,
                    device=row.device,
                )
                row = torch.cat((row, padding_values), dim=0)
            padded.append(row)
            masks.append(
                torch.cat((
                    torch.ones(length, dtype=torch.long, device=row.device),
                    torch.zeros(amount, dtype=torch.long, device=row.device),
                )))
        if return_tensors not in (None, "pt"):
            raise ValueError("Native SpeechT5 processing supports `return_tensors='pt'`.")
        key = "labels" if target else "input_values"
        mask_key = "decoder_attention_mask" if target else "attention_mask"
        if return_tensors == "pt":
            return {
                key: torch.stack(padded),
                mask_key: torch.stack(masks),
            }
        return {
            key: [row.tolist() for row in padded],
            mask_key: [mask.tolist() for mask in masks],
        }

    def save_pretrained(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / _PREPROCESSOR_CONFIG_NAME
        write_json_file(path, self.config.to_dict())
        return path


class SpeechT5Processor:
    """Composable native tokenizer and feature extractor."""

    def __init__(
        self,
        tokenizer: SpeechT5Tokenizer,
        feature_extractor: SpeechT5FeatureExtractor | None = None,
    ) -> None:
        if not isinstance(tokenizer, SpeechT5Tokenizer):
            raise TypeError("`tokenizer` must be a native SpeechT5Tokenizer.")
        if (feature_extractor is not None and not isinstance(feature_extractor, SpeechT5FeatureExtractor)):
            raise TypeError("`feature_extractor` must be a native SpeechT5FeatureExtractor.")
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor or SpeechT5FeatureExtractor()

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
    ) -> SpeechT5Processor:
        root = Path(directory).expanduser().resolve()
        tokenizer_config_path = root / _TOKENIZER_CONFIG_NAME
        tokenizer_config = (read_json_file(tokenizer_config_path) if tokenizer_config_path.is_file() else {})
        preprocessor_path = root / _PREPROCESSOR_CONFIG_NAME
        feature_config = (
            SpeechT5FeatureConfig.from_mapping(read_json_file(preprocessor_path))
            if preprocessor_path.is_file() else SpeechT5FeatureConfig())
        return cls(
            SpeechT5Tokenizer(
                root / _TOKENIZER_MODEL_NAME,
                model_max_length=int(tokenizer_config.get("model_max_length", 600)),
            ),
            SpeechT5FeatureExtractor(feature_config),
        )

    def __call__(
        self,
        *,
        text: str | Sequence[str] | None = None,
        audio: Any | None = None,
        text_target: str | Sequence[str] | None = None,
        audio_target: Any | None = None,
        sampling_rate: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if text is not None and audio is not None:
            raise ValueError("Pass SpeechT5 text or audio input, not both.")
        if text_target is not None and audio_target is not None:
            raise ValueError("Pass SpeechT5 text or audio target, not both.")
        if all(value is None for value in (text, audio, text_target, audio_target)):
            raise ValueError("SpeechT5 processor received no input or target.")

        if text is not None:
            inputs = self.tokenizer(text, **kwargs)
        elif audio is not None:
            inputs = self.feature_extractor(
                audio,
                sampling_rate=sampling_rate,
                **kwargs,
            )
        else:
            inputs = {}

        if audio_target is not None:
            targets = self.feature_extractor(
                audio_target=audio_target,
                sampling_rate=sampling_rate,
                **kwargs,
            )
            inputs["labels"] = targets["labels"]
            inputs["decoder_attention_mask"] = targets["decoder_attention_mask"]
        elif text_target is not None:
            targets = self.tokenizer(text_target, **kwargs)
            inputs["labels"] = targets["input_ids"]
            inputs["decoder_attention_mask"] = targets["attention_mask"]
        return inputs or targets

    def save_pretrained(self, directory: str | Path) -> Path:
        self.tokenizer.save_pretrained(directory)
        return self.feature_extractor.save_pretrained(directory)


__all__ = [
    "SpeechT5FeatureConfig",
    "SpeechT5FeatureExtractor",
    "SpeechT5Processor",
    "SpeechT5Tokenizer",
]
