"""Inference runtime assembled from the official StyleTTS 2 source."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class StyleTTS2Runtime:
    """Load and execute the architecture published with StyleTTS 2."""

    _CRITICAL_CHECKPOINT_MODULES = frozenset({
        "bert",
        "bert_encoder",
        "decoder",
        "diffusion",
        "predictor",
        "predictor_encoder",
        "style_encoder",
        "text_encoder",
    })

    def train(self, mode: bool = True):
        """Set every owned neural component to the requested mode."""
        for module in self.model.values():
            train = getattr(module, "train", None)
            if callable(train):
                train(mode)
        return self

    def eval(self):
        """Put every owned neural component in inference mode."""
        for module in self.model.values():
            evaluate = getattr(module, "eval", None)
            if callable(evaluate):
                evaluate()
        return self

    def __init__(
        self,
        *,
        checkpoint_path: str,
        config_path: str,
        assets_directory: str | None,
        device: str,
        language: str,
    ):
        import yaml

        from voicehub.models.styletts2.source.styletts2 import models, utils
        from voicehub.models.styletts2.source.styletts2.Modules.diffusion.sampler import (
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
        )
        from voicehub.models.styletts2.source.styletts2.text_utils import TextCleaner
        from voicehub.models.styletts2.source.styletts2.Utils.PLBERT.util import load_plbert

        self._torch = __import__("torch")
        self._librosa = __import__("librosa")
        self._word_tokenize = __import__(
            "nltk.tokenize",
            fromlist=["wordpunct_tokenize"],
        ).wordpunct_tokenize
        phonemizer = __import__("phonemizer")
        self.device = device
        self.text_cleaner = TextCleaner()
        self.phonemizer = phonemizer.backend.EspeakBackend(
            language=language,
            preserve_punctuation=True,
            with_stress=True,
        )

        config_file = Path(config_path).expanduser().resolve()
        config = self._load_config(yaml, config_file)
        source_root = Path(models.__file__).resolve().parent
        asset_root = (
            Path(assets_directory).expanduser().resolve()
            if assets_directory else Path(checkpoint_path).expanduser().resolve().parent)
        search_roots = (asset_root, source_root, config_file.parent)
        asr_config = self._resolve_asset(config["ASR_config"], search_roots)
        asr_path = self._resolve_asset(config["ASR_path"], search_roots)
        f0_path = self._resolve_asset(config["F0_path"], search_roots)
        plbert_directory = self._resolve_asset(config["PLBERT_dir"], search_roots)
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        self._require_assets((
            asr_config,
            asr_path,
            f0_path,
            plbert_directory,
            checkpoint_file,
        ))

        text_aligner = models.load_ASR_models(
            str(asr_path),
            str(asr_config),
        )
        pitch_extractor = models.load_F0_models(str(f0_path))
        plbert = load_plbert(str(plbert_directory))
        self.model_params = utils.recursive_munch(config["model_params"])
        self.model = models.build_model(
            self.model_params,
            text_aligner,
            pitch_extractor,
            plbert,
        )
        for module in self.model.values():
            module.eval().to(device)

        checkpoint = self._load_checkpoint(checkpoint_file)
        parameters = checkpoint.get("net", checkpoint)
        if not isinstance(parameters, Mapping):
            raise TypeError("StyleTTS 2 checkpoint must contain a state mapping under `net`.")
        loaded_modules: set[str] = set()
        for key, module in self.model.items():
            if key not in parameters:
                continue
            state = parameters[key]
            if not isinstance(state, Mapping):
                raise TypeError(f"StyleTTS 2 checkpoint entry {key!r} is not a state mapping.")
            self._load_module_checkpoint(key, module, state)
            module.eval()
            loaded_modules.add(key)
        if not loaded_modules:
            raise RuntimeError(
                "The StyleTTS 2 checkpoint does not contain weights for any "
                "runtime component.")
        missing_critical = sorted(self._CRITICAL_CHECKPOINT_MODULES - loaded_modules)
        if missing_critical:
            raise RuntimeError(
                "The StyleTTS 2 checkpoint is missing weights for critical "
                "runtime component(s): " + ", ".join(missing_critical) + ".")

        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(
                sigma_min=0.0001,
                sigma_max=3.0,
                rho=9.0,
            ),
            clamp=False,
        )
        preprocess = config.get("preprocess_params", {})
        if not isinstance(preprocess, Mapping):
            raise TypeError("StyleTTS 2 `preprocess_params` must be a mapping.")
        spectrogram = preprocess.get("spect_params", {})
        if not isinstance(spectrogram, Mapping):
            raise TypeError("StyleTTS 2 `spect_params` must be a mapping.")
        torchaudio = __import__("torchaudio")
        self.sample_rate = int(preprocess.get("sr", 24000))
        if self.sample_rate <= 0:
            raise ValueError("StyleTTS 2 sample rate must be positive.")
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.model_params.n_mels,
            n_fft=spectrogram.get("n_fft", 2048),
            win_length=spectrogram.get("win_length", 1200),
            hop_length=spectrogram.get("hop_length", 300),
        )

    @staticmethod
    def _load_config(yaml: Any, config_file: Path) -> Mapping[str, Any]:
        if not config_file.is_file():
            raise FileNotFoundError(f"StyleTTS 2 configuration was not found: {config_file}.")
        with config_file.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if not isinstance(config, Mapping):
            raise TypeError("StyleTTS 2 configuration must contain a YAML mapping.")
        required = {
            "ASR_config",
            "ASR_path",
            "F0_path",
            "PLBERT_dir",
            "model_params",
        }
        missing = sorted(required - set(config))
        if missing:
            raise ValueError(
                "StyleTTS 2 configuration is missing required key(s): " + ", ".join(missing) + ".")
        return config

    @staticmethod
    def _resolve_asset(value: Any, search_roots: tuple[Path, ...]) -> Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise TypeError("StyleTTS 2 asset paths must be non-empty strings.")
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        candidates = tuple(root / path for root in search_roots)
        return next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            candidates[0].resolve(),
        )

    @staticmethod
    def _require_assets(paths: tuple[Path, ...]) -> None:
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing StyleTTS 2 checkpoint asset(s): " + ", ".join(missing))

    @staticmethod
    def _normalize_state_dict(state: Mapping[str, Any]) -> OrderedDict[str, Any]:
        return OrderedDict((
            name[7:] if name.startswith("module.") else name,
            value,
        ) for name, value in state.items())

    @classmethod
    def _load_module_checkpoint(
        cls,
        module_name: str,
        module: Any,
        state: Mapping[str, Any],
    ) -> set[str]:
        """Load one component only when checkpoint parameters really match."""
        module_parameters = {name for name, _ in module.named_parameters()}
        if not module_parameters:
            module_parameters = set(module.state_dict())

        candidate = state
        matching = module_parameters.intersection(candidate)
        try:
            module.load_state_dict(candidate)
        except RuntimeError:
            candidate = cls._normalize_state_dict(state)
            matching = module_parameters.intersection(candidate)
            if not matching:
                raise RuntimeError(
                    f"StyleTTS 2 checkpoint component {module_name!r} has "
                    "no parameter keys matching the runtime module.")
            incompatible = module.load_state_dict(candidate, strict=False)
            unexpected = set(getattr(incompatible, "unexpected_keys", ()))
            matching.difference_update(unexpected)

        if not matching:
            raise RuntimeError(
                f"StyleTTS 2 checkpoint component {module_name!r} loaded no "
                "matching parameter keys.")
        return matching

    def _load_checkpoint(self, checkpoint_file: Path) -> Mapping[str, Any]:
        try:
            checkpoint = self._torch.load(
                str(checkpoint_file),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            # ``weights_only`` was introduced in PyTorch 2.0. The StyleTTS 2
            # extra still permits older compatible patch versions.
            checkpoint = self._torch.load(
                str(checkpoint_file),
                map_location="cpu",
            )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("StyleTTS 2 checkpoint must contain a state mapping.")
        return checkpoint

    def _length_to_mask(self, lengths):
        torch = self._torch
        positions = torch.arange(
            lengths.max(),
            device=lengths.device,
        ).unsqueeze(0).expand(lengths.shape[0], -1)
        return positions + 1 > lengths.unsqueeze(1)

    def _tokens(self, text: str):
        torch = self._torch
        phonemes = self.phonemizer.phonemize([text.strip().replace('"', "")])[0]
        phonemes = " ".join(self._word_tokenize(phonemes))
        tokens = self.text_cleaner(phonemes)
        if not tokens:
            raise ValueError("StyleTTS 2 phonemization produced no speakable tokens.")
        tokens.insert(0, 0)
        return torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    def _reference_style(self, audio_path: str):
        torch = self._torch
        audio, _ = self._librosa.load(audio_path, sr=self.sample_rate)
        audio, _ = self._librosa.effects.trim(audio, top_db=30)
        if audio.size == 0:
            raise ValueError("StyleTTS 2 reference audio contains no samples after silence trimming.")
        waveform = torch.from_numpy(audio).float()
        mel = self.to_mel(waveform)
        mel = (self._torch.log(1e-5 + mel.unsqueeze(0)) + 4) / 4
        mel = mel.to(self.device).unsqueeze(1)
        with torch.no_grad():
            reference = self.model.style_encoder(mel)
            prosody = self.model.predictor_encoder(mel)
        return torch.cat([reference, prosody], dim=1)

    def generate(
        self,
        text: str,
        *,
        speaker_audio_path: str | None,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
        seed: int | None,
    ):
        self._validate_request(
            text=text,
            speaker_audio_path=speaker_audio_path,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            seed=seed,
        )
        torch = self._torch
        tokens = self._tokens(text)
        reference_style = (self._reference_style(speaker_audio_path) if speaker_audio_path else None)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths)
            text_encoding = self.model.text_encoder(
                tokens,
                input_lengths,
                text_mask,
            )
            bert_duration = self.model.bert(
                tokens,
                attention_mask=(~text_mask).int(),
            )
            duration_encoding = self.model.bert_encoder(bert_duration).transpose(-1, -2)
            noise = torch.randn((1, 256), device=self.device).unsqueeze(1)

            if reference_style is None:
                style_prediction = self.sampler(
                    noise,
                    embedding=bert_duration[0].unsqueeze(0),
                    num_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                ).squeeze(0)
            else:
                style_prediction = self.sampler(
                    noise=noise,
                    embedding=bert_duration,
                    embedding_scale=embedding_scale,
                    features=reference_style,
                    num_steps=diffusion_steps,
                ).squeeze(1)

            style = style_prediction[:, 128:]
            reference = style_prediction[:, :128]
            if reference_style is not None:
                reference = (alpha * reference + (1 - alpha) * reference_style[:, :128])
                style = (beta * style + (1 - beta) * reference_style[:, 128:])

            predictor_encoding = self.model.predictor.text_encoder(
                duration_encoding,
                style,
                input_lengths,
                text_mask,
            )
            duration_hidden, _ = self.model.predictor.lstm(predictor_encoding)
            duration = self.model.predictor.duration_proj(duration_hidden)
            duration = torch.sigmoid(duration).sum(axis=-1)
            predicted_duration = torch.round(duration).clamp(min=1).reshape(-1)
            if reference_style is None:
                predicted_duration[-1] += 5

            alignment = torch.zeros(
                tokens.shape[-1],
                int(predicted_duration.sum().item()),
                device=self.device,
            )
            frame = 0
            for token_index, token_duration in enumerate(predicted_duration):
                next_frame = frame + int(token_duration.item())
                alignment[token_index, frame:next_frame] = 1
                frame = next_frame
            alignment = alignment.unsqueeze(0)

            prosody_encoding = (predictor_encoding.transpose(-1, -2) @ alignment)
            text_decoder_encoding = text_encoding @ alignment
            if self.model_params.decoder.type == "hifigan":
                prosody_encoding = self._shift(prosody_encoding)
                text_decoder_encoding = self._shift(text_decoder_encoding)

            f0, noise_prediction = self.model.predictor.F0Ntrain(
                prosody_encoding,
                style,
            )
            output = self.model.decoder(
                text_decoder_encoding,
                f0,
                noise_prediction,
                reference.squeeze().unsqueeze(0),
            )
        audio = output.squeeze().cpu().numpy()
        if getattr(audio, "size", 0) == 0:
            raise RuntimeError("StyleTTS 2 returned an empty audio waveform.")
        if self.model_params.decoder.type != "hifigan":
            return audio
        if audio.shape[-1] <= 50:
            raise RuntimeError("StyleTTS 2 returned fewer samples than the HiFi-GAN trim size.")
        return audio[..., :-50]

    @staticmethod
    def _validate_request(
        *,
        text: str,
        speaker_audio_path: str | None,
        alpha: float,
        beta: float,
        diffusion_steps: int,
        embedding_scale: float,
        seed: int | None,
    ) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("`text` must be a non-empty string.")
        if speaker_audio_path is not None and not Path(speaker_audio_path).expanduser().is_file():
            raise FileNotFoundError(f"StyleTTS 2 reference audio was not found: {speaker_audio_path}.")
        for name, value in (("alpha", alpha), ("beta", beta)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"`{name}` must be numeric.")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"`{name}` must be in the interval [0, 1].")
        if (not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool) or
                diffusion_steps <= 0):
            raise ValueError("`diffusion_steps` must be a positive integer.")
        if (not isinstance(embedding_scale, (int, float)) or isinstance(embedding_scale, bool) or
                not math.isfinite(embedding_scale) or embedding_scale <= 0):
            raise ValueError("`embedding_scale` must be a finite positive number.")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("`seed` must be an integer or None.")

    def _shift(self, encoding):
        shifted = self._torch.zeros_like(encoding)
        shifted[:, :, 0] = encoding[:, :, 0]
        shifted[:, :, 1:] = encoding[:, :, :-1]
        return shifted
