"""Inference runtime assembled from the official StyleTTS 2 source."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


class StyleTTS2Runtime:
    """Load and execute the architecture published with StyleTTS 2."""

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

        from voicehub.models.styletts2.source.styletts2 import models
        from voicehub.models.styletts2.source.styletts2 import utils
        from voicehub.models.styletts2.source.styletts2.Modules.diffusion.sampler import (
            ADPM2Sampler,
            DiffusionSampler,
            KarrasSchedule,
        )
        from voicehub.models.styletts2.source.styletts2.text_utils import (
            TextCleaner,
        )
        from voicehub.models.styletts2.source.styletts2.Utils.PLBERT.util import (
            load_plbert,
        )

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
        with config_file.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        source_root = (
            Path(models.__file__).resolve().parent
        )
        asset_root = (
            Path(assets_directory).expanduser().resolve()
            if assets_directory
            else Path(checkpoint_path).expanduser().resolve().parent
        )

        def resolve_asset(value: str) -> Path:
            path = Path(value).expanduser()
            if path.is_absolute():
                return path
            candidates = (
                asset_root / path,
                source_root / path,
                config_file.parent / path,
            )
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            return candidates[0]

        asr_config = resolve_asset(config["ASR_config"])
        asr_path = resolve_asset(config["ASR_path"])
        f0_path = resolve_asset(config["F0_path"])
        plbert_directory = resolve_asset(config["PLBERT_dir"])
        missing = [
            str(path)
            for path in (
                asr_config,
                asr_path,
                f0_path,
                plbert_directory,
                Path(checkpoint_path).expanduser(),
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing StyleTTS 2 checkpoint assets: "
                + ", ".join(missing)
            )

        text_aligner = models.load_ASR_models(
            str(asr_path),
            str(asr_config),
        )
        pitch_extractor = models.load_F0_models(str(f0_path))
        plbert = load_plbert(str(plbert_directory))
        self.model_params = utils.recursive_munch(
            config["model_params"]
        )
        self.model = models.build_model(
            self.model_params,
            text_aligner,
            pitch_extractor,
            plbert,
        )
        for module in self.model.values():
            module.eval().to(device)

        checkpoint = self._torch.load(
            str(Path(checkpoint_path).expanduser()),
            map_location="cpu",
            weights_only=False,
        )
        parameters = checkpoint.get("net", checkpoint)
        for key, module in self.model.items():
            if key not in parameters:
                continue
            state = parameters[key]
            try:
                module.load_state_dict(state)
            except RuntimeError:
                normalized = OrderedDict(
                    (
                        name[7:] if name.startswith("module.") else name,
                        value,
                    )
                    for name, value in state.items()
                )
                module.load_state_dict(normalized, strict=False)
            module.eval()

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
        spectrogram = preprocess.get("spect_params", {})
        torchaudio = __import__("torchaudio")
        self.sample_rate = preprocess.get("sr", 24000)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.model_params.n_mels,
            n_fft=spectrogram.get("n_fft", 2048),
            win_length=spectrogram.get("win_length", 1200),
            hop_length=spectrogram.get("hop_length", 300),
        )

    def _length_to_mask(self, lengths):
        torch = self._torch
        positions = (
            torch.arange(lengths.max(), device=lengths.device)
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
        )
        return positions + 1 > lengths.unsqueeze(1)

    def _tokens(self, text: str):
        torch = self._torch
        phonemes = self.phonemizer.phonemize(
            [text.strip().replace('"', "")]
        )[0]
        phonemes = " ".join(self._word_tokenize(phonemes))
        tokens = self.text_cleaner(phonemes)
        tokens.insert(0, 0)
        return torch.LongTensor(tokens).to(self.device).unsqueeze(0)

    def _reference_style(self, audio_path: str):
        torch = self._torch
        audio, _ = self._librosa.load(audio_path, sr=self.sample_rate)
        audio, _ = self._librosa.effects.trim(audio, top_db=30)
        waveform = torch.from_numpy(audio).float()
        mel = self.to_mel(waveform)
        mel = (
            self._torch.log(1e-5 + mel.unsqueeze(0)) + 4
        ) / 4
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
        torch = self._torch
        if seed is not None:
            torch.manual_seed(seed)
        tokens = self._tokens(text)
        reference_style = (
            self._reference_style(speaker_audio_path)
            if speaker_audio_path
            else None
        )

        with torch.no_grad():
            input_lengths = torch.LongTensor(
                [tokens.shape[-1]]
            ).to(self.device)
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
            duration_encoding = self.model.bert_encoder(
                bert_duration
            ).transpose(-1, -2)
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
                reference = (
                    alpha * reference
                    + (1 - alpha) * reference_style[:, :128]
                )
                style = (
                    beta * style
                    + (1 - beta) * reference_style[:, 128:]
                )

            predictor_encoding = self.model.predictor.text_encoder(
                duration_encoding,
                style,
                input_lengths,
                text_mask,
            )
            duration_hidden, _ = self.model.predictor.lstm(
                predictor_encoding
            )
            duration = self.model.predictor.duration_proj(duration_hidden)
            duration = torch.sigmoid(duration).sum(axis=-1)
            predicted_duration = (
                torch.round(duration.squeeze()).clamp(min=1)
            )
            if reference_style is None:
                predicted_duration[-1] += 5

            alignment = torch.zeros(
                tokens.shape[-1],
                int(predicted_duration.sum().item()),
                device=self.device,
            )
            frame = 0
            for token_index, token_duration in enumerate(
                predicted_duration
            ):
                next_frame = frame + int(token_duration.item())
                alignment[token_index, frame:next_frame] = 1
                frame = next_frame
            alignment = alignment.unsqueeze(0)

            prosody_encoding = (
                predictor_encoding.transpose(-1, -2) @ alignment
            )
            text_decoder_encoding = text_encoding @ alignment
            if self.model_params.decoder.type == "hifigan":
                prosody_encoding = self._shift(prosody_encoding)
                text_decoder_encoding = self._shift(
                    text_decoder_encoding
                )

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
        return (
            audio[..., :-50]
            if self.model_params.decoder.type == "hifigan"
            else audio
        )

    def _shift(self, encoding):
        shifted = self._torch.zeros_like(encoding)
        shifted[:, :, 0] = encoding[:, :, 0]
        shifted[:, :, 1:] = encoding[:, :, :-1]
        return shifted
