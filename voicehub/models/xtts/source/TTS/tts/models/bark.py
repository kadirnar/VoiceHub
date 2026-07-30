import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from torch import nn
from coqpit import Coqpit

from voicehub.components.audio.codecs.encodec import (
    EncodecModel,
    load_encodec_model,
)

from voicehub.models.xtts.source.TTS.tts.layers.bark.inference_funcs import (
    codec_decode,
    generate_coarse,
    generate_fine,
    generate_text_semantic,
    generate_voice,
    load_voice,
)
from voicehub.models.xtts.source.TTS.tts.layers.bark.load_model import load_model
from voicehub.models.xtts.source.TTS.tts.layers.bark.model import GPT
from voicehub.models.xtts.source.TTS.tts.layers.bark.model_fine import FineGPT

if TYPE_CHECKING:
    from transformers import BertTokenizer


@dataclass
class BarkAudioConfig(Coqpit):
    sample_rate: int = 24000
    output_sample_rate: int = 24000


def _load_default_tokenizer():
    """Load Bark's legacy text tokenizer without import-time network access."""
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def _config_value(config: Coqpit, name: str, default):
    value = getattr(config, name, default)
    return default if value is None else value


def _boolean_option(value, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"`{name}` must be a boolean.")
    return value


def load_bark_encodec(
    config: Coqpit,
    *,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool | None = None,
    trust_official_pickle: bool | None = None,
) -> EncodecModel:
    """Load Bark's exact 24 kHz codec through VoiceHub's safe boundary.

    Native Safetensors checkpoints are accepted without a trust override.
    Meta's published ``.th`` artifact is a legacy pickle container and is
    therefore loaded only after the caller explicitly opts in; VoiceHub then
    verifies its pinned size, digest, tensor namespace, shapes, and inventory.
    """
    resolved_checkpoint = checkpoint
    if resolved_checkpoint is None:
        resolved_checkpoint = _config_value(
            config,
            "ENCODEC_CHECKPOINT",
            None,
        )
    resolved_cache_dir = (
        cache_dir
        if cache_dir is not None
        else _config_value(config, "ENCODEC_CACHE_DIR", None)
    )
    resolved_local_only = _boolean_option(
        (
            local_files_only
            if local_files_only is not None
            else _config_value(config, "ENCODEC_LOCAL_FILES_ONLY", False)
        ),
        name="local_files_only",
    )
    resolved_trust = _boolean_option(
        (
            trust_official_pickle
            if trust_official_pickle is not None
            else _config_value(
                config,
                "TRUST_OFFICIAL_ENCODEC_PICKLE",
                False,
            )
        ),
        name="trust_official_pickle",
    )

    load_options = {
        "checkpoint": resolved_checkpoint,
        "cache_dir": resolved_cache_dir,
        "local_files_only": resolved_local_only,
        "trust_official_pickle": resolved_trust,
    }
    if resolved_checkpoint is None and not resolved_trust:
        # Resolve only already-cached artifacts here. This prevents an
        # untrusted pickle from being downloaded before it is rejected while
        # still allowing a converted Safetensors artifact to work by default.
        load_options["local_files_only"] = True
        try:
            codec = load_encodec_model("encodec_24khz", **load_options)
        except FileNotFoundError as error:
            if resolved_local_only:
                raise
            raise PermissionError(
                "Bark needs pretrained Encodec weights. Provide a converted "
                "`.safetensors` checkpoint through `ENCODEC_CHECKPOINT`, or "
                "set `TRUST_OFFICIAL_ENCODEC_PICKLE=True` to download and "
                "strictly verify Meta's pinned legacy `.th` release."
            ) from error
    else:
        codec = load_encodec_model("encodec_24khz", **load_options)

    expected_sample_rate = _config_value(config, "sample_rate", 24_000)
    if (
        isinstance(expected_sample_rate, bool)
        or not isinstance(expected_sample_rate, int)
        or expected_sample_rate <= 0
    ):
        raise ValueError("Bark's `sample_rate` must be a positive integer.")
    if codec.sample_rate != expected_sample_rate:
        raise ValueError(
            "Bark and Encodec sample rates must match; received "
            f"{expected_sample_rate} Hz and {codec.sample_rate} Hz."
        )
    if codec.channels != 1 or codec.quantizer.bins != 1_024:
        raise ValueError(
            "Bark requires the official mono Encodec graph with 1,024-entry "
            "codebooks."
        )
    codec.set_target_bandwidth(6.0)
    return codec


class Bark(nn.Module):
    def __init__(
        self,
        config: Coqpit,
        tokenizer: "BertTokenizer | None" = None,
        *,
        encodec_model: EncodecModel | None = None,
        encodec_checkpoint: str | Path | None = None,
        encodec_cache_dir: str | Path | None = None,
        encodec_local_files_only: bool | None = None,
        trust_official_encodec_pickle: bool | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if tokenizer is None:
            tokenizer = _load_default_tokenizer()
        self.config.num_chars = len(tokenizer)
        self.tokenizer = tokenizer
        self.semantic_model = GPT(config.semantic_config)
        self.coarse_model = GPT(config.coarse_config)
        self.fine_model = FineGPT(config.fine_config)
        if encodec_model is None:
            encodec_model = load_bark_encodec(
                config,
                checkpoint=encodec_checkpoint,
                cache_dir=encodec_cache_dir,
                local_files_only=encodec_local_files_only,
                trust_official_pickle=trust_official_encodec_pickle,
            )
        elif not isinstance(encodec_model, EncodecModel):
            raise TypeError("`encodec_model` must be a native EncodecModel.")
        self.encodec = encodec_model
        self.encodec.set_target_bandwidth(6.0)

    @property
    def device(self):
        return next(self.parameters()).device

    def load_bark_models(self):
        self.semantic_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["text"], device=self.device, config=self.config, model_type="text"
        )
        self.coarse_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["coarse"],
            device=self.device,
            config=self.config,
            model_type="coarse",
        )
        self.fine_model, self.config = load_model(
            ckpt_path=self.config.LOCAL_MODEL_PATHS["fine"], device=self.device, config=self.config, model_type="fine"
        )

    def train_step(
        self,
    ):
        pass

    def text_to_semantic(
        self,
        text: str,
        history_prompt: Optional[str] = None,
        temp: float = 0.7,
        base=None,
        allow_early_stop=True,
        **kwargs,
    ):
        """Generate semantic array from text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy semantic array to be fed into `semantic_to_waveform`
        """
        x_semantic = generate_text_semantic(
            text,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        return x_semantic

    def semantic_to_waveform(
        self,
        semantic_tokens: np.ndarray,
        history_prompt: Optional[str] = None,
        temp: float = 0.7,
        base=None,
    ):
        """Generate audio array from semantic input.

        Args:
            semantic_tokens: semantic token output from `text_to_semantic`
            history_prompt: history choice for audio cloning
            temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_coarse_gen = generate_coarse(
            semantic_tokens,
            self,
            history_prompt=history_prompt,
            temp=temp,
            base=base,
        )
        x_fine_gen = generate_fine(
            x_coarse_gen,
            self,
            history_prompt=history_prompt,
            temp=0.5,
            base=base,
        )
        audio_arr = codec_decode(x_fine_gen, self)
        return audio_arr, x_coarse_gen, x_fine_gen

    def generate_audio(
        self,
        text: str,
        history_prompt: Optional[str] = None,
        text_temp: float = 0.7,
        waveform_temp: float = 0.7,
        base=None,
        allow_early_stop=True,
        **kwargs,
    ):
        """Generate audio array from input text.

        Args:
            text: text to be turned into audio
            history_prompt: history choice for audio cloning
            text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
            waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)

        Returns:
            numpy audio array at sample frequency 24khz
        """
        x_semantic = self.text_to_semantic(
            text,
            history_prompt=history_prompt,
            temp=text_temp,
            base=base,
            allow_early_stop=allow_early_stop,
            **kwargs,
        )
        audio_arr, c, f = self.semantic_to_waveform(
            x_semantic, history_prompt=history_prompt, temp=waveform_temp, base=base
        )
        return audio_arr, [x_semantic, c, f]

    def generate_voice(self, audio, speaker_id, voice_dir):
        """Generate a voice from the given audio and text.

        Args:
            audio (str): Path to the audio file.
            speaker_id (str): Speaker name.
            voice_dir (str): Path to the directory to save the generate voice.
        """
        if voice_dir is not None:
            voice_dirs = [voice_dir]
            try:
                _ = load_voice(speaker_id, voice_dirs)
            except (KeyError, FileNotFoundError):
                output_path = os.path.join(voice_dir, speaker_id + ".npz")
                os.makedirs(voice_dir, exist_ok=True)
                generate_voice(audio, self, output_path)

    def _set_voice_dirs(self, voice_dirs):
        def_voice_dir = None
        if isinstance(self.config.DEF_SPEAKER_DIR, str):
            os.makedirs(self.config.DEF_SPEAKER_DIR, exist_ok=True)
            if os.path.isdir(self.config.DEF_SPEAKER_DIR):
                def_voice_dir = self.config.DEF_SPEAKER_DIR
        _voice_dirs = [def_voice_dir] if def_voice_dir is not None else []
        if voice_dirs is not None:
            if isinstance(voice_dirs, str):
                voice_dirs = [voice_dirs]
            _voice_dirs = voice_dirs + _voice_dirs
        return _voice_dirs

    # TODO: remove config from synthesize
    def synthesize(
        self, text, config, speaker_id="random", voice_dirs=None, **kwargs
    ):  # pylint: disable=unused-argument
        """Synthesize speech with the given input text.

        Args:
            text (str): Input text.
            config (BarkConfig): Config with inference parameters.
            speaker_id (str): One of the available speaker names. If `random`, it generates a random speaker.
            speaker_wav (str): Path to the speaker audio file for cloning a new voice. It is cloned and saved in
                `voice_dirs` with the name `speaker_id`. Defaults to None.
            voice_dirs (List[str]): List of paths that host reference audio files for speakers. Defaults to None.
            **kwargs: Model specific inference settings used by `generate_audio()` and `TTS.tts.layers.bark.inference_funcs.generate_text_semantic().

        Returns:
            A dictionary of the output values with `wav` as output waveform, `deterministic_seed` as seed used at inference,
            `text_input` as text token IDs after tokenizer, `voice_samples` as samples used for cloning, `conditioning_latents`
            as latents used at inference.

        """
        speaker_id = "random" if speaker_id is None else speaker_id
        voice_dirs = self._set_voice_dirs(voice_dirs)
        history_prompt = load_voice(self, speaker_id, voice_dirs)
        outputs = self.generate_audio(text, history_prompt=history_prompt, **kwargs)
        return_dict = {
            "wav": outputs[0],
            "text_inputs": text,
        }

        return return_dict

    def eval_step(self):
        ...

    def forward(self):
        ...

    def inference(self):
        ...

    @staticmethod
    def init_from_config(config: "BarkConfig", **kwargs):  # pylint: disable=unused-argument
        return Bark(config)

    # pylint: disable=unused-argument, redefined-builtin
    def load_checkpoint(
        self,
        config,
        checkpoint_dir,
        text_model_path=None,
        coarse_model_path=None,
        fine_model_path=None,
        hubert_model_path=None,
        hubert_tokenizer_path=None,
        eval=False,
        strict=True,
        **kwargs,
    ):
        """Load a model checkpoints from a directory. This model is with multiple checkpoint files and it
        expects to have all the files to be under the given `checkpoint_dir` with the rigth names.
        If eval is True, set the model to eval mode.

        Args:
            config (TortoiseConfig): The model config.
            checkpoint_dir (str): The directory where the checkpoints are stored.
            ar_checkpoint_path (str, optional): The path to the autoregressive checkpoint. Defaults to None.
            diff_checkpoint_path (str, optional): The path to the diffusion checkpoint. Defaults to None.
            clvp_checkpoint_path (str, optional): The path to the CLVP checkpoint. Defaults to None.
            vocoder_checkpoint_path (str, optional): The path to the vocoder checkpoint. Defaults to None.
            eval (bool, optional): Whether to set the model to eval mode. Defaults to False.
            strict (bool, optional): Whether to load the model strictly. Defaults to True.
        """
        text_model_path = text_model_path or os.path.join(checkpoint_dir, "text_2.pt")
        coarse_model_path = coarse_model_path or os.path.join(checkpoint_dir, "coarse_2.pt")
        fine_model_path = fine_model_path or os.path.join(checkpoint_dir, "fine_2.pt")
        hubert_model_path = hubert_model_path or os.path.join(checkpoint_dir, "hubert.pt")
        hubert_tokenizer_path = hubert_tokenizer_path or os.path.join(checkpoint_dir, "tokenizer.pth")

        self.config.LOCAL_MODEL_PATHS["text"] = text_model_path
        self.config.LOCAL_MODEL_PATHS["coarse"] = coarse_model_path
        self.config.LOCAL_MODEL_PATHS["fine"] = fine_model_path
        self.config.LOCAL_MODEL_PATHS["hubert"] = hubert_model_path
        self.config.LOCAL_MODEL_PATHS["hubert_tokenizer"] = hubert_tokenizer_path

        self.load_bark_models()

        if eval:
            self.eval()
