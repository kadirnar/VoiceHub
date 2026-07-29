"""Native Whisper inference and fine-tuning wrapper."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from voicehub.audio_modeling_utils import PreTrainedASRModel
from voicehub.hub import read_json_file, write_json_file
from voicehub.modeling_outputs import ASROutput, ASRSegment
from voicehub.models.asr_whisper_native.configuration_asr_whisper_native import WhisperASRConfig


def _token_set_from_tokenizer(tokenizer: Any) -> Any:
    from voicehub.architectures.whisper.decoding import WhisperTokenSet

    language_tokens = {code: tokenizer.to_language_token(code) for code in tokenizer.all_language_codes}
    return WhisperTokenSet(
        eot=tokenizer.eot,
        sot=tokenizer.sot,
        translate=tokenizer.translate,
        transcribe=tokenizer.transcribe,
        sot_lm=tokenizer.sot_lm,
        sot_prev=tokenizer.sot_prev,
        no_speech=tokenizer.no_speech,
        no_timestamps=tokenizer.no_timestamps,
        timestamp_begin=tokenizer.timestamp_begin,
        language_tokens=language_tokens,
        non_speech_tokens=tokenizer.non_speech_tokens(),
        blank_token_ids=tuple(tokenizer.encode(" ").input_ids),
    )


class WhisperForSpeechRecognition(PreTrainedASRModel):
    """Run and fine-tune official Whisper Safetensors with VoiceHub code."""

    config_class = WhisperASRConfig
    default_model_name_or_path = "openai/whisper-large-v3-turbo"
    architecture_family = "speech-seq2seq"

    def __init__(
        self,
        config: WhisperASRConfig | str | Path | None = None,
        *,
        model_path: str | Path | None = None,
        device: str = "auto",
        lazy_load: bool = True,
        token: str | bool | None = None,
        **kwargs: Any,
    ) -> None:
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("`token` must be a string, boolean, or None.")
        if isinstance(token, str) and not token.strip():
            raise ValueError("String `token` values must be non-empty.")
        self._hub_token = token
        self.artifacts: Any | None = None
        self.native_config: Any | None = None
        self.tokenizer: Any | None = None
        self.generation_adapter: Any | None = None
        self._generation_values: dict[str, Any] = {}
        config = self._coerce_config(
            config,
            model_path=model_path,
            **kwargs,
        )
        super().__init__(
            config,
            device=device,
            lazy_load=lazy_load,
        )

    def _model_dtype(self) -> Any:
        import torch

        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        configured = self.config.torch_dtype
        if configured == "auto":
            return (torch.float16 if torch.device(self.device).type in {"cuda", "mps"} else torch.float32)
        dtype = dtypes[configured]
        if torch.device(self.device).type == "cpu" and dtype == torch.float16:
            raise ValueError(
                "Native Whisper does not support float16 execution on CPU; "
                "use float32 or bfloat16.")
        return dtype

    def _validate_tokenizer_vocabulary(
        self,
        tokenizer: Any,
        native_config: Any,
    ) -> None:
        """Validate the standard Whisper tokenizer/model ID space."""
        if tokenizer.vocabulary_size != native_config.vocab_size:
            raise ValueError(
                "Whisper tokenizer/model vocabulary mismatch: tokenizer has "
                f"{tokenizer.vocabulary_size} IDs, model expects "
                f"{native_config.vocab_size}.")

    @staticmethod
    def _validate_preprocessor(
        values: Mapping[str, Any],
        config: Any,
    ) -> None:
        expected = {
            "feature_size": config.num_mel_bins,
            "sampling_rate": 16_000,
            "hop_length": 160,
            "n_fft": 400,
        }
        for name, expected_value in expected.items():
            value = values.get(name)
            if value is not None and value != expected_value:
                raise ValueError(
                    f"Whisper preprocessor {name!r} is {value!r}; the model "
                    f"requires {expected_value!r}.")

    def _load_pretrained_model(self) -> None:
        from voicehub.architectures.whisper.artifacts import resolve_whisper_artifacts
        from voicehub.architectures.whisper.checkpoint import (
            HuggingFaceWhisperCheckpointAdapter,
            NativeWhisperCheckpointAdapter,
        )
        from voicehub.architectures.whisper.configuration import WhisperConfig
        from voicehub.architectures.whisper.decoding import WhisperGenerationAdapter, WhisperTokenSet
        from voicehub.architectures.whisper.modeling import WhisperModel
        from voicehub.architectures.whisper.tokenization import WhisperTokenizer
        from voicehub.checkpointing import SafeTensorReader, ShardedSafeTensorReader

        source = self.config.name_or_path or self.default_model_name_or_path
        artifacts = resolve_whisper_artifacts(
            source,
            checkpoint_filename=self.config.checkpoint_filename,
            tokenizer_filename=self.config.tokenizer_filename,
            cache_dir=self.config.cache_dir,
            revision=self.config.revision,
            token=self._hub_token,
            local_files_only=self.config.local_files_only,
        )
        architecture_values = read_json_file(artifacts.config)
        native_config = WhisperConfig.from_dict(architecture_values)
        if artifacts.preprocessor_config is not None:
            self._validate_preprocessor(
                read_json_file(artifacts.preprocessor_config),
                native_config,
            )

        model = WhisperModel(native_config)
        reader_type = (ShardedSafeTensorReader if artifacts.is_sharded else SafeTensorReader)
        adapter = (
            NativeWhisperCheckpointAdapter() if architecture_values.get("voicehub_checkpoint_format")
            == "native-whisper-v1" else HuggingFaceWhisperCheckpointAdapter())
        with reader_type(artifacts.checkpoint) as reader:
            adapter.load_streaming(
                model,
                reader,
                architecture_values,
                strict=True,
            )

        dtype = self._model_dtype()
        model.to(device=self.device, dtype=dtype)
        tokenizer = WhisperTokenizer.from_tokenizer_json(
            artifacts.tokenizer,
            multilingual=None,
        )
        self._validate_tokenizer_vocabulary(tokenizer, native_config)
        generation_values = ({} if artifacts.generation_config is None else read_json_file(
            artifacts.generation_config))
        token_set = (
            WhisperTokenSet.from_huggingface_config(generation_values)
            if generation_values else _token_set_from_tokenizer(tokenizer))

        self.artifacts = artifacts
        self.native_config = native_config
        self.tokenizer = tokenizer
        self._generation_values = generation_values
        self.generation_adapter = WhisperGenerationAdapter(
            model,
            token_set,
            tokenizer=tokenizer,
        )
        self.model = model

    def _feature_operation(self) -> Any:
        from voicehub.processing.audio import LogMelSpectrogram

        if self.native_config is None:
            raise RuntimeError("Whisper must be loaded before preprocessing.")
        return LogMelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            hop_length=160,
            n_mels=self.native_config.num_mel_bins,
            dynamic_range=8.0,
            whisper_scaling=True,
        )

    def _chunk_features(self, waveform: Any) -> Any:
        from voicehub.processing.audio import PadOrTrimAudio

        if self.native_config is None:
            raise RuntimeError("Whisper must be loaded before preprocessing.")
        target_samples = self.native_config.expected_input_frames * 160
        padded = PadOrTrimAudio(target_samples).process({"waveform": waveform})["padded_waveform"]
        features = self._feature_operation().process({"waveform": padded})["input_features"]
        return features.unsqueeze(0).to(
            device=self.model.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _normalized_language(self, language: str | None) -> str | None:
        if language is None:
            return None
        if self.tokenizer is None or self.generation_adapter is None:
            raise RuntimeError("Whisper must be loaded before decoding.")
        token_id = self.tokenizer.to_language_token(language)
        for code, candidate in (self.generation_adapter.token_set.language_tokens.items()):
            if candidate == token_id:
                return code
        raise ValueError(
            f"Language {language!r} is not declared by the checkpoint's "
            "generation configuration.")

    def _decode_segments(
        self,
        token_ids: Sequence[int],
        *,
        chunk_offset: float,
        chunk_duration: float,
        language: str | None,
    ) -> tuple[ASRSegment, ...]:
        if self.tokenizer is None:
            raise RuntimeError("Whisper tokenizer is not loaded.")
        segments: list[ASRSegment] = []
        text_tokens: list[int] = []
        start_seconds: float | None = None
        for token_id in token_ids:
            if token_id == self.tokenizer.eot:
                break
            timestamp = self.tokenizer.timestamp_for_token(token_id)
            if timestamp is None:
                text_tokens.append(token_id)
                continue
            if start_seconds is None:
                start_seconds = timestamp.seconds
                continue
            if text_tokens:
                text = self.tokenizer.decode(
                    text_tokens,
                    skip_special_tokens=True,
                ).strip()
                if text:
                    start = min(start_seconds, chunk_duration)
                    end = min(
                        max(timestamp.seconds, start),
                        chunk_duration,
                    )
                    if end > start:
                        segments.append(
                            ASRSegment(
                                text=text,
                                start=chunk_offset + start,
                                end=chunk_offset + end,
                                language=language,
                            ))
                text_tokens = []
            start_seconds = timestamp.seconds
        if text_tokens:
            text = self.tokenizer.decode(
                text_tokens,
                skip_special_tokens=True,
            ).strip()
            start = 0.0 if start_seconds is None else start_seconds
            if text and chunk_duration > start:
                segments.append(
                    ASRSegment(
                        text=text,
                        start=chunk_offset + start,
                        end=chunk_offset + chunk_duration,
                        language=language,
                    ))
        return tuple(segments)

    @staticmethod
    def _join_chunk_text(chunks: Sequence[str]) -> str:
        combined = ""
        for chunk in chunks:
            if not chunk:
                continue
            if not combined:
                combined = chunk
            elif chunk[0].isspace() or combined[-1].isspace():
                combined += chunk
            else:
                combined += " " + chunk
        return combined.strip()

    def _transcribe(
        self,
        audio: Any,
        *,
        sampling_rate: int | None = None,
        language: str | None = None,
        task: str = "transcribe",
        return_timestamps: bool | str = False,
        chunk_length_s: float | None = None,
        stride_length_s: float | tuple[float, float] | None = None,
        batch_size: int | None = None,
        num_beams: int | None = None,
        max_new_tokens: int | None = None,
        hotwords: str | tuple[str, ...] | list[str] | None = None,
    ) -> ASROutput:
        from voicehub.architectures.whisper.decoding import WhisperDecodingConfig
        from voicehub.generation import GenerationConfig
        from voicehub.processing.waveform import load_native_audio

        if stride_length_s is not None:
            raise ValueError(
                "Native Whisper currently owns exact 30-second framing; "
                "`stride_length_s` is not yet exposed.")
        if batch_size not in (None, 1):
            raise ValueError("One public audio request requires `batch_size=1`.")
        if num_beams not in (None, 1):
            raise ValueError(
                "Native Whisper currently provides greedy decoding and "
                "requires `num_beams=1`.")
        if hotwords is not None:
            raise ValueError(
                "Native Whisper does not expose `hotwords` biasing through "
                "the common ASR option.")
        if return_timestamps == "word":
            raise ValueError(
                "Native Whisper provides segment timestamps; word timestamps "
                "require a separate alignment architecture.")
        if self.native_config is None or self.tokenizer is None:
            raise RuntimeError("Whisper runtime is not loaded.")
        if self.generation_adapter is None:
            raise RuntimeError("Whisper generation adapter is not loaded.")

        materialized = load_native_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=16_000,
        )
        maximum_chunk_seconds = (self.native_config.expected_input_frames * 160 / 16_000)
        resolved_chunk_seconds = (maximum_chunk_seconds if chunk_length_s is None else float(chunk_length_s))
        if resolved_chunk_seconds > maximum_chunk_seconds:
            raise ValueError(f"Whisper chunks cannot exceed {maximum_chunk_seconds:g} "
                             "seconds.")
        chunk_samples = max(1, round(resolved_chunk_seconds * 16_000))
        generated_limit = (
            self.native_config.max_target_positions - 4 if max_new_tokens is None else max_new_tokens)
        requested_language = self._normalized_language(language)
        chunk_texts: list[str] = []
        segments: list[ASRSegment] = []
        detected_language: str | None = requested_language
        reverse_languages = {
            token_id: code
            for code, token_id in (self.generation_adapter.token_set.language_tokens.items())
        }

        for start in range(0, materialized.waveform.numel(), chunk_samples):
            stop = min(
                start + chunk_samples,
                materialized.waveform.numel(),
            )
            chunk = materialized.waveform[start:stop]
            features = self._chunk_features(chunk)
            decoding = WhisperDecodingConfig(
                generation=GenerationConfig(
                    max_new_tokens=generated_limit,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eot,
                    pad_token_id=self.tokenizer.eot,
                    use_cache=True,
                ),
                task=task,
                language=requested_language,
                return_timestamps=bool(return_timestamps),
                suppress_tokens=tuple(self._generation_values.get("suppress_tokens", ())),
            )
            generated = self.generation_adapter.generate(
                features,
                config=decoding,
            )
            tokens = generated.generated_sequences[0].tolist()
            raw_text = self.tokenizer.decode(
                tokens,
                skip_special_tokens=True,
            )
            chunk_texts.append(raw_text)
            if (detected_language is None and generated.language_token_ids is not None):
                detected_language = reverse_languages.get(int(generated.language_token_ids[0].item()))
            if return_timestamps:
                segments.extend(
                    self._decode_segments(
                        tokens,
                        chunk_offset=start / 16_000,
                        chunk_duration=(stop - start) / 16_000,
                        language=detected_language,
                    ))

        return ASROutput(
            text=self._join_chunk_text(chunk_texts),
            segments=tuple(segments),
            language=detected_language,
            duration=materialized.duration,
            metadata={
                "architecture": "whisper",
                "backend": "voicehub-native",
                "checkpoint_revision": (None if self.artifacts is None else self.artifacts.revision),
            },
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        import torch

        from voicehub.processing.waveform import load_native_audio

        del phase
        if "input_features" in inputs and "labels" in inputs:
            return dict(inputs)
        if self.model is None:
            self.load_for_training()
        if self.native_config is None or self.tokenizer is None:
            raise RuntimeError("Whisper training processor is not loaded.")
        audio = inputs.get("audio")
        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if audio is None or not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Whisper training records require `audio` and non-empty "
                "`text`/`transcription`.")
        materialized = load_native_audio(
            audio,
            sampling_rate=inputs.get(
                "sampling_rate",
                inputs.get("sample_rate"),
            ),
            target_sampling_rate=16_000,
        )
        maximum_samples = self.native_config.expected_input_frames * 160
        if materialized.waveform.numel() > maximum_samples:
            raise ValueError(
                "One Whisper training example cannot exceed the model's "
                "30-second audio context.")
        features = self._chunk_features(materialized.waveform).squeeze(0)
        language = inputs.get("language")
        if language is None:
            language = self.inference_config.to_dict().get("language")
        if language is None and self.generation_adapter.token_set.is_multilingual:
            language = "en"
        task = inputs.get(
            "task",
            self.inference_config.to_dict().get("task", "transcribe"),
        )
        prefix = self.tokenizer.prompt_tokens(
            language=language,
            task=task,
            include_no_timestamps=True,
        )
        content = self.tokenizer.encode(text).input_ids
        label_ids = (*prefix, *content, self.tokenizer.eot)
        if len(label_ids) > self.native_config.max_target_positions:
            raise ValueError(
                "Whisper transcript exceeds the decoder context after "
                f"tokenization ({len(label_ids)} > "
                f"{self.native_config.max_target_positions}).")
        return {
            "input_features": features,
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

    def _save_pretrained(self, save_directory: Path) -> None:
        from voicehub.checkpointing import save_safetensors

        if self.model is None or self.native_config is None:
            self.load()
        if self.artifacts is None:
            raise RuntimeError("Whisper artifact metadata is unavailable.")
        save_directory.mkdir(parents=True, exist_ok=True)
        save_safetensors(
            self.model.state_dict(),
            save_directory / "model.safetensors",
            metadata={"format": "voicehub-native-whisper-v1"},
        )
        config_values = self.native_config.to_dict()
        config_values["model_type"] = self.config_class.model_type
        config_values["sample_rate"] = 16_000
        config_values["torch_dtype"] = self.config.torch_dtype
        config_values["voicehub_checkpoint_format"] = "native-whisper-v1"
        config_values["architectures"] = ["WhisperModel"]
        write_json_file(save_directory / "config.json", config_values)
        shutil.copy2(
            self.artifacts.tokenizer,
            save_directory / "tokenizer.json",
        )
        for source, filename in (
            (self.artifacts.generation_config, "generation_config.json"),
            (self.artifacts.preprocessor_config, "preprocessor_config.json"),
        ):
            if source is not None:
                shutil.copy2(source, save_directory / filename)


__all__ = ["WhisperForSpeechRecognition"]
