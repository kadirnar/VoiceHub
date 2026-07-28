"""Tiron inference with lossless speaker and timestamp token parsing."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

from voicehub.audio import load_audio
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput, ASRSegment
from voicehub.models.asr_tiron.configuration_asr_tiron import TironASRConfig
from voicehub.models.asr_transformers.modeling_asr_transformers import TransformersASRForSpeechRecognition

_SPEAKER_TOKEN = re.compile(r"^<\|speaker(?P<index>[1-9][0-9]*)\|>$")


class TironForSpeechRecognition(TransformersASRForSpeechRecognition):
    """Transcribe one Tiron window with local speaker attribution.

    Tiron is fine-tuned from Whisper and remains compatible with the
    standard Transformers native loss and safetensors export.  Inference
    cannot use the ordinary Whisper pipeline, however: that decoder
    retains either timestamp tokens or Tiron's added speaker tokens, but
    not both.  This provider walks generated token IDs once and
    preserves all three signal types.

    Speaker labels are local to each input window.  Cross-window speaker
    linking requires an embedding/clustering pipeline and is
    deliberately not inferred by this dependency-light provider.
    """

    config_class = TironASRConfig
    default_model_name_or_path = "Trelis/tiron"
    default_max_new_tokens = 444
    maximum_window_seconds = 30.0
    expected_native_model_type = "whisper"

    def _load_native_model(self, transformers: Any) -> tuple[Any, str]:
        native_model_type = str(getattr(self.native_config, "model_type",
                                        ""), ).strip().lower().replace("-", "_")
        if native_model_type != self.expected_native_model_type:
            raise ValueError(
                f"{self.__class__.__name__} requires a native Whisper "
                f"checkpoint; received {native_model_type or '<missing>'!r}.")
        return super()._load_native_model(transformers)

    def _load_pretrained_model(self) -> None:
        super()._load_pretrained_model()
        self._configure_tiron_generation()

    def _prepare_for_inference(self) -> None:
        super()._prepare_for_inference()
        self._configure_tiron_generation()

    def _configure_tiron_generation(self) -> None:
        """Disable Whisper controls that suppress Tiron's native grammar."""
        model = self.model
        if model is None:
            return
        model_config = getattr(model, "config", None)
        if model_config is not None:
            model_config.forced_decoder_ids = None
            model_config.suppress_tokens = []
            model_config.begin_suppress_tokens = []

        generation_config = getattr(model, "generation_config", None)
        if generation_config is None:
            raise RuntimeError("The Tiron Whisper checkpoint has no generation configuration.")
        generation_config.forced_decoder_ids = None
        generation_config.language = None
        generation_config.task = None
        generation_config.suppress_tokens = None
        generation_config.begin_suppress_tokens = None
        generation_config.no_speech_threshold = None
        if hasattr(generation_config, "no_timestamps_token_id"):
            delattr(generation_config, "no_timestamps_token_id")

    @staticmethod
    def _token_id(tokenizer: Any, token: str, *, purpose: str) -> int:
        converter = getattr(tokenizer, "convert_tokens_to_ids", None)
        if not callable(converter):
            raise TypeError("The Tiron tokenizer cannot convert tokens to IDs.")
        token_id = converter(token)
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise ValueError(f"The Tiron tokenizer has no valid {purpose} token {token!r}.")
        token_id = int(token_id)
        if token_id < 0 or token_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"The Tiron tokenizer has no valid {purpose} token {token!r}.")
        return token_id

    @classmethod
    def _language_token_id(
        cls,
        tokenizer: Any,
        language: str,
    ) -> int:
        normalized = language.strip().lower()
        if normalized == "auto":
            raise ValueError(
                "Dependency-light Tiron inference requires an explicit "
                "Whisper language code. Automatic language detection is "
                "provided by the full Tiron meeting harness.")
        language_token = (
            normalized if normalized.startswith("<|") and normalized.endswith("|>") else f"<|{normalized}|>")
        try:
            return cls._token_id(
                tokenizer,
                language_token,
                purpose="language",
            )
        except ValueError as direct_error:
            prompt_builder = getattr(
                tokenizer,
                "get_decoder_prompt_ids",
                None,
            )
            if not callable(prompt_builder):
                raise direct_error
            try:
                prompt_ids = prompt_builder(
                    language=normalized,
                    task="transcribe",
                    no_timestamps=False,
                )
            except TypeError:
                prompt_ids = prompt_builder(
                    language=normalized,
                    task="transcribe",
                )
            if not isinstance(prompt_ids, Sequence) or not prompt_ids:
                raise direct_error
            first = prompt_ids[0]
            token_id = (
                first[1] if isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and
                len(first) >= 2 else first)
            if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                raise direct_error
            return int(token_id)

    @classmethod
    def _decoder_prefix(
        cls,
        tokenizer: Any,
        *,
        language: str,
    ) -> tuple[list[int], frozenset[int]]:
        start_id = cls._token_id(
            tokenizer,
            "<|startoftranscript|>",
            purpose="decoder start",
        )
        language_id = cls._language_token_id(tokenizer, language)
        transcribe_id = cls._token_id(
            tokenizer,
            "<|transcribe|>",
            purpose="transcription task",
        )
        skipped_ids = {
            start_id,
            language_id,
            transcribe_id,
        }
        for optional_id_name in ("eos_token_id", "pad_token_id"):
            optional_id = getattr(tokenizer, optional_id_name, None)
            if isinstance(optional_id, Integral) and not isinstance(
                    optional_id,
                    bool,
            ):
                skipped_ids.add(int(optional_id))
        try:
            skipped_ids.add(cls._token_id(
                tokenizer,
                "<|endoftext|>",
                purpose="end-of-text",
            ))
        except ValueError:
            pass
        return [start_id, language_id, transcribe_id], frozenset(skipped_ids)

    @classmethod
    def _timestamp_token_bounds(cls, tokenizer: Any) -> tuple[int, int]:
        no_timestamps_id = cls._token_id(
            tokenizer,
            "<|notimestamps|>",
            purpose="timestamp boundary",
        )
        last_timestamp_id = cls._token_id(
            tokenizer,
            "<|30.00|>",
            purpose="timestamp boundary",
        )
        first_timestamp_id = no_timestamps_id + 1
        if last_timestamp_id < first_timestamp_id:
            raise ValueError("The Tiron tokenizer reported an invalid timestamp token range.")
        return first_timestamp_id, last_timestamp_id

    @classmethod
    def _decode_text_tokens(
        cls,
        tokenizer: Any,
        token_ids: Sequence[int],
    ) -> str:
        if not token_ids:
            return ""
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("The Tiron tokenizer does not implement decode().")
        options = {}
        if cls._accepts_keyword(decode, "skip_special_tokens"):
            options["skip_special_tokens"] = False
        if cls._accepts_keyword(decode, "clean_up_tokenization_spaces"):
            options["clean_up_tokenization_spaces"] = False
        value = decode(list(token_ids), **options)
        if not isinstance(value, str):
            raise TypeError("The Tiron tokenizer returned non-text decoded output.")
        return value.strip()

    @staticmethod
    def _speaker_label(token: str) -> tuple[str, int] | None:
        match = _SPEAKER_TOKEN.fullmatch(token)
        if match is None:
            return None
        one_based_index = int(match.group("index"))
        return f"SPEAKER_{one_based_index - 1:02d}", one_based_index

    @classmethod
    def _parse_generated_tokens(
        cls,
        token_ids: Sequence[int],
        *,
        tokenizer: Any,
        skipped_ids: frozenset[int],
        language: str,
    ) -> tuple[str, tuple[ASRSegment, ...]]:
        """Parse Tiron's ``speaker → start → text → end`` token grammar."""
        timestamp_begin, timestamp_end = cls._timestamp_token_bounds(tokenizer)
        token_name = getattr(tokenizer, "convert_ids_to_tokens", None)
        if not callable(token_name):
            raise TypeError("The Tiron tokenizer cannot convert IDs to tokens.")

        segments: list[ASRSegment] = []
        text_buffer: list[int] = []
        current_speaker: str | None = None
        current_speaker_index: int | None = None
        current_start: float | None = None

        def flush(*, end: float | None = None) -> None:
            nonlocal current_start
            text = cls._decode_text_tokens(tokenizer, text_buffer)
            text_buffer.clear()
            if not text:
                current_start = None
                return
            valid_end = (end if end is None or current_start is None or end >= current_start else None)
            metadata = {
                "speaker_scope": "window",
            }
            if current_speaker_index is not None:
                metadata["local_speaker_index"] = current_speaker_index
            segments.append(
                ASRSegment(
                    text=text,
                    start=current_start,
                    end=valid_end,
                    language=language,
                    speaker=current_speaker,
                    metadata=metadata,
                ))
            current_start = None

        for raw_token_id in token_ids:
            if isinstance(raw_token_id, bool) or not isinstance(
                    raw_token_id,
                    Integral,
            ):
                raise TypeError("Tiron generation returned a non-integer token ID.")
            token_id = int(raw_token_id)
            if token_id in skipped_ids:
                continue
            rendered_token = token_name(token_id)
            rendered_token = (str(rendered_token) if rendered_token is not None else "")
            if rendered_token == "<|nospeech|>":
                text_buffer.clear()
                current_speaker = None
                current_speaker_index = None
                current_start = None
                continue
            speaker = cls._speaker_label(rendered_token)
            if speaker is not None:
                flush()
                current_speaker, current_speaker_index = speaker
                continue
            if timestamp_begin <= token_id <= timestamp_end:
                timestamp = round(
                    (token_id - timestamp_begin) * 0.02,
                    2,
                )
                if text_buffer:
                    flush(end=timestamp)
                else:
                    current_start = timestamp
                continue
            text_buffer.append(token_id)
        flush()

        text = " ".join(segment.text for segment in segments if segment.text).strip()
        return text, tuple(segments)

    @staticmethod
    def _generated_token_ids(generated: Any) -> list[int]:
        if isinstance(generated, Mapping):
            generated = generated.get("sequences", generated)
        else:
            generated = getattr(generated, "sequences", generated)
        if isinstance(generated, tuple):
            generated = generated[0] if generated else []
        tolist = getattr(generated, "tolist", None)
        if callable(tolist):
            generated = tolist()
        if (isinstance(generated, Sequence) and generated and isinstance(generated[0], Sequence) and
                not isinstance(generated[0], (str, bytes))):
            generated = generated[0]
        if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
            raise TypeError("Tiron generation did not return a token sequence.")
        return list(generated)

    def _model_input_device(self):
        return getattr(self.model, "device", None) or self.device

    def _move_input_features(self, input_features: Any) -> Any:
        move = getattr(input_features, "to", None)
        if not callable(move):
            return input_features
        options = {
            "device": self._model_input_device(),
        }
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            options["dtype"] = model_dtype
        try:
            moved = move(**options)
        except TypeError:
            moved = move(options["device"])
            if model_dtype is not None:
                moved = moved.to(dtype=model_dtype)
        return input_features if moved is None else moved

    def _extract_input_features(self, waveform: Any, *, sampling_rate: int):
        processor = self.transformers_processor
        feature_extractor = getattr(processor, "feature_extractor", None)
        if not callable(feature_extractor):
            raise RuntimeError("The Tiron processor has no callable feature extractor.")
        encoded = feature_extractor(
            waveform,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        if isinstance(encoded, Mapping):
            input_features = encoded.get("input_features")
        else:
            input_features = getattr(encoded, "input_features", None)
        if input_features is None:
            raise TypeError("The Tiron feature extractor did not return `input_features`.")
        return self._move_input_features(input_features)

    def _process_training_audio(
        self,
        audio: Any,
        *,
        sampling_rate: int,
    ) -> Mapping[str, Any]:
        """Create the fixed-length mel input required by Whisper's encoder."""
        processor = self.transformers_processor
        if not callable(processor):
            raise RuntimeError("Tiron training input preparation requires a loaded processor.")
        encoded = processor(
            audio=audio,
            sampling_rate=sampling_rate,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("The Tiron processor did not return a mapping.")
        return encoded

    def _tokenize_training_labels(
        self,
        text: str | list[str],
    ) -> Mapping[str, Any]:
        """Tokenize Tiron targets with its timestamp-emitting decoder prefix.

        ``WhisperTokenizer`` defaults to ``<|notimestamps|>`` when it
        adds special tokens. That prefix contradicts Tiron targets
        containing speaker and timestamp tokens. Tokenize the payload
        without implicit specials, then add the language/transcription
        controls used for inference and one EOS token before padding.
        The model itself inserts ``decoder_start_token_id`` when it
        shifts labels, so that token must not also lead the label
        sequence.
        """
        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            None,
        )
        if not callable(tokenizer):
            raise TypeError("The Tiron processor has no callable tokenizer.")
        values = [text] if isinstance(text, str) else list(text)
        encoded = tokenizer(
            values,
            add_special_tokens=False,
            padding=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("The Tiron tokenizer did not return `input_ids`.")
        token_rows = encoded["input_ids"]
        invalid_token_rows = (not isinstance(token_rows, Sequence) or isinstance(token_rows, (str, bytes)))
        if invalid_token_rows:
            raise TypeError("The Tiron tokenizer returned invalid target IDs.")
        if token_rows and isinstance(token_rows[0], Integral):
            token_rows = [token_rows]

        decoder_prefix, _ = self._decoder_prefix(
            tokenizer,
            language=self.config.default_language,
        )
        if len(decoder_prefix) < 2:
            raise ValueError("The Tiron decoder prefix is incomplete.")
        label_prefix = decoder_prefix[1:]
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, bool) or not isinstance(
                eos_token_id,
                Integral,
        ):
            raise ValueError("The Tiron tokenizer has no valid EOS token ID.")
        eos_token_id = int(eos_token_id)

        sequences = []
        for row in token_rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError("The Tiron tokenizer returned invalid target IDs.")
            normalized_row = []
            for token_id in row:
                if isinstance(token_id, bool) or not isinstance(
                        token_id,
                        Integral,
                ):
                    raise TypeError("The Tiron tokenizer returned a non-integer target ID.")
                normalized_row.append(int(token_id))
            sequences.append([
                *label_prefix,
                *normalized_row,
                eos_token_id,
            ])
        if not sequences:
            raise ValueError("Tiron training targets cannot be empty.")

        pad = getattr(tokenizer, "pad", None)
        if not callable(pad):
            raise TypeError("The Tiron tokenizer does not implement pad().")
        padded = pad(
            {
                "input_ids": sequences,
            },
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if not isinstance(padded, Mapping):
            raise TypeError("The Tiron tokenizer did not return a padded mapping.")
        return padded

    @staticmethod
    def _validate_inference_controls(
        *,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        hotwords,
    ) -> None:
        if task != "transcribe":
            raise ValueError("Tiron supports transcription, not speech translation.")
        if return_timestamps == "word":
            raise ValueError(
                "Tiron exposes native speaker-segment timestamps, not "
                "word-level timestamps.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Dependency-light Tiron accepts one audio window up to 30 "
                "seconds; chunking and cross-window speaker linking belong "
                "to the full Tiron meeting harness.")
        if batch_size not in (None, 1):
            raise ValueError("Dependency-light Tiron transcribes one audio window at a time.")
        if num_beams not in (None, 1):
            raise ValueError("Tiron's published decoding contract requires `num_beams=1`.")
        if hotwords is not None:
            raise ValueError("Tiron does not expose a hotword decoding input.")

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
        hotwords=None,
    ) -> ASROutput:
        self._validate_inference_controls(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            hotwords=hotwords,
        )
        resolved_language = (self.config.default_language if language is None else language.strip().lower())
        target_rate = self._processor_sample_rate()
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_rate,
        )
        tolerance = 1.0 / target_rate
        if materialized.duration > self.maximum_window_seconds + tolerance:
            raise ValueError("Dependency-light Tiron accepts at most 30 seconds of audio "
                             "per call.")

        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            None,
        )
        if tokenizer is None:
            raise RuntimeError("The Tiron processor has no tokenizer.")
        decoder_prefix, skipped_ids = self._decoder_prefix(
            tokenizer,
            language=resolved_language,
        )
        input_features = self._extract_input_features(
            materialized.waveform,
            sampling_rate=materialized.sampling_rate,
        )

        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        decoder_input_ids = torch.tensor(
            [decoder_prefix],
            device=self._model_input_device(),
            dtype=torch.long,
        )
        generation_limit = max_new_tokens or self.default_max_new_tokens
        with torch.inference_mode():
            generated = self.model.generate(
                input_features=input_features,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=generation_limit,
                do_sample=False,
                num_beams=1,
            )
        token_ids = self._generated_token_ids(generated)
        text, segments = self._parse_generated_tokens(
            token_ids,
            tokenizer=tokenizer,
            skipped_ids=skipped_ids,
            language=resolved_language,
        )
        return ASROutput(
            text=text,
            segments=segments,
            language=resolved_language,
            duration=materialized.duration,
            metadata={
                "backend": "tiron",
                "architecture_family": "speech-seq2seq",
                "speaker_scope": "window",
                "native_segment_timestamps": True,
                "requested_timestamps": return_timestamps,
                "max_new_tokens": generation_limit,
            },
        )


__all__ = ["TironForSpeechRecognition"]
