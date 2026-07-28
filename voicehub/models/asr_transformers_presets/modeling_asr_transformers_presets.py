"""Architecture-specific presets built on the universal Transformers ASR
runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from voicehub.audio import load_audio
from voicehub.dependencies import import_optional
from voicehub.modeling_outputs import ASROutput, ASRSegment, ASRWord
from voicehub.models.asr_transformers.modeling_asr_transformers import TransformersASRForSpeechRecognition
from voicehub.models.asr_transformers_presets.configuration_asr_transformers_presets import (
    CohereASRConfig,
    HubertASRConfig,
    MedASRConfig,
    MoonshineASRConfig,
    NemotronASRConfig,
    ParakeetTDTASRConfig,
    SeamlessM4Tv2ASRConfig,
    Wav2Vec2ASRConfig,
    WavLMASRConfig,
    WhisperASRConfig,
)

_NEMOTRON_LANGUAGE_TAG = re.compile(r"<(?P<locale>[a-z]{2,3}-[A-Za-z]{2})>", )


class _TransformersASRPresetForSpeechRecognition(TransformersASRForSpeechRecognition):
    """Validate a preset's native architecture before allocating weights."""

    expected_native_model_types: frozenset[str] = frozenset()

    def _load_native_model(self, transformers: Any) -> tuple[Any, str]:
        native_model_type = str(getattr(self.native_config, "model_type", ""), ).strip().lower()
        normalized_model_type = native_model_type.replace("-", "_")
        if normalized_model_type not in self.expected_native_model_types:
            expected = ", ".join(sorted(self.expected_native_model_types))
            raise ValueError(
                f"{self.__class__.__name__} requires a Transformers "
                f"checkpoint with model type {expected}; received "
                f"{native_model_type or '<missing>'!r}. Use "
                "TransformersASRForSpeechRecognition for dynamic "
                "architecture dispatch.")
        family = self.config.architecture_family
        model_class = self._auto_model_class(transformers, family)
        model = model_class.from_pretrained(
            self._transformers_model_source(),
            **self._model_load_kwargs(),
        )
        return model, family


class _JointProcessorTrainingASRPreset(_TransformersASRPresetForSpeechRecognition):
    """Prepare native ASR labels through one audio-and-text processor call.

    Transducer processors construct decoder inputs alongside labels,
    while some sequence-to-sequence processors add language prompts.
    Tokenizing the transcript separately would silently discard those
    architecture-specific tensors. Presets using this base therefore
    retain their processor's complete, differentiable training contract.
    """

    _RAW_TRAINING_FIELDS = frozenset({
        "audio",
        "audio_lengths",
        "language",
        "punctuation",
        "sample_rate",
        "sampling_rate",
        "text",
        "transcript",
        "transcription",
    })

    def _joint_processor_options(
        self,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        del inputs
        return {}

    def _postprocess_joint_training_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize processor fields for both raw and cached batches."""
        return batch

    def _joint_processor_training_batch(
        self,
        *,
        audio: Any,
        text: str | list[str] | None,
        sampling_rate: int,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        processor = self.transformers_processor
        if processor is None:
            raise RuntimeError("Training input preparation requires load_for_training().")
        encoded = processor(
            audio=audio,
            text=text,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
            **self._joint_processor_options(inputs),
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("The native Transformers ASR processor did not return a "
                            "mapping.")
        batch = dict(encoded)
        if text is not None and "labels" not in batch:
            raise TypeError(
                "The native Transformers ASR processor did not return "
                "`labels` for the supplied transcriptions.")
        return self._postprocess_joint_training_batch(batch)

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Prepare raw records without dropping native decoder inputs."""
        del phase
        if "input_values" in inputs or "input_features" in inputs:
            return self._postprocess_joint_training_batch(dict(inputs))
        audio = inputs.get("audio")
        if audio is None:
            return dict(inputs)
        if self.transformers_processor is None:
            raise RuntimeError("Training input preparation requires load_for_training().")

        text = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if text is not None and not isinstance(text, (str, list, tuple)):
            raise TypeError(
                "ASR training transcription must be a string, a sequence of "
                "strings, or None.")
        text_values = (list(text) if isinstance(text, (list, tuple)) and not isinstance(text, str) else None)
        candidate_texts = (text_values if text_values is not None else ([] if text is None else [text]))
        if any(not isinstance(value, str) or not value.strip() for value in candidate_texts):
            raise ValueError("ASR training transcriptions must contain non-empty strings.")

        expected_size = len(text_values) if text_values is not None else None
        audio_values = self._as_batch(
            audio,
            expected_size=expected_size,
        )
        if text_values is None and text is not None and len(audio_values) != 1:
            raise ValueError("Batched ASR audio requires one transcription per waveform.")
        audio_values = self._trim_audio_batch(
            audio_values,
            inputs.get("audio_lengths"),
        )
        rate_values = self._batch_scalar_values(
            inputs.get("sampling_rate", inputs.get("sample_rate")),
            batch_size=len(audio_values),
            name="sampling_rate",
            broadcast=True,
        )
        target_rate = self._processor_sample_rate()
        waveforms = [
            load_audio(
                value,
                sampling_rate=rate,
                target_sampling_rate=target_rate,
            ).waveform for value, rate in zip(audio_values, rate_values)
        ]
        processor_audio = waveforms if len(waveforms) > 1 else waveforms[0]
        processor_text = text_values if text_values is not None else text
        batch = self._joint_processor_training_batch(
            audio=processor_audio,
            text=processor_text,
            sampling_rate=target_rate,
            inputs=inputs,
        )

        # Explicit pre-tokenized fields remain authoritative. This permits
        # advanced callers to resume from cached native decoder inputs while
        # raw text follows the processor-owned preparation path above.
        for name, value in inputs.items():
            if name not in self._RAW_TRAINING_FIELDS:
                batch[name] = value
        return batch


class _NativeProcessorGenerationASRPreset(_JointProcessorTrainingASRPreset):
    """Run architectures whose processor contract cannot be split apart."""

    supports_native_timestamps = False
    accepts_processor_language = False
    supports_beam_search = True

    def _resolve_native_processor_language(
        self,
        language: str | None,
    ) -> str | None:
        if language is None:
            return None
        if not self.accepts_processor_language:
            raise ValueError(f"{self.__class__.__name__} does not support a runtime "
                             "`language` override.")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty string or None.")
        return language.strip()

    def _native_processor_inference_options(
        self,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        del language
        return {}

    def _validate_native_inference_request(
        self,
        *,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
    ) -> None:
        if task != "transcribe":
            raise ValueError(f"{self.__class__.__name__} does not support speech "
                             "translation.")
        if chunk_length_s is not None or stride_length_s is not None:
            raise ValueError(
                "Native processor ASR presets own their long-audio framing; "
                "`chunk_length_s` and `stride_length_s` are not supported.")
        if batch_size is not None:
            raise ValueError(
                "`batch_size` is not supported for a single native-processor "
                "transcription request.")
        if (num_beams not in (None, 1) and not self.supports_beam_search):
            raise ValueError(
                f"{self.__class__.__name__} uses greedy transducer decoding "
                "and requires `num_beams=1` or None.")
        if hotwords is not None:
            raise ValueError(
                f"{self.__class__.__name__} does not expose native hotword "
                "biasing through the common ASR option.")
        if return_timestamps:
            if not self.supports_native_timestamps:
                raise ValueError(f"{self.__class__.__name__} does not expose native "
                                 "timestamps.")
            if return_timestamps not in (True, "word"):
                raise ValueError("Native transducer timestamps support `True`, `'word'`, "
                                 "or `False`.")

    def _native_generation_options(
        self,
        *,
        num_beams: int | None,
        max_new_tokens: int | None,
        options: dict[str, Any],
        processor_inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        nested = options.pop("generate_kwargs", None)
        if nested is None:
            generation_options: dict[str, Any] = {}
        elif isinstance(nested, Mapping):
            generation_options = dict(nested)
        else:
            raise TypeError("`generate_kwargs` must be a mapping or None.")
        generation_options.update(options)
        self._merge_generation_option(
            generation_options,
            "num_beams",
            num_beams,
        )
        self._merge_generation_option(
            generation_options,
            "max_new_tokens",
            max_new_tokens,
        )
        self._merge_generation_option(
            generation_options,
            "return_dict_in_generate",
            True,
        )
        conflicts = sorted(set(processor_inputs).intersection(generation_options))
        if conflicts:
            names = ", ".join(conflicts)
            raise ValueError("Generation options cannot replace native processor "
                             f"tensor(s): {names}.")
        return generation_options

    def _move_native_processor_batch(
        self,
        batch: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        model_device = getattr(self.model, "device", None) or self.device
        model_dtype = getattr(self.model, "dtype", None)
        move_batch = getattr(batch, "to", None)
        if callable(move_batch):
            try:
                moved = (
                    move_batch(device=model_device, dtype=model_dtype)
                    if model_dtype is not None else move_batch(device=model_device))
            except TypeError:
                moved = move_batch(model_device)
            if moved is not None:
                batch = moved
            if not isinstance(batch, Mapping):
                raise TypeError(
                    "The native ASR processor batch did not remain a "
                    "mapping after device placement.")
            return batch

        moved_batch: dict[str, Any] = {}
        for name, value in batch.items():
            move_value = getattr(value, "to", None)
            if not callable(move_value):
                moved_batch[name] = value
                continue
            dtype = getattr(value, "dtype", None)
            is_floating = bool(getattr(dtype, "is_floating_point", False))
            try:
                moved_batch[name] = (
                    move_value(device=model_device, dtype=model_dtype)
                    if model_dtype is not None and is_floating else move_value(device=model_device))
            except TypeError:
                moved_batch[name] = move_value(model_device)
        return moved_batch

    @staticmethod
    def _generation_field(
        generation_output: Any,
        name: str,
    ) -> Any:
        if isinstance(generation_output, Mapping):
            return generation_output.get(name)
        return getattr(generation_output, name, None)

    @classmethod
    def _generation_sequences(cls, generation_output: Any) -> Any:
        sequences = cls._generation_field(generation_output, "sequences")
        return generation_output if sequences is None else sequences

    @staticmethod
    def _single_decoded_text(decoded: Any) -> str:
        if isinstance(decoded, str):
            return decoded.strip()
        if (isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes))):
            if len(decoded) != 1:
                raise TypeError("A single-audio native ASR request must decode to one "
                                "transcription.")
            return str(decoded[0]).strip()
        raise TypeError("The native ASR processor must decode text or a one-item "
                        "sequence.")

    @classmethod
    def _timestamp_words(
        cls,
        offsets: Any,
    ) -> tuple[ASRWord, ...]:
        plain_offsets = cls._plain_value(offsets)
        if (isinstance(plain_offsets, Sequence) and not isinstance(plain_offsets, (str, bytes)) and
                len(plain_offsets) == 1 and isinstance(plain_offsets[0], Sequence) and
                not isinstance(plain_offsets[0], (str, bytes, Mapping))):
            plain_offsets = plain_offsets[0]
        if not isinstance(plain_offsets, Sequence) or isinstance(plain_offsets, (str, bytes)):
            return ()

        words: list[ASRWord] = []
        for value in plain_offsets:
            if not isinstance(value, Mapping):
                continue
            is_explicit_word = "word" in value or ("text" in value and "token" not in value)
            raw_text = str(value.get(
                "word",
                value.get("text", value.get("token", "")),
            ))
            has_boundary = (bool(raw_text[:1].isspace()) or raw_text.startswith(("▁", "Ġ")))
            is_continuation = raw_text.startswith("##")
            text = raw_text.replace("▁", " ").replace("Ġ", " ")
            if is_continuation:
                text = text.removeprefix("##")
            text = text.strip()
            if not text:
                continue
            start, end = cls._timestamp(value)
            if start is None and end is None:
                continue
            is_punctuation = all(not character.isalnum() for character in text)
            if words and (is_continuation or is_punctuation or not is_explicit_word and not has_boundary):
                previous = words[-1]
                words[-1] = ASRWord(
                    text=previous.text + text,
                    start=previous.start,
                    end=end if end is not None else previous.end,
                    confidence=previous.confidence,
                    speaker=previous.speaker,
                )
                continue
            # Boundary markers are removed above. Unmarked tokenizer pieces
            # extend the preceding word; raw token offsets remain in output
            # metadata for consumers that need subword-level detail.
            words.append(ASRWord(
                text=text,
                start=start,
                end=end,
            ))
        return tuple(words)

    def _decode_native_generation(
        self,
        *,
        generation_output: Any,
        processor_inputs: Mapping[str, Any],
        language: str | None,
        return_timestamps: bool | str,
    ) -> tuple[str, tuple[ASRSegment, ...], dict[str, Any]]:
        del processor_inputs
        sequences = self._generation_sequences(generation_output)
        decode_options: dict[str, Any] = {
            "skip_special_tokens": True,
        }
        durations = self._generation_field(
            generation_output,
            "durations",
        )
        if return_timestamps:
            if durations is None:
                raise RuntimeError(
                    "Native timestamp decoding requested generation "
                    "`durations`, but the model did not return them.")
            decode_options["durations"] = durations
        decoded = self.transformers_processor.decode(
            sequences,
            **decode_options,
        )
        timestamp_offsets = None
        if return_timestamps:
            if (not isinstance(decoded, tuple) or len(decoded) != 2):
                raise TypeError("Native timestamp decoding must return "
                                "(text, offsets).")
            decoded, timestamp_offsets = decoded
        text = self._single_decoded_text(decoded)

        metadata: dict[str, Any] = {}
        segments: tuple[ASRSegment, ...] = ()
        if timestamp_offsets is not None:
            plain_offsets = self._plain_value(timestamp_offsets)
            words = self._timestamp_words(plain_offsets)
            if not text:
                text = " ".join(word.text for word in words).strip()
            starts = [word.start for word in words if word.start is not None]
            ends = [word.end for word in words if word.end is not None]
            if text or words:
                segments = (
                    ASRSegment(
                        text=text,
                        start=min(starts) if starts else None,
                        end=max(ends) if ends else None,
                        language=language,
                        words=words,
                    ), )
            metadata["native_token_timestamps"] = plain_offsets
        return text, segments, metadata

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
        **kwargs,
    ) -> ASROutput:
        resolved_language = self._resolve_native_processor_language(language)
        self._validate_native_inference_request(
            task=task,
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            hotwords=hotwords,
        )
        target_rate = self._processor_sample_rate()
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_rate,
        )
        processor_options = {
            "sampling_rate": target_rate,
            "return_tensors": "pt",
            **self._native_processor_inference_options(language=resolved_language),
        }
        processor_inputs = self.transformers_processor(
            audio=materialized.waveform,
            **processor_options,
        )
        if not isinstance(processor_inputs, Mapping):
            raise TypeError("The native Transformers ASR processor did not return a "
                            "mapping.")
        processor_inputs = self._move_native_processor_batch(processor_inputs)
        generation_options = self._native_generation_options(
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            options=dict(kwargs),
            processor_inputs=processor_inputs,
        )
        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        with torch.inference_mode():
            generation_output = self.model.generate(
                **dict(processor_inputs),
                **generation_options,
            )
        text, segments, metadata = self._decode_native_generation(
            generation_output=generation_output,
            processor_inputs=processor_inputs,
            language=resolved_language,
            return_timestamps=return_timestamps,
        )
        metadata.update({
            "backend": "transformers",
            "architecture_family": self.architecture_family,
            "native_processor": self.transformers_processor.__class__.__name__,
        })
        detected_language = metadata.get("detected_language")
        output_language = (
            detected_language if isinstance(detected_language, str) else
            None if resolved_language == "auto" else resolved_language)
        return ASROutput(
            text=text,
            segments=segments,
            language=output_language,
            duration=materialized.duration,
            metadata=metadata,
        )


class Wav2Vec2ForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """English Wav2Vec2 CTC ASR with native differentiable fine-tuning."""

    config_class = Wav2Vec2ASRConfig
    default_model_name_or_path = "facebook/wav2vec2-base-960h"
    expected_native_model_types = frozenset({"wav2vec2"})


class HubertForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """English HuBERT CTC ASR with native differentiable fine-tuning."""

    config_class = HubertASRConfig
    default_model_name_or_path = "facebook/hubert-large-ls960-ft"
    expected_native_model_types = frozenset({"hubert"})


class WavLMForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """English WavLM CTC ASR with native differentiable fine-tuning."""

    config_class = WavLMASRConfig
    default_model_name_or_path = ("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
    expected_native_model_types = frozenset({"wavlm"})


class MoonshineForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """Compact English Moonshine encoder-decoder ASR."""

    config_class = MoonshineASRConfig
    default_model_name_or_path = "UsefulSensors/moonshine-tiny"
    expected_native_model_types = frozenset({
        "moonshine",
        "moonshine_streaming",
    })


class SeamlessM4Tv2ForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """Multilingual SeamlessM4T v2 speech recognition and translation."""

    config_class = SeamlessM4Tv2ASRConfig
    default_model_name_or_path = "facebook/seamless-m4t-v2-large"
    expected_native_model_types = frozenset({"seamless_m4t_v2"})

    def _pipeline_call_options(
        self,
        *,
        language: str | None,
        task: str,
        return_timestamps: bool | str,
        chunk_length_s: float | None,
        stride_length_s: float | tuple[float, float] | None,
        batch_size: int | None,
        num_beams: int | None,
        max_new_tokens: int | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        # SeamlessM4T selects both recognition and translation output with
        # `tgt_lang`; its generate method does not accept Whisper's `task` or
        # `language` keywords.
        call_options = super()._pipeline_call_options(
            language=None,
            task="transcribe",
            return_timestamps=return_timestamps,
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            hotwords=hotwords,
            options=options,
        )
        target_language = language or self.config.target_language
        generation_options = dict(call_options.pop("generate_kwargs", {}) or {}, )
        self._merge_generation_option(
            generation_options,
            "tgt_lang",
            target_language,
        )
        call_options["generate_kwargs"] = generation_options
        return call_options

    def _transcribe(
        self,
        audio: Any,
        *,
        language: str | None = None,
        **kwargs,
    ):
        return super()._transcribe(
            audio,
            language=language or self.config.target_language,
            **kwargs,
        )

    def _tokenize_training_labels(
        self,
        text: str | list[str],
    ) -> Mapping[str, Any]:
        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            self.transformers_processor,
        )
        encoded = tokenizer(
            text_target=text,
            tgt_lang=self.config.target_language,
            padding=True,
            return_tensors="pt",
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("The SeamlessM4T tokenizer did not return a mapping.")
        return encoded


class WhisperForSpeechRecognition(_TransformersASRPresetForSpeechRecognition):
    """Multilingual Whisper large-v3 Turbo inference and fine-tuning."""

    config_class = WhisperASRConfig
    default_model_name_or_path = "openai/whisper-large-v3-turbo"
    expected_native_model_types = frozenset({"whisper"})


class ParakeetTDTForSpeechRecognition(_NativeProcessorGenerationASRPreset):
    """Multilingual Parakeet v3 with its native TDT objective."""

    config_class = ParakeetTDTASRConfig
    default_model_name_or_path = "nvidia/parakeet-tdt-0.6b-v3"
    expected_native_model_types = frozenset({"parakeet_tdt"})
    supports_native_timestamps = True
    supports_beam_search = False


class NemotronForSpeechRecognition(_NativeProcessorGenerationASRPreset):
    """Nemotron 3.5 streaming ASR with its native RNN-T objective."""

    config_class = NemotronASRConfig
    default_model_name_or_path = ("nvidia/nemotron-3.5-asr-streaming-0.6b")
    expected_native_model_types = frozenset({"nemotron3_5_asr"})
    accepts_processor_language = True
    supports_native_timestamps = True
    supports_beam_search = False

    def _resolve_native_processor_language(
        self,
        language: str | None,
    ) -> str:
        resolved = (self.config.target_language if language is None else language)
        validated = super()._resolve_native_processor_language(resolved)
        if validated is None:  # pragma: no cover - guarded by the config.
            raise RuntimeError("Nemotron ASR requires a processor language.")
        return validated

    def _native_processor_inference_options(
        self,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        return {
            "language": language,
            "is_streaming": False,
            "is_first_audio_chunk": True,
        }

    def _joint_processor_options(
        self,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "language": inputs.get("language", self.config.target_language),
        }

    def _postprocess_joint_training_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        decoder_input_ids = batch.get("decoder_input_ids")
        processor_blank_id = getattr(
            self.transformers_processor,
            "blank_token_id",
            None,
        )
        model_blank_id = getattr(
            getattr(self.model, "config", self.native_config),
            "blank_token_id",
            None,
        )
        if (decoder_input_ids is not None and isinstance(processor_blank_id, int) and
                isinstance(model_blank_id, int) and processor_blank_id != model_blank_id):
            masked_fill = getattr(decoder_input_ids, "masked_fill", None)
            if not callable(masked_fill):
                raise TypeError("Nemotron decoder inputs must support blank-token "
                                "normalization.")
            batch["decoder_input_ids"] = masked_fill(
                decoder_input_ids == processor_blank_id,
                model_blank_id,
            )
        return batch

    def _prepare_for_training(self) -> None:
        super()._prepare_for_training()
        loss_module = import_optional(
            "transformers.loss.loss_rnnt",
            model_type=self.config.model_type,
            install_extra=None,
        )
        native_loss = getattr(
            loss_module,
            "ParakeetForRNNTLoss",
            None,
        )
        if not callable(native_loss):
            raise RuntimeError(
                "The installed Transformers runtime does not expose its "
                "native RNN-T loss.")
        model_config = getattr(self.model, "config", None)
        blank_token_id = getattr(model_config, "blank_token_id", None)
        if not isinstance(blank_token_id, int):
            raise RuntimeError("The Nemotron checkpoint does not define a valid "
                               "`blank_token_id`.")

        def nemotron_rnnt_loss(
            *,
            logits,
            labels,
            encoder_outputs,
            **kwargs,
        ):
            attention_mask = getattr(
                encoder_outputs,
                "attention_mask",
                None,
            )
            if attention_mask is None:
                raise RuntimeError("Nemotron RNN-T loss requires the encoder output "
                                   "attention mask.")
            logit_lengths = attention_mask.sum(-1)
            return native_loss(
                logits=logits[:, :int(logit_lengths.max())],
                labels=labels,
                logit_lengths=logit_lengths,
                label_lengths=(labels != blank_token_id).sum(-1),
                blank_token_id=blank_token_id,
                **kwargs,
            )

        self.model.loss_function = nemotron_rnnt_loss

    def _decode_native_generation(
        self,
        *,
        generation_output: Any,
        processor_inputs: Mapping[str, Any],
        language: str | None,
        return_timestamps: bool | str,
    ) -> tuple[str, tuple[ASRSegment, ...], dict[str, Any]]:
        text, segments, metadata = super()._decode_native_generation(
            generation_output=generation_output,
            processor_inputs=processor_inputs,
            language=language,
            return_timestamps=return_timestamps,
        )
        if language != "auto":
            return text, segments, metadata

        tagged = self.transformers_processor.decode(
            self._generation_sequences(generation_output),
            skip_special_tokens=False,
        )
        raw_tagged = self._single_decoded_text(tagged)
        matches = tuple(_NEMOTRON_LANGUAGE_TAG.finditer(raw_tagged))
        if not matches:
            return text, segments, metadata

        detected_language = matches[-1].group("locale")
        metadata["detected_language"] = detected_language
        segments = tuple(replace(segment, language=detected_language) for segment in segments)
        return text, segments, metadata


class CohereForSpeechRecognition(_NativeProcessorGenerationASRPreset):
    """Cohere Transcribe with language-conditioned native labels."""

    config_class = CohereASRConfig
    default_model_name_or_path = ("CohereLabs/cohere-transcribe-03-2026")
    expected_native_model_types = frozenset({"cohere_asr"})
    accepts_processor_language = True

    def _resolve_native_processor_language(
        self,
        language: str | None,
    ) -> str:
        resolved = (self.config.target_language if language is None else language)
        validated = super()._resolve_native_processor_language(resolved)
        if validated is None:  # pragma: no cover - guarded by the config.
            raise RuntimeError("Cohere ASR requires a processor language.")
        return validated.lower()

    def _native_processor_inference_options(
        self,
        *,
        language: str | None,
    ) -> dict[str, Any]:
        return {
            "language": language,
            "punctuation": self.config.punctuation,
        }

    def _decode_native_generation(
        self,
        *,
        generation_output: Any,
        processor_inputs: Mapping[str, Any],
        language: str | None,
        return_timestamps: bool | str,
    ) -> tuple[str, tuple[ASRSegment, ...], dict[str, Any]]:
        del return_timestamps
        chunk_index = processor_inputs.get("audio_chunk_index")
        decoded = self.transformers_processor.decode(
            self._generation_sequences(generation_output),
            audio_chunk_index=chunk_index,
            language=language,
            skip_special_tokens=True,
        )
        text = self._single_decoded_text(decoded)
        metadata: dict[str, Any] = {}
        if chunk_index is not None:
            plain_chunk_index = self._plain_value(chunk_index)
            metadata["audio_chunk_index"] = plain_chunk_index
            if (isinstance(plain_chunk_index, Sequence) and not isinstance(plain_chunk_index, (str, bytes))):
                metadata["audio_chunk_count"] = len(plain_chunk_index)
                metadata["long_form_reassembled"] = any(
                    isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and
                    len(value) >= 2 and value[1] is not None for value in plain_chunk_index)
        return text, (), metadata

    def _joint_processor_options(
        self,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        language = inputs.get("language", self.config.target_language)
        if (isinstance(language, Sequence) and not isinstance(language, (str, bytes))):
            normalized_languages = []
            for value in language:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Cohere ASR training languages must be non-empty "
                                     "strings.")
                normalized_languages.append(value.strip().lower())
            if not normalized_languages:
                raise ValueError("Cohere ASR training requires one non-empty batch "
                                 "language.")
            if len(set(normalized_languages)) != 1:
                raise ValueError(
                    "Cohere ASR's processor accepts one language per batch. "
                    "Group training records by language before collation.")
            language = normalized_languages[0]
        if not isinstance(language, str) or not language.strip():
            raise ValueError("Cohere ASR training requires one non-empty batch language.")
        return {
            "language": language.strip().lower(),
            "punctuation": self.config.punctuation,
        }

    def _postprocess_joint_training_batch(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        had_chunk_index = "audio_chunk_index" in batch
        chunk_index = batch.pop("audio_chunk_index", None)
        if chunk_index is not None:
            plain_chunk_index = self._plain_value(chunk_index)
            if (not isinstance(plain_chunk_index, Sequence) or isinstance(plain_chunk_index, (str, bytes))):
                raise TypeError(
                    "The Cohere ASR processor returned an invalid "
                    "`audio_chunk_index` for training.")
            is_chunked = any(
                isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 2 and
                value[1] is not None for value in plain_chunk_index)
            if is_chunked:
                raise ValueError(
                    "Cohere ASR split a long training waveform into multiple "
                    "feature rows but produced one transcript label row. "
                    "Pre-segment long recordings and provide one aligned "
                    "transcript per segment before fine-tuning.")
        labels = batch.get("labels")
        decoder_prompt_ids = batch.get("decoder_input_ids")
        if labels is None or decoder_prompt_ids is None:
            raise TypeError(
                "Cohere ASR training requires processor-generated labels "
                "and decoder prompt IDs.")
        label_shape = getattr(labels, "shape", None)
        prompt_shape = getattr(decoder_prompt_ids, "shape", None)
        if label_shape is None and prompt_shape is None:
            # Dependency-free test doubles do not need to emulate tensor
            # concatenation. Real processor batches are validated below.
            return batch
        if (label_shape is None or prompt_shape is None or len(label_shape) != 2 or len(prompt_shape) != 2 or
                label_shape[0] != prompt_shape[0] or label_shape[1] < 1 or prompt_shape[1] < 1):
            raise ValueError(
                "Cohere ASR processor labels and decoder prompts must be "
                "non-empty rank-2 tensors with the same batch size.")

        # VoiceHub's normalized cache has equal decoder/label shapes, an
        # explicit decoder mask, and no inference-only chunk map. Reapplying
        # prompt alignment would duplicate the prompt on every epoch.
        if (not had_chunk_index and "decoder_attention_mask" in batch and
                tuple(label_shape) == tuple(prompt_shape)):
            return batch

        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            None,
        )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if not isinstance(pad_token_id, int):
            pad_token_id = getattr(
                getattr(self.model, "config", self.native_config),
                "pad_token_id",
                None,
            )
        if not isinstance(pad_token_id, int):
            raise RuntimeError("Cohere ASR training requires a valid tokenizer pad token.")

        torch = import_optional(
            "torch",
            model_type=self.config.model_type,
            install_extra=None,
        )
        masked_labels = labels.masked_fill(
            labels == pad_token_id,
            -100,
        )
        decoder_transcript = masked_labels[:, :-1].masked_fill(
            masked_labels[:, :-1] == -100,
            pad_token_id,
        )
        decoder_input_ids = torch.cat(
            (decoder_prompt_ids, decoder_transcript),
            dim=-1,
        )
        prompt_mask = torch.full(
            (
                int(label_shape[0]),
                int(prompt_shape[1]) - 1,
            ),
            -100,
            dtype=masked_labels.dtype,
            device=masked_labels.device,
        )
        aligned_labels = torch.cat(
            (prompt_mask, masked_labels),
            dim=-1,
        )
        if decoder_input_ids.shape != aligned_labels.shape:
            raise RuntimeError("Cohere ASR decoder inputs and aligned labels have "
                               "different shapes.")

        batch["decoder_input_ids"] = decoder_input_ids
        batch["decoder_attention_mask"] = (decoder_input_ids != pad_token_id).long()
        batch["labels"] = aligned_labels
        return batch

    def _prepare_for_training(self) -> None:
        super()._prepare_for_training()
        loss_utils = import_optional(
            "transformers.loss.loss_utils",
            model_type=self.config.model_type,
            install_extra=None,
        )
        unshifted_cross_entropy = getattr(
            loss_utils,
            "ForMaskedLMLoss",
            None,
        )
        if not callable(unshifted_cross_entropy):
            raise RuntimeError(
                "The installed Transformers runtime does not expose an "
                "unshifted token cross-entropy loss.")
        self.model.loss_function = unshifted_cross_entropy


class MedASRForSpeechRecognition(_JointProcessorTrainingASRPreset):
    """Medical LASR CTC recognition with native joint preprocessing."""

    config_class = MedASRConfig
    default_model_name_or_path = "google/medasr"
    expected_native_model_types = frozenset({"lasr_ctc"})


__all__ = [
    "CohereForSpeechRecognition",
    "HubertForSpeechRecognition",
    "MedASRForSpeechRecognition",
    "MoonshineForSpeechRecognition",
    "NemotronForSpeechRecognition",
    "ParakeetTDTForSpeechRecognition",
    "SeamlessM4Tv2ForSpeechRecognition",
    "Wav2Vec2ForSpeechRecognition",
    "WavLMForSpeechRecognition",
    "WhisperForSpeechRecognition",
]
