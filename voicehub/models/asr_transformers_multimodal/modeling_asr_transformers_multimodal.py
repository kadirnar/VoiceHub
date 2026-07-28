"""Native chat-template Transformers providers for current ASR models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from typing import Any

from voicehub.audio import load_audio
from voicehub.errors import OptionalDependencyError
from voicehub.modeling_outputs import ASROutput, ASRSegment
from voicehub.models.asr_transformers.modeling_asr_transformers import TransformersASRForSpeechRecognition
from voicehub.models.asr_transformers_multimodal.configuration_asr_transformers_multimodal import (
    MultimodalTransformersASRConfig,
    Qwen3ASRConfig,
    VibeVoiceASRConfig,
)

_QWEN3_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


class MultimodalTransformersASRForSpeechRecognition(TransformersASRForSpeechRecognition):
    """Base class for processor-native multimodal ASR checkpoints.

    Unlike the conventional Transformers ASR provider, this runtime
    never creates an ASR pipeline. The full ``AutoProcessor`` remains
    responsible for the model's chat template, multimodal placeholders,
    and causal label mask in both inference and training.
    """

    config_class = MultimodalTransformersASRConfig
    default_model_name_or_path = ""
    expected_native_model_types: frozenset[str] = frozenset()
    backend_name = "transformers-multimodal-asr"
    supports_native_timestamps = False

    @staticmethod
    def _normalized_model_type(value: Any) -> str:
        return str(value or "").strip().lower().replace("-", "_")

    def _validate_native_config(self) -> None:
        if not self.expected_native_model_types:
            return
        model_type = self._normalized_model_type(getattr(self.native_config, "model_type", None))
        expected = {self._normalized_model_type(value) for value in self.expected_native_model_types}
        if model_type not in expected:
            choices = ", ".join(sorted(self.expected_native_model_types))
            raise ValueError(
                f"{self.__class__.__name__} expected a checkpoint with native "
                f"model type {choices}; received {model_type or 'unknown'!r}.")

    def _load_multimodal_model(self, transformers: Any):
        primary_class = getattr(
            transformers,
            "AutoModelForMultimodalLM",
            None,
        )
        fallback_class = getattr(
            transformers,
            "AutoModelForSpeechSeq2Seq",
            None,
        )
        if primary_class is None and fallback_class is None:
            raise OptionalDependencyError(
                f"{self.config.model_type!r} requires a Transformers release "
                "that exposes `AutoModelForMultimodalLM` or "
                "`AutoModelForSpeechSeq2Seq`. Upgrade Transformers and retry.")

        load_options = self._model_load_kwargs()
        if primary_class is not None:
            try:
                return primary_class.from_pretrained(
                    self._transformers_model_source(),
                    **load_options,
                )
            except ValueError as error:
                if fallback_class is None or not self._is_auto_class_mismatch(error):
                    raise
        return fallback_class.from_pretrained(
            self._transformers_model_source(),
            **load_options,
        )

    def _validate_processor_contract(self) -> None:
        """Validate the processor surface required by this provider.

        Most current multimodal ASR families expose both a high-level
        transcription request and their chat template. Architectures
        with a different native contract can override this hook without
        weakening the checks for Qwen3-ASR and VibeVoice-ASR.
        """
        apply_request = getattr(
            self.transformers_processor,
            "apply_transcription_request",
            None,
        )
        apply_template = getattr(
            self.transformers_processor,
            "apply_chat_template",
            None,
        )
        if not callable(apply_request) or not callable(apply_template):
            raise TypeError(
                "The checkpoint processor must implement both "
                "`apply_transcription_request()` and `apply_chat_template()`.")

    def _load_pretrained_model(self) -> None:
        from voicehub.dependencies import import_optional

        transformers = import_optional(
            "transformers",
            model_type=self.config.model_type,
            install_extra=None,
        )
        self.native_config = transformers.AutoConfig.from_pretrained(
            self._transformers_config_source(),
            trust_remote_code=self.config.trust_remote_code,
            **self._hub_kwargs(),
        )
        self._validate_native_config()

        processor_class = getattr(transformers, "AutoProcessor", None)
        if processor_class is None:
            raise OptionalDependencyError(
                f"{self.config.model_type!r} requires `AutoProcessor`. "
                "Upgrade Transformers and retry.")
        processor_options = {
            **self._hub_kwargs(),
            **self.config.processor_kwargs,
            "trust_remote_code": self.config.trust_remote_code,
        }
        self.transformers_processor = processor_class.from_pretrained(
            self._transformers_processor_source(),
            **processor_options,
        )
        self._validate_processor_contract()

        self.model = self._load_multimodal_model(transformers)
        self.architecture_family = "speech-seq2seq"
        has_device_map = (
            "device_map" in self.config.model_kwargs or bool(getattr(self.model, "hf_device_map", None)))
        if not has_device_map:
            move = getattr(self.model, "to", None)
            if callable(move):
                moved = move(self.device)
                if moved is not None:
                    self.model = moved
        self.config.sample_rate = self._processor_sample_rate()

    def _prepare_processor_output(self, encoded: Any) -> dict[str, Any]:
        if not isinstance(encoded, Mapping):
            raise TypeError("The multimodal ASR processor must return a mapping.")
        move = getattr(encoded, "to", None)
        if callable(move):
            device = getattr(self.model, "device", self.device)
            dtype = getattr(self.model, "dtype", None)
            if dtype is None:
                moved = move(device)
            else:
                try:
                    moved = move(device, dtype)
                except TypeError:
                    moved = move(device)
            if moved is not None:
                encoded = moved
        else:
            device = getattr(self.model, "device", self.device)
            encoded = {key: self._move_value_to_device(value, device) for key, value in encoded.items()}
        return dict(encoded)

    @staticmethod
    def _move_value_to_device(value: Any, device: Any) -> Any:
        move = getattr(value, "to", None)
        if not callable(move):
            return value
        try:
            moved = move(device)
        except (TypeError, ValueError):
            return value
        return value if moved is None else moved

    @staticmethod
    def _prompt_length(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", None)
        if shape is not None:
            try:
                return int(shape[-1])
            except (IndexError, TypeError, ValueError):
                pass
        if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
            if not input_ids:
                return 0
            first = input_ids[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return len(first)
            return len(input_ids)
        raise TypeError("The transcription request did not expose measurable `input_ids`.")

    @staticmethod
    def _slice_generated_tokens(generated: Any, prompt_length: int) -> Any:
        sequences = getattr(generated, "sequences", generated)
        try:
            return sequences[:, prompt_length:]
        except (IndexError, TypeError):
            if not isinstance(sequences, Sequence) or isinstance(sequences, (str, bytes)):
                raise TypeError("The multimodal ASR model returned unsupported generated "
                                "token data.")
            if not sequences:
                return sequences
            first = sequences[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return [row[prompt_length:] for row in sequences]
            return sequences[prompt_length:]

    @staticmethod
    def _hotword_prompt(
        prompt: str | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
    ) -> str | None:
        if hotwords is None:
            return prompt
        words = [hotwords] if isinstance(hotwords, str) else list(hotwords)
        vocabulary = ", ".join(word.strip() for word in words)
        context = f"Vocabulary: {vocabulary}"
        return f"{prompt.rstrip()}\n{context}" if prompt else context

    @staticmethod
    def _generation_options(
        configured: Mapping[str, Any],
        *,
        num_beams: int | None,
        max_new_tokens: int | None,
        explicit: Mapping[str, Any] | None,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        options = dict(configured)
        options.update(dict(explicit or {}))
        options.update(extra)
        for name, value in {
                "num_beams": num_beams,
                "max_new_tokens": max_new_tokens,
        }.items():
            if value is None:
                continue
            existing = options.get(name)
            if existing is not None and existing != value:
                raise ValueError(f"Conflicting values for generation option {name!r}.")
            options[name] = value
        return options

    def _request_options(
        self,
        *,
        language: str | None,
        prompt: str | None,
        processor_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        del language
        return {
            "prompt": prompt,
            **processor_kwargs,
        }

    def _apply_transcription_request(
        self,
        *,
        waveform: Any,
        language: str | None,
        prompt: str | None,
        processor_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        request_options = self._request_options(
            language=language,
            prompt=prompt,
            processor_kwargs=processor_kwargs,
        )
        return self.transformers_processor.apply_transcription_request(
            audio=waveform,
            **request_options,
        )

    def _decode_output(
        self,
        generated_tokens: Any,
        *,
        duration: float,
        language: str | None,
    ) -> ASROutput:
        decoded = self.transformers_processor.decode(
            generated_tokens,
            return_format="parsed",
        )
        if isinstance(decoded, list) and len(decoded) == 1:
            decoded = decoded[0]
        if isinstance(decoded, Mapping):
            text = str(decoded.get(
                "transcription",
                decoded.get("text", ""),
            )).strip()
            decoded_language = decoded.get("language")
            resolved_language = (
                decoded_language.strip()
                if isinstance(decoded_language, str) and decoded_language.strip() else language)
            metadata = {
                key: value
                for key, value in decoded.items() if key not in {"language", "text", "transcription"}
            }
        else:
            text = str(decoded).strip()
            resolved_language = language
            metadata = {}
        metadata.update({
            "backend":
            self.backend_name,
            "native_model_type":
            self._normalized_model_type(getattr(self.native_config, "model_type", None)),
        })
        return ASROutput(
            text=text,
            language=resolved_language,
            duration=duration,
            metadata=metadata,
        )

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
        prompt: str | None = None,
        processor_kwargs: Mapping[str, Any] | None = None,
        generate_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> ASROutput:
        if task != "transcribe":
            raise ValueError(
                "Chat-template ASR checkpoints do not expose speech "
                "translation through the VoiceHub transcription contract.")
        unsupported = {
            "chunk_length_s": chunk_length_s,
            "stride_length_s": stride_length_s,
            "batch_size": batch_size,
        }
        enabled = [name for name, value in unsupported.items() if value is not None]
        if enabled:
            raise ValueError(
                "Chat-template ASR performs checkpoint-native long-audio "
                "handling and does not accept: "
                f"{', '.join(enabled)}.")
        if return_timestamps and not self.supports_native_timestamps:
            raise ValueError(
                "This checkpoint does not emit timestamps. Use a forced "
                "alignment model after transcription.")
        if (return_timestamps and self.supports_native_timestamps and
                return_timestamps not in (True, "segment")):
            raise ValueError(
                "This checkpoint emits speaker-segment timestamps. Use "
                "`return_timestamps=True`, `'segment'`, or False.")
        if prompt is not None and (not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("`prompt` must be a non-empty string or None.")
        if processor_kwargs is not None and not isinstance(processor_kwargs, Mapping):
            raise TypeError("`processor_kwargs` must be a mapping or None.")
        if generate_kwargs is not None and not isinstance(generate_kwargs, Mapping):
            raise TypeError("`generate_kwargs` must be a mapping or None.")

        target_rate = self._processor_sample_rate()
        materialized = load_audio(
            audio,
            sampling_rate=sampling_rate,
            target_sampling_rate=target_rate,
        )
        prompt = self._hotword_prompt(
            prompt.strip() if prompt else None,
            hotwords,
        )
        encoded = self._apply_transcription_request(
            waveform=materialized.waveform,
            language=language,
            prompt=prompt,
            processor_kwargs=dict(processor_kwargs or {}),
        )
        model_inputs = self._prepare_processor_output(encoded)
        if "input_ids" not in model_inputs:
            raise TypeError(
                "The transcription request must emit `input_ids` so generated "
                "prompt tokens can be removed.")
        generation_options = self._generation_options(
            self.config.generation_config,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            explicit=generate_kwargs,
            extra=kwargs,
        )
        conflicts = set(model_inputs).intersection(generation_options)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Generation options cannot replace processor-emitted model "
                f"inputs: {names}.")
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            context = nullcontext()
        else:
            inference_mode = getattr(torch, "inference_mode", None)
            context = (inference_mode() if callable(inference_mode) else nullcontext())
        with context:
            generated = self.model.generate(
                **model_inputs,
                **generation_options,
            )
        generated_tokens = self._slice_generated_tokens(
            generated,
            self._prompt_length(model_inputs["input_ids"]),
        )
        return self._decode_output(
            generated_tokens,
            duration=materialized.duration,
            language=language,
        )

    @staticmethod
    def _audio_content(waveform: Any) -> dict[str, Any]:
        return {
            "type": "audio",
            "audio": waveform,
        }

    def _training_conversation(
        self,
        *,
        waveform: Any,
        transcription: str,
        language: str | None,
    ) -> list[dict[str, Any]]:
        del transcription, language
        return [{
            "role": "user",
            "content": [self._audio_content(waveform)],
        }]

    def _apply_training_template(
        self,
        conversations: list[list[dict[str, Any]]],
    ) -> Mapping[str, Any]:
        return self.transformers_processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            output_labels=True,
        )

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Create the checkpoint's own multimodal prompt and causal labels."""
        del phase
        if "input_ids" in inputs and "labels" in inputs:
            return dict(inputs)
        audio = inputs.get("audio")
        if audio is None:
            return dict(inputs)
        if self.transformers_processor is None:
            raise RuntimeError("Training input preparation requires load_for_training().")

        transcription = inputs.get(
            "text",
            inputs.get("transcription", inputs.get("transcript")),
        )
        if transcription is None:
            raise ValueError(
                "Raw multimodal ASR training batches require `text`, "
                "`transcription`, or `transcript`.")
        if isinstance(transcription, str):
            texts = [transcription]
        elif isinstance(transcription, (list, tuple)):
            texts = list(transcription)
        else:
            raise TypeError("ASR training transcriptions must be a string or a sequence "
                            "of strings.")
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("ASR training transcriptions must contain non-empty strings.")

        audio_values = self._as_batch(audio, expected_size=len(texts))
        audio_values = self._trim_audio_batch(
            audio_values,
            inputs.get("audio_lengths"),
        )
        rates = self._batch_scalar_values(
            inputs.get("sampling_rate"),
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
            ).waveform for value, rate in zip(audio_values, rates)
        ]
        languages = self._batch_scalar_values(
            inputs.get("language", self.config.training_language),
            batch_size=len(texts),
            name="language",
            broadcast=True,
        )
        for language in languages:
            if language is not None and (not isinstance(language, str) or not language.strip()):
                raise ValueError("ASR training languages must be non-empty strings or None.")

        conversations = [
            self._training_conversation(
                waveform=waveform,
                transcription=text.strip(),
                language=(language.strip() if isinstance(language, str) else None),
            ) for waveform, text, language in zip(waveforms, texts, languages)
        ]
        encoded = self._apply_training_template(conversations)
        if not isinstance(encoded, Mapping):
            raise TypeError("The multimodal ASR training template must return a mapping.")
        if "labels" not in encoded:
            raise TypeError(
                "The multimodal ASR processor did not emit native `labels` "
                "with `output_labels=True`.")
        return dict(encoded)


class Qwen3ASRForSpeechRecognition(MultimodalTransformersASRForSpeechRecognition):
    """Qwen3-ASR inference and native chat-template fine-tuning."""

    config_class = Qwen3ASRConfig
    default_model_name_or_path = "Qwen/Qwen3-ASR-0.6B-hf"
    expected_native_model_types = frozenset({"qwen3_asr"})
    backend_name = "transformers-qwen3-asr"
    _assistant_marker = "<|im_start|>assistant\n"

    @staticmethod
    def _canonical_language(language: str | None) -> str | None:
        if language is None:
            return None
        if not isinstance(language, str) or not language.strip():
            raise ValueError("`language` must be a non-empty string or None.")
        normalized = language.strip().casefold()
        by_name = {name.casefold(): name for name in _QWEN3_LANGUAGE_NAMES.values()}
        resolved = _QWEN3_LANGUAGE_NAMES.get(normalized, by_name.get(normalized))
        if resolved is None:
            supported_codes = ", ".join(sorted(_QWEN3_LANGUAGE_NAMES))
            raise ValueError(
                f"Unsupported Qwen3-ASR language {language!r}. Use a "
                f"supported language name or code: {supported_codes}.")
        return resolved

    @staticmethod
    def _token_rows(value: Any, *, name: str) -> list[list[int]]:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Qwen3-ASR training `{name}` must be a token sequence.")
        if value and isinstance(value[0], Integral):
            value = [value]

        rows = []
        for row in value:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(f"Qwen3-ASR training `{name}` must be rank two.")
            normalized = []
            for token_id in row:
                if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                    raise TypeError(f"Qwen3-ASR training `{name}` contains a non-integer value.")
                normalized.append(int(token_id))
            rows.append(normalized)
        if not rows:
            raise ValueError(f"Qwen3-ASR training `{name}` cannot be empty.")
        return rows

    def _assistant_marker_ids(self) -> list[int]:
        tokenizer = getattr(
            self.transformers_processor,
            "tokenizer",
            None,
        )
        if not callable(tokenizer):
            raise TypeError("The Qwen3-ASR processor has no callable tokenizer.")
        encoded = tokenizer(
            self._assistant_marker,
            add_special_tokens=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise TypeError("The Qwen3-ASR tokenizer did not encode the assistant marker.")
        rows = self._token_rows(
            encoded["input_ids"],
            name="assistant marker",
        )
        if len(rows) != 1 or not rows[0]:
            raise ValueError("The Qwen3-ASR tokenizer returned an invalid assistant marker.")
        return rows[0]

    @staticmethod
    def _completion_start(row: Sequence[int], marker: Sequence[int]) -> int:
        marker_length = len(marker)
        for index in range(len(row) - marker_length + 1):
            if list(row[index:index + marker_length]) == list(marker):
                return index + marker_length
        raise ValueError(
            "The Qwen3-ASR chat template did not emit its assistant marker. "
            "The checkpoint template is incompatible with native fine-tuning.")

    @staticmethod
    def _clone_token_ids(input_ids: Any) -> Any:
        clone = getattr(input_ids, "clone", None)
        if callable(clone):
            return clone()
        copy = getattr(input_ids, "copy", None)
        if callable(copy):
            return copy()
        return [list(row) for row in input_ids]

    @staticmethod
    def _mask_label_prefix(labels: Any, *, row: int, end: int) -> None:
        try:
            labels[row, :end] = -100
        except (IndexError, TypeError):
            labels[row][:end] = [-100] * end

    @staticmethod
    def _mask_label_padding(
        labels: Any,
        attention_mask: Any,
        attention_rows: Sequence[Sequence[int]],
    ) -> Any:
        not_attended = getattr(attention_mask, "ne", None)
        masked_fill = getattr(labels, "masked_fill", None)
        if callable(not_attended) and callable(masked_fill):
            return masked_fill(not_attended(1), -100)
        if hasattr(attention_mask, "shape"):
            try:
                labels[attention_mask != 1] = -100
                return labels
            except (IndexError, TypeError, ValueError):
                pass
        for row_index, row in enumerate(attention_rows):
            for column, attended in enumerate(row):
                if attended != 1:
                    try:
                        labels[row_index, column] = -100
                    except (IndexError, TypeError):
                        labels[row_index][column] = -100
        return labels

    def _causal_completion_labels(
        self,
        encoded: Mapping[str, Any],
    ) -> Any:
        """Build vocabulary labels without trusting Transformers 5.14's labels.

        Transformers 5.14.1 exposes ``output_labels=True`` for
        Qwen3-ASR, but returns multimodal token-type IDs (0/3) in the
        ``labels`` field. The model's causal loss instead requires
        vocabulary IDs. Locate the assistant boundary emitted by the
        checkpoint template, copy ``input_ids``, and supervise only its
        completion.
        """
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise TypeError(
                "Qwen3-ASR native fine-tuning requires `input_ids` and "
                "`attention_mask` from the processor.")
        input_rows = self._token_rows(input_ids, name="input_ids")
        attention_rows = self._token_rows(
            attention_mask,
            name="attention_mask",
        )
        if len(input_rows) != len(attention_rows) or any(len(tokens) != len(mask)
                                                         for tokens, mask in zip(input_rows, attention_rows)):
            raise ValueError(
                "Qwen3-ASR `input_ids` and `attention_mask` must have "
                "identical batch shapes.")

        marker = self._assistant_marker_ids()
        completion_starts = [self._completion_start(row, marker) for row in input_rows]
        for row, attention, start in zip(
                input_rows,
                attention_rows,
                completion_starts,
        ):
            if not any(attended == 1 for attended in attention[start:len(row)]):
                raise ValueError("Qwen3-ASR training produced an empty assistant target.")

        labels = self._clone_token_ids(input_ids)
        for row_index, start in enumerate(completion_starts):
            self._mask_label_prefix(
                labels,
                row=row_index,
                end=start,
            )
        return self._mask_label_padding(
            labels,
            attention_mask,
            attention_rows,
        )

    def _apply_transcription_request(
        self,
        *,
        waveform: Any,
        language: str | None,
        prompt: str | None,
        processor_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        language = self._canonical_language(language)

        conversation: list[dict[str, Any]] = []
        if prompt is not None:
            conversation.append({
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": prompt,
                }],
            })
        conversation.append({
            "role": "user",
            "content": [self._audio_content(waveform)],
        })

        template_options: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "processor_kwargs": dict(processor_kwargs),
        }
        if language is None:
            template_options["add_generation_prompt"] = True
        else:
            conversation.append({
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"language {language}<asr_text>",
                }],
            })
            template_options["continue_final_message"] = True

        return self.transformers_processor.apply_chat_template(
            [conversation],
            **template_options,
        )

    def _training_conversation(
        self,
        *,
        waveform: Any,
        transcription: str,
        language: str | None,
    ) -> list[dict[str, Any]]:
        language = self._canonical_language(language)
        if language is None:
            raise ValueError(
                "Qwen3-ASR training requires `language` in the batch or "
                "`training_language` in the configuration.")
        return [
            {
                "role": "user",
                "content": [self._audio_content(waveform)],
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": f"language {language}<asr_text>{transcription}",
                }],
            },
        ]

    def _apply_training_template(
        self,
        conversations: list[list[dict[str, Any]]],
    ) -> Mapping[str, Any]:
        encoded = self.transformers_processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("The Qwen3-ASR training template must return a mapping.")
        output = dict(encoded)
        output["labels"] = self._causal_completion_labels(encoded)
        return output

    def prepare_training_inputs(
        self,
        inputs: dict[str, Any],
        *,
        phase: str,
    ) -> dict[str, Any]:
        """Normalize cached and raw batches to completion-only causal
        labels."""
        if "input_ids" in inputs:
            if self.transformers_processor is None:
                raise RuntimeError("Training input preparation requires load_for_training().")
            output = dict(inputs)
            output["labels"] = self._causal_completion_labels(output)
            return output
        return super().prepare_training_inputs(
            inputs,
            phase=phase,
        )


class VibeVoiceASRForSpeechRecognition(MultimodalTransformersASRForSpeechRecognition):
    """VibeVoice-ASR inference and processor-native fine-tuning."""

    config_class = VibeVoiceASRConfig
    default_model_name_or_path = "microsoft/VibeVoice-ASR-HF"
    expected_native_model_types = frozenset({"vibevoice_asr"})
    backend_name = "transformers-vibevoice-asr"
    supports_native_timestamps = True

    def _request_options(
        self,
        *,
        language: str | None,
        prompt: str | None,
        processor_kwargs: Mapping[str, Any],
    ) -> dict[str, Any]:
        if language is not None:
            raise ValueError(
                "VibeVoice-ASR does not expose language forcing. Omit "
                "`language` and let the checkpoint identify it from audio.")
        return super()._request_options(
            language=None,
            prompt=prompt,
            processor_kwargs=processor_kwargs,
        )

    @staticmethod
    def _safe_timestamp(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        value = float(value)
        return value if isfinite(value) and value >= 0 else None

    def _decode_output(
        self,
        generated_tokens: Any,
        *,
        duration: float,
        language: str | None,
    ) -> ASROutput:
        decoded = self.transformers_processor.decode(
            generated_tokens,
            return_format="parsed",
        )
        if (isinstance(decoded, list) and len(decoded) == 1 and isinstance(decoded[0], (list, str))):
            decoded = decoded[0]
        if isinstance(decoded, str):
            return ASROutput(
                text=decoded,
                language=language,
                duration=duration,
                metadata={
                    "backend":
                    self.backend_name,
                    "native_model_type":
                    self._normalized_model_type(getattr(self.native_config, "model_type", None)),
                    "structured_output":
                    False,
                },
            )
        if not isinstance(decoded, Sequence):
            raise TypeError("VibeVoice-ASR parsed output must be a segment sequence or "
                            "fallback string.")

        segments = []
        for entry in decoded:
            if not isinstance(entry, Mapping):
                continue
            start = self._safe_timestamp(entry.get("Start", entry.get("start")))
            end = self._safe_timestamp(entry.get("End", entry.get("end")))
            if start is not None and end is not None and end < start:
                start = None
                end = None
            speaker_value = entry.get("Speaker", entry.get("speaker"))
            speaker = (
                str(speaker_value).strip()
                if speaker_value is not None and str(speaker_value).strip() else None)
            text = str(entry.get(
                "Content",
                entry.get("content", entry.get("text", "")),
            )).strip()
            metadata = {
                key: value
                for key, value in entry.items() if key not in {
                    "Content",
                    "content",
                    "text",
                    "Start",
                    "start",
                    "End",
                    "end",
                    "Speaker",
                    "speaker",
                }
            }
            segments.append(
                ASRSegment(
                    text=text,
                    start=start,
                    end=end,
                    language=language,
                    speaker=speaker,
                    metadata=metadata,
                ))
        text = " ".join(segment.text for segment in segments if segment.text).strip()
        return ASROutput(
            text=text,
            segments=tuple(segments),
            language=language,
            duration=duration,
            metadata={
                "backend": self.backend_name,
                "native_model_type":
                self._normalized_model_type(getattr(self.native_config, "model_type", None)),
                "structured_output": True,
            },
        )

    def _training_conversation(
        self,
        *,
        waveform: Any,
        transcription: str,
        language: str | None,
    ) -> list[dict[str, Any]]:
        if language is not None:
            raise ValueError(
                "VibeVoice-ASR training does not expose language "
                "conditioning. Remove `language` from the batch.")
        return [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": transcription,
                },
                self._audio_content(waveform),
            ],
        }]


__all__ = [
    "MultimodalTransformersASRForSpeechRecognition",
    "Qwen3ASRForSpeechRecognition",
    "VibeVoiceASRForSpeechRecognition",
]
