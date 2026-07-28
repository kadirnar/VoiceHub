"""Native Transformers integration for IBM Granite Speech ASR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from typing import Any

from voicehub.modeling_outputs import ASROutput
from voicehub.models import asr_transformers_multimodal as multimodal_asr
from voicehub.models.asr_granite_speech.configuration_asr_granite_speech import GraniteSpeechASRConfig


class GraniteSpeechForSpeechRecognition(multimodal_asr.MultimodalTransformersASRForSpeechRecognition):
    """Granite Speech inference and completion-only supervised fine-tuning.

    IBM's processor consumes a rendered text prompt and waveform
    separately; it does not expose the transcription-request helper used
    by other current multimodal ASR families. Training follows IBM's
    published collator: processor-owned prompt/audio inputs are
    concatenated with tokenized target text, while prompt and target-
    padding positions are masked with ``-100``.
    """

    config_class = GraniteSpeechASRConfig
    default_model_name_or_path = "ibm-granite/granite-speech-4.1-2b"
    expected_native_model_types = frozenset({"granite_speech"})
    backend_name = "transformers-granite-speech-asr"

    def _validate_processor_contract(self) -> None:
        processor = self.transformers_processor
        tokenizer = getattr(processor, "tokenizer", None)
        apply_template = getattr(tokenizer, "apply_chat_template", None)
        batch_decode = getattr(tokenizer, "batch_decode", None)
        if (not callable(processor) or not callable(tokenizer) or not callable(apply_template) or
                not callable(batch_decode)):
            raise TypeError(
                "Granite Speech requires a callable processor plus a "
                "callable tokenizer exposing `apply_chat_template()` and "
                "`batch_decode()`.")

    def _processor_sample_rate(self) -> int:
        audio_processor = getattr(
            self.transformers_processor,
            "audio_processor",
            None,
        )
        sample_rate = getattr(
            audio_processor,
            "sampling_rate",
            self.config.sample_rate,
        )
        if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Real) or
                not isfinite(float(sample_rate)) or float(sample_rate) <= 0 or
                not float(sample_rate).is_integer()):
            raise ValueError("The Granite Speech processor reported an invalid sampling rate.")
        return int(sample_rate)

    def _instruction_prompt(self, prompt: str | None) -> str:
        instruction = prompt or self.config.transcription_prompt
        instruction = instruction.strip()
        if "<|audio|>" not in instruction:
            instruction = f"<|audio|>{instruction}"
        return instruction

    def _render_instruction(self, prompt: str | None) -> str:
        tokenizer = self.transformers_processor.tokenizer
        rendered = tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": self._instruction_prompt(prompt),
            }],
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(rendered, str) or not rendered:
            raise TypeError(
                "The Granite Speech tokenizer must render the transcription "
                "instruction as a non-empty string.")
        return rendered

    def _hotword_prompt(
        self,
        prompt: str | None,
        hotwords: str | tuple[str, ...] | list[str] | None,
    ) -> str | None:
        if hotwords is None:
            return prompt
        values = [hotwords] if isinstance(hotwords, str) else list(hotwords)
        words = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Granite Speech hotwords must be non-empty strings.")
            words.append(value.strip())
        if not words:
            raise ValueError("Granite Speech hotwords cannot be empty.")
        instruction = self._instruction_prompt(prompt)
        return f"{instruction.rstrip()} Keywords: {', '.join(words)}"

    @staticmethod
    def _reject_owned_processor_options(options: Mapping[str, Any]) -> None:
        reserved = {
            "padding",
            "padding_side",
            "return_tensors",
        }
        conflicts = reserved.intersection(options)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                "Granite Speech `processor_kwargs` cannot replace "
                f"provider-owned option(s): {names}.")

    def _apply_transcription_request(
        self,
        *,
        waveform: Any,
        language: str | None,
        prompt: str | None,
        processor_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if language is not None:
            raise ValueError(
                "Granite Speech does not expose a language-ID forcing "
                "argument. Put language guidance in `prompt` instead.")
        self._reject_owned_processor_options(processor_kwargs)
        return self.transformers_processor(
            self._render_instruction(prompt),
            waveform,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            **processor_kwargs,
        )

    def _decode_output(
        self,
        generated_tokens: Any,
        *,
        duration: float,
        language: str | None,
    ) -> ASROutput:
        decoded = self.transformers_processor.tokenizer.batch_decode(
            generated_tokens,
            add_special_tokens=False,
            skip_special_tokens=True,
        )
        if isinstance(decoded, str):
            text = decoded.strip()
        elif (isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes)) and len(decoded) == 1):
            text = str(decoded[0]).strip()
        else:
            raise TypeError(
                "The Granite Speech tokenizer must decode a single "
                "transcription for a single audio input.")
        return ASROutput(
            text=text,
            language=language,
            duration=duration,
            metadata={
                "backend": self.backend_name,
                "native_model_type":
                self._normalized_model_type(getattr(self.native_config, "model_type", None)),
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
                "Granite Speech fine-tuning is prompt-conditioned. Remove "
                "`language` from the batch or express it in the configured "
                "`transcription_prompt`.")
        return [
            {
                "role": "user",
                "content": [self._audio_content(waveform)],
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": transcription,
                }],
            },
        ]

    @staticmethod
    def _training_example(conversation: Sequence[Mapping[str, Any]], ) -> tuple[Any, str]:
        if len(conversation) != 2:
            raise ValueError(
                "A Granite Speech training conversation must contain one "
                "user audio turn and one assistant transcript.")
        user_content = conversation[0].get("content")
        assistant_content = conversation[1].get("content")
        if (not isinstance(user_content, Sequence) or isinstance(user_content, (str, bytes)) or
                not isinstance(assistant_content, Sequence) or isinstance(assistant_content, (str, bytes))):
            raise TypeError("Granite Speech training turns must contain structured content.")
        audio_entries = [
            value for value in user_content if isinstance(value, Mapping) and value.get("type") == "audio"
        ]
        text_entries = [
            value for value in assistant_content if isinstance(value, Mapping) and value.get("type") == "text"
        ]
        if len(audio_entries) != 1 or len(text_entries) != 1:
            raise ValueError(
                "Granite Speech training requires exactly one waveform and "
                "one transcript per example.")
        transcript = text_entries[0].get("text")
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("Granite Speech training transcripts must be non-empty strings.")
        return audio_entries[0].get("audio"), transcript.strip()

    @staticmethod
    def _token_rows(value: Any, *, name: str) -> list[list[int]]:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"Granite Speech training `{name}` must be a token sequence.")
        if value and isinstance(value[0], Integral):
            value = [value]
        rows = []
        for row in value:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise TypeError(f"Granite Speech training `{name}` must be rank two.")
            normalized = []
            for token_id in row:
                if isinstance(token_id, bool) or not isinstance(token_id, Integral):
                    raise TypeError(f"Granite Speech training `{name}` contains a "
                                    "non-integer value.")
                normalized.append(int(token_id))
            rows.append(normalized)
        if not rows:
            raise ValueError(f"Granite Speech training `{name}` cannot be empty.")
        return rows

    @classmethod
    def _combine_list_tokens(
        cls,
        *,
        prompt_ids: Any,
        prompt_mask: Any,
        target_ids: Any,
        target_mask: Any,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
        prompt_rows = cls._token_rows(prompt_ids, name="input_ids")
        prompt_mask_rows = cls._token_rows(
            prompt_mask,
            name="attention_mask",
        )
        target_rows = cls._token_rows(target_ids, name="target input_ids")
        target_mask_rows = cls._token_rows(
            target_mask,
            name="target attention_mask",
        )
        batch_size = len(prompt_rows)
        if not all(len(rows) == batch_size for rows in (
                prompt_mask_rows,
                target_rows,
                target_mask_rows,
        )):
            raise ValueError("Granite Speech prompt and target tensors must share a batch size.")

        combined_ids = []
        combined_mask = []
        labels = []
        for prompt_row, prompt_attention, target_row, target_attention in zip(
                prompt_rows,
                prompt_mask_rows,
                target_rows,
                target_mask_rows,
        ):
            if len(prompt_row) != len(prompt_attention):
                raise ValueError(
                    "Granite Speech prompt IDs and attention mask must have "
                    "identical shapes.")
            if len(target_row) != len(target_attention):
                raise ValueError(
                    "Granite Speech target IDs and attention mask must have "
                    "identical shapes.")
            if not any(value == 1 for value in target_attention):
                raise ValueError("Granite Speech training produced an empty transcript target.")
            combined_ids.append(prompt_row + target_row)
            combined_mask.append(prompt_attention + target_attention)
            labels.append(([-100] * len(prompt_row)) + [
                token_id if attended == 1 else -100
                for token_id, attended in zip(target_row, target_attention)
            ])
        return combined_ids, combined_mask, labels

    @staticmethod
    def _combine_tensor_tokens(
        *,
        prompt_ids: Any,
        prompt_mask: Any,
        target_ids: Any,
        target_mask: Any,
    ) -> tuple[Any, Any, Any] | None:
        try:
            torch = import_module("torch")
        except ModuleNotFoundError:
            return None
        is_tensor = getattr(torch, "is_tensor", None)
        concatenate = getattr(torch, "cat", None)
        full_like = getattr(torch, "full_like", None)
        if (not callable(is_tensor) or not callable(concatenate) or not callable(full_like) or
                not all(is_tensor(value) for value in (
                    prompt_ids,
                    prompt_mask,
                    target_ids,
                    target_mask,
                ))):
            return None
        if (prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape or target_ids.ndim != 2 or
                target_mask.shape != target_ids.shape):
            raise ValueError(
                "Granite Speech prompt and target IDs/masks must be rank-two "
                "tensors with matching shapes.")
        if prompt_ids.shape[0] != target_ids.shape[0]:
            raise ValueError("Granite Speech prompt and target tensors must share a batch size.")
        if not bool(target_mask.ne(0).any(dim=-1).all()):
            raise ValueError("Granite Speech training produced an empty transcript target.")
        target_labels = target_ids.clone().masked_fill(target_mask.ne(1), -100)
        labels = concatenate(
            (
                full_like(prompt_ids, -100),
                target_labels,
            ),
            dim=-1,
        )
        return (
            concatenate((prompt_ids, target_ids), dim=-1),
            concatenate((prompt_mask, target_mask), dim=-1),
            labels,
        )

    def _apply_training_template(
        self,
        conversations: list[list[dict[str, Any]]],
    ) -> Mapping[str, Any]:
        examples = [self._training_example(conversation) for conversation in conversations]
        waveforms = [waveform for waveform, _text in examples]
        texts = [text for _waveform, text in examples]
        prompts = [self._render_instruction(None) for _ in examples]

        processed = self.transformers_processor(
            prompts,
            waveforms,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )
        if not isinstance(processed, Mapping):
            raise TypeError("The Granite Speech processor must return a mapping for training.")
        prompt_ids = processed.get("input_ids")
        prompt_mask = processed.get("attention_mask")
        if prompt_ids is None or prompt_mask is None:
            raise TypeError(
                "The Granite Speech processor must emit `input_ids` and "
                "`attention_mask` for training.")

        tokenizer = self.transformers_processor.tokenizer
        eos_token = getattr(tokenizer, "eos_token", None)
        if not isinstance(eos_token, str) or not eos_token:
            raise ValueError("The Granite Speech tokenizer must expose a non-empty EOS token.")
        targets = tokenizer(
            [f"{text}{eos_token}" for text in texts],
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        if not isinstance(targets, Mapping):
            raise TypeError("The Granite Speech tokenizer must return a mapping for targets.")
        target_ids = targets.get("input_ids")
        target_mask = targets.get("attention_mask")
        if target_ids is None or target_mask is None:
            raise TypeError(
                "The Granite Speech tokenizer must emit target `input_ids` "
                "and `attention_mask`.")

        combined = self._combine_tensor_tokens(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            target_ids=target_ids,
            target_mask=target_mask,
        )
        if combined is None:
            combined = self._combine_list_tokens(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                target_ids=target_ids,
                target_mask=target_mask,
            )
        input_ids, attention_mask, labels = combined
        batch = dict(processed)
        batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
        return batch


__all__ = ["GraniteSpeechForSpeechRecognition"]
