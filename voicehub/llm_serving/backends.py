"""Request adapters for vLLM and SGLang LLM-based speech synthesis."""

from __future__ import annotations

import base64
import binascii
import math
import mimetypes
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from voicehub.errors import LLMBackendCompatibilityError, LLMBackendRequestError
from voicehub.llm_serving.configuration import LLMBackend, LLMBackendConfig, LLMBackendTransport
from voicehub.llm_serving.http import HTTPBackendClient
from voicehub.llm_serving.protocol import TokenGenerationRequest, TokenGenerationResult
from voicehub.llm_serving.support import LLMBackendSupport, get_llm_backend_support
from voicehub.models._shared import finish_audio_output
from voicehub.processing.waveform import decode_pcm_wave


def _usage_count(usage: Any, name: str) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    value = usage.get(name)
    return (int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None)


class LLMServingClient:
    """One configured vLLM or SGLang HTTP client."""

    _REFERENCE_LIMIT = 64 * 1024 * 1024

    def __init__(self, config: LLMBackendConfig):
        if config.backend is LLMBackend.NATIVE:
            raise ValueError("LLMServingClient requires an external backend.")
        self.config = config
        self.http = HTTPBackendClient(config)

    def generate_tokens(
        self,
        request: TokenGenerationRequest,
    ) -> TokenGenerationResult:
        """Generate one suffix through a tokenizer-free engine API."""
        if self.config.backend is LLMBackend.VLLM:
            return self._generate_vllm_tokens(request)
        if self.config.backend is LLMBackend.SGLANG:
            return self._generate_sglang_tokens(request)
        raise AssertionError(f"Unhandled backend {self.config.backend!r}.")

    def _generate_vllm_tokens(
        self,
        request: TokenGenerationRequest,
    ) -> TokenGenerationResult:
        model = self.config.model
        if model is None:
            raise LLMBackendRequestError("vLLM token generation requires `LLMBackendConfig.model`.")
        payload = {
            **self.config.request_extra_body(),
            "model": model,
            "prompt": list(request.prompt_token_ids),
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
            "return_token_ids": True,
            "skip_special_tokens": False,
        }
        for name, value in (
            ("top_p", request.top_p),
            ("top_k", request.top_k),
            ("min_p", request.min_p),
            ("seed", request.seed),
        ):
            if value is not None:
                payload[name] = value
        if request.stop_token_ids:
            payload["stop_token_ids"] = list(request.stop_token_ids)
        document = self.http.post_json_document(
            "/v1/completions",
            payload,
        )
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise LLMBackendRequestError("vLLM completion response has no first choice.")
        choice = choices[0]
        token_ids = choice.get("token_ids")
        if token_ids is None:
            token_ids = choice.get("output_token_ids")
        if not isinstance(token_ids, list):
            raise LLMBackendRequestError(
                "vLLM did not return token IDs. Start a compatible server and "
                "enable the completion protocol's `return_token_ids` field.")
        usage = document.get("usage")
        return TokenGenerationResult(
            token_ids=tuple(token_ids),
            finish_reason=(str(choice["finish_reason"]) if choice.get("finish_reason") is not None else None),
            prompt_tokens=_usage_count(usage, "prompt_tokens"),
            completion_tokens=_usage_count(usage, "completion_tokens"),
        )

    def _generate_sglang_tokens(
        self,
        request: TokenGenerationRequest,
    ) -> TokenGenerationResult:
        if request.repetition_penalty > 2.0:
            raise LLMBackendCompatibilityError(
                "SGLang token generation requires `repetition_penalty` to "
                "be at most 2.0.")
        sampling = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "repetition_penalty": request.repetition_penalty,
        }
        for name, value in (
            ("top_p", request.top_p),
            ("top_k", request.top_k),
            ("min_p", request.min_p),
        ):
            if value is not None:
                sampling[name] = (-1 if name == "top_k" and value == 0 else value)
        if request.stop_token_ids:
            sampling["stop_token_ids"] = list(request.stop_token_ids)
        if request.seed is not None:
            sampling["sampling_seed"] = request.seed
        payload = {
            **self.config.request_extra_body(),
            "input_ids": list(request.prompt_token_ids),
            "sampling_params": sampling,
            "stream": False,
        }
        document = self.http.post_json_document(
            "/generate",
            payload,
        )
        token_ids = document.get("output_ids")
        if token_ids is None:
            token_ids = document.get("token_ids")
        meta = document.get("meta_info")
        if token_ids is None and isinstance(meta, Mapping):
            token_ids = meta.get("output_token_ids")
        if not isinstance(token_ids, list):
            raise LLMBackendRequestError(
                "SGLang did not return `output_ids` for the token-in/token-out "
                "request.")
        finish_reason = None
        if isinstance(meta, Mapping):
            reason = meta.get("finish_reason")
            if isinstance(reason, Mapping):
                reason = reason.get("type")
            if reason is not None:
                finish_reason = str(reason)
        return TokenGenerationResult(
            token_ids=tuple(token_ids),
            finish_reason=finish_reason,
            prompt_tokens=_usage_count(meta, "prompt_tokens"),
            completion_tokens=_usage_count(meta, "completion_tokens"),
        )

    @staticmethod
    def _audio_source(
        value: Any,
        *,
        option_name: str,
        max_bytes: int,
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, Path):
            path = value.expanduser()
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                raise ValueError(f"`{option_name}` must be non-empty.")
            parsed = urlsplit(stripped)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return stripped
            if stripped.startswith("data:audio/"):
                header, separator, encoded = stripped.partition(",")
                if not separator or not header.lower().endswith(";base64"):
                    raise ValueError(f"`{option_name}` must be a base64-encoded audio data URL.")
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ValueError(f"`{option_name}` contains invalid base64 audio data.") from error
                if not decoded:
                    raise ValueError(f"`{option_name}` cannot be empty.")
                if len(decoded) > max_bytes:
                    raise ValueError(
                        f"`{option_name}` is {len(decoded)} bytes; the "
                        f"remote-reference limit is {max_bytes}.")
                return stripped
            compact = "".join(stripped.split())
            if compact.startswith("UklGR"):
                try:
                    decoded = base64.b64decode(compact, validate=True)
                except (ValueError, binascii.Error) as error:
                    raise ValueError(f"`{option_name}` contains invalid base64 audio data.") from error
                if len(decoded) > max_bytes:
                    raise ValueError(
                        f"`{option_name}` is {len(decoded)} bytes; the "
                        f"remote-reference limit is {max_bytes}.")
                return "data:audio/wav;base64," + compact
            path = Path(stripped).expanduser()
        else:
            raise LLMBackendCompatibilityError(
                f"External speech serving cannot serialize `{option_name}` "
                f"from {type(value).__name__}. Pass a local path, HTTP(S) URL, "
                "or audio data URL.")
        if not path.is_file():
            raise FileNotFoundError(f"`{option_name}` was not found: {path}.")
        size = path.stat().st_size
        if size <= 0:
            raise ValueError(f"`{option_name}` cannot be empty.")
        if size > max_bytes:
            raise ValueError(
                f"`{option_name}` is {size} bytes; the remote-reference "
                f"limit is {max_bytes}.")
        media_type = (
            "audio/x-wav"
            if path.suffix.lower() == ".wav" else mimetypes.guess_type(path.name)[0] or "audio/wav")
        if not media_type.startswith("audio/"):
            media_type = "audio/wav"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    @staticmethod
    def _task_type(
        support: LLMBackendSupport,
        inputs: Mapping[str, Any],
        *,
        has_reference: bool,
    ) -> str | None:
        explicit = inputs.get("task_type")
        mode = inputs.get("mode", explicit)
        if mode is None:
            return (
                support.task_type_with_reference if has_reference else support.task_type_without_reference)
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError("`mode`/`task_type` must be a non-empty string.")
        normalized = mode.strip().lower().replace("-", "_")
        if normalized == "auto":
            task_type = (
                support.task_type_with_reference if has_reference else support.task_type_without_reference)
            if task_type is not None:
                return task_type
        try:
            return dict(support.task_type_aliases)[normalized]
        except KeyError as error:
            raise ValueError(f"Unsupported external TTS task type {mode!r}.") from error

    def _speech_payload(
        self,
        model_type: str,
        inputs: Mapping[str, Any],
    ) -> dict[str, Any]:
        support, transport = get_llm_backend_support(
            model_type,
            self.config.backend,
            transport=self.config.transport,
        )
        if transport is not LLMBackendTransport.SPEECH:
            raise LLMBackendCompatibilityError(
                f"{model_type!r} is configured for token transport, not the "
                "complete speech endpoint.")
        unknown = sorted(set(inputs) - {"text"} - set(support.speech_input_options))
        if unknown:
            recognized = ", ".join(support.speech_input_options)
            invalid = ", ".join(unknown)
            raise ValueError(
                f"Unsupported external speech option(s): {invalid}. "
                f"Recognized options: {recognized}.")
        text = inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("External speech generation requires non-empty `text`.")
        if inputs.get("speaker_embedding") is not None:
            raise LLMBackendCompatibilityError(
                "VoiceHub does not serialize speaker-embedding tensors to "
                "external speech APIs. Pass `voice` or reference audio.")
        if inputs.get("prompt_speech_tokens") is not None or inputs.get("prompt_features") is not None:
            raise LLMBackendCompatibilityError(
                "Precomputed CosyVoice prompt tensors are native-runtime "
                "inputs. External speech APIs accept reference audio instead.")
        reference_value = None
        reference_name = "speaker_audio_path"
        for name in (
                "speaker_audio_path",
                "prompt_audio_path",
                "ref_audio",
                "reference_audio",
                "speaker_audio",
        ):
            if inputs.get(name) is not None:
                if reference_value is not None:
                    raise ValueError("Pass only one reference-audio input to an external "
                                     "speech backend.")
                reference_value = inputs[name]
                reference_name = name
        ref_audio = self._audio_source(
            reference_value,
            option_name=reference_name,
            max_bytes=self._REFERENCE_LIMIT,
        )
        ref_text = inputs.get("reference_text", inputs.get("ref_text"))
        if ref_text is not None and (not isinstance(ref_text, str) or not ref_text.strip()):
            raise ValueError("`reference_text` must be a non-empty string or None.")
        if ref_text is not None and ref_audio is None:
            raise ValueError("`reference_text` requires reference audio.")
        instruction = None
        for name in ("instructions", "instruct", "instruction"):
            if inputs.get(name) is not None:
                if instruction is not None:
                    raise ValueError("Pass only one of `instructions`, `instruct`, or "
                                     "`instruction`.")
                instruction = inputs[name]
        if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
            raise ValueError("Speech instructions must be a non-empty string or None.")
        voice = inputs.get("voice", inputs.get("speaker"))
        if voice is not None and (not isinstance(voice, str) or not voice.strip()):
            raise ValueError("`voice`/`speaker` must be a non-empty string or None.")
        payload = {
            **self.config.request_extra_body(),
            "input": text,
            "response_format": "wav",
            "stream": False,
        }
        if voice is not None:
            payload["voice"] = voice.strip()
        model = self.config.model
        if model is not None:
            payload["model"] = model
        if ref_audio is not None:
            if support.reference_format == "references":
                reference = {
                    "audio_path": ref_audio,
                }
                if ref_text is not None:
                    reference["text"] = ref_text.strip()
                payload["references"] = [reference]
            else:
                payload["ref_audio"] = ref_audio
                if ref_text is not None:
                    payload["ref_text"] = ref_text.strip()
        task_type = self._task_type(
            support,
            inputs,
            has_reference=ref_audio is not None,
        )
        if task_type is not None:
            payload["task_type"] = task_type
        if instruction is not None:
            payload["instructions"] = instruction.strip()
        for name in support.speech_direct_options:
            value = inputs.get(name)
            if value is not None:
                payload[name] = value
        for name in ("duration_tokens", "max_new_tokens", "token_count"):
            value = payload.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Integral) or
                                      value <= 0):
                raise ValueError(f"`{name}` must be a positive integer.")
            if value is not None:
                payload[name] = int(value)
        initial_frames = payload.get("initial_codec_chunk_frames")
        if initial_frames is not None:
            if (isinstance(initial_frames, bool) or not isinstance(initial_frames, Integral) or
                    initial_frames < 0):
                raise ValueError("`initial_codec_chunk_frames` must be a non-negative "
                                 "integer.")
            payload["initial_codec_chunk_frames"] = int(initial_frames)
        for name in ("temperature", "top_p", "repetition_penalty"):
            value = payload.get(name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"`{name}` must be a real number.")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"`{name}` must be finite.")
            if name == "temperature" and value < 0:
                raise ValueError("`temperature` must be non-negative.")
            if name == "top_p" and not 0 < value <= 1:
                raise ValueError("`top_p` must be in the interval (0, 1].")
            if name == "repetition_penalty" and value <= 0:
                raise ValueError("`repetition_penalty` must be greater than zero.")
            payload[name] = value
        top_k = payload.get("top_k")
        if top_k is not None:
            if (isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0):
                raise ValueError("`top_k` must be a non-negative integer.")
            payload["top_k"] = int(top_k)
        speed = payload.get("speed")
        if speed is not None:
            if isinstance(speed, bool) or not isinstance(speed, Real):
                raise TypeError("`speed` must be a real number.")
            speed = float(speed)
            if not math.isfinite(speed) or not 0.25 <= speed <= 4.0:
                raise ValueError("`speed` must be finite and in the interval [0.25, 4.0].")
            payload["speed"] = speed
        for name in ("non_streaming_mode", "x_vector_only_mode"):
            value = payload.get(name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"`{name}` must be a boolean.")
        seed = payload.get("seed")
        if seed is not None:
            if (isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**63 - 1):
                raise ValueError(
                    "`seed` must be a non-negative signed 64-bit integer for "
                    "external speech serving.")
            payload["seed"] = int(seed)
        unsupported_by_backend = {"stage_params"}
        if self.config.backend is LLMBackend.VLLM:
            unsupported_by_backend.update({
                "duration_tokens",
                "repetition_penalty",
                "token_count",
            })
            configured_extra_params = payload.get("extra_params", {})
            if not isinstance(configured_extra_params, Mapping):
                raise LLMBackendCompatibilityError("vLLM speech `extra_params` must be a JSON object.")
            extra_params = dict(configured_extra_params)
            for name in ("temperature", "top_p", "top_k"):
                if name in payload:
                    extra_params[name] = payload.pop(name)
            if extra_params:
                payload["extra_params"] = extra_params
        else:
            if payload.get("top_k") == 0:
                payload["top_k"] = -1
            if payload.get("repetition_penalty", 1.0) > 2.0:
                raise LLMBackendCompatibilityError(
                    "SGLang speech generation requires "
                    "`repetition_penalty` to be at most 2.0.")
            unsupported_by_backend.add("non_streaming_mode")
        for name in unsupported_by_backend:
            if name in payload:
                raise LLMBackendCompatibilityError(
                    f"{self.config.backend.value} speech serving does not "
                    f"support VoiceHub option `{name}`.")
        native_only = set(support.speech_native_only_options)
        for name in support.speech_string_options:
            value = inputs.get(name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"`{name}` must be a non-empty string.")
            payload[name] = value.strip()
        supplied_native_only = sorted(name for name in native_only if inputs.get(name) is not None)
        if supplied_native_only:
            raise LLMBackendCompatibilityError(
                f"{self.config.backend.value} speech serving cannot preserve "
                "native-only option(s): " + ", ".join(supplied_native_only) +
                ". Put a documented server field in "
                "`LLMBackendConfig.extra_body` when the deployed engine "
                "supports an equivalent.")
        return payload

    def synthesize(
        self,
        model_type: str,
        inputs: Mapping[str, Any],
        *,
        default_sample_rate: int,
    ):
        """Run one complete Omni speech pipeline and return ``TTSOutput``."""
        payload = self._speech_payload(
            model_type,
            inputs,
        )
        response = self.http.post_json(
            "/v1/audio/speech",
            payload,
            accept="audio/wav, audio/pcm",
        )
        content_type = (response.header("content-type", "") or "").split(";", 1)[0].lower()
        if content_type == "application/json" or response.body.lstrip().startswith(b"{"):
            raise LLMBackendRequestError(
                f"{self.config.backend.value} returned JSON instead of audio "
                "for /v1/audio/speech.")
        if response.body[:4] == b"RIFF":
            audio, sample_rate = decode_pcm_wave(response.body)
        elif content_type in {"audio/pcm", "audio/l16", "application/octet-stream"}:
            audio, sample_rate = self._decode_raw_pcm(
                response.body,
                response.headers,
                default_sample_rate=default_sample_rate,
            )
        else:
            raise LLMBackendRequestError(
                "VoiceHub requested PCM WAVE audio, but the backend returned "
                f"{content_type or 'an unknown content type'!r}.")
        output_file = inputs.get("output_file")
        return finish_audio_output(
            audio,
            sample_rate,
            output_file=output_file,
            metadata={
                "backend": self.config.backend.value,
                "engine_transport": "speech",
                "model_type": model_type,
                "server_model": self.config.model,
            },
        )

    @staticmethod
    def _decode_raw_pcm(
        payload: bytes,
        headers: Mapping[str, str],
        *,
        default_sample_rate: int,
    ):
        if not payload:
            raise LLMBackendRequestError("The speech backend returned an empty PCM payload.")
        try:
            sample_rate = int(headers.get("x-sample-rate", default_sample_rate))
            channels = int(headers.get("x-channels", "1"))
            bit_depth = int(headers.get("x-bit-depth", "16"))
        except ValueError as error:
            raise LLMBackendRequestError("The speech backend returned invalid PCM metadata.") from error
        if sample_rate <= 0 or channels <= 0:
            raise LLMBackendRequestError("The speech backend returned non-positive PCM metadata.")
        if bit_depth != 16:
            raise LLMBackendRequestError(f"Only signed 16-bit PCM is supported, received {bit_depth}-bit.")
        if len(payload) % (2 * channels):
            raise LLMBackendRequestError("The raw PCM payload is not aligned to complete frames.")
        import torch

        samples = torch.frombuffer(
            memoryview(bytearray(payload)),
            dtype=torch.int16,
        ).clone().float().div_(32768.0)
        if channels > 1:
            samples = samples.view(-1, channels).transpose(0, 1).mean(dim=0)
        return samples, sample_rate


class RemoteCausalLMProxy:
    """Drop-in ``generate`` surface backed by a remote flat-token engine."""

    def __init__(self, client: LLMServingClient, *, model_type: str):
        self.client = client
        self.model_type = model_type

    def eval(self):
        return self

    def generate(
        self,
        *,
        input_ids,
        attention_mask=None,
        generation_config,
        **kwargs,
    ):
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise LLMBackendCompatibilityError(
                "Remote causal-LM generation received unsupported option(s): "
                f"{unknown}.")
        if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
            raise LLMBackendCompatibilityError(
                "Remote TTS token generation currently requires one rank-2 "
                "prompt with batch size 1.")
        row = input_ids[0]
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("`attention_mask` must match `input_ids`.")
            mask = attention_mask[0].detach().to(device="cpu").bool().tolist()
            ids = row.detach().to(device="cpu").tolist()
            prompt_ids = [int(token) for token, keep in zip(ids, mask) if keep]
        else:
            prompt_ids = [int(token) for token in row.detach().to(device="cpu").tolist()]
        result = self.client.generate_tokens(
            TokenGenerationRequest(
                prompt_token_ids=prompt_ids,
                max_new_tokens=generation_config.max_new_tokens,
                temperature=generation_config.temperature,
                top_p=generation_config.top_p,
                top_k=generation_config.top_k,
                min_p=generation_config.min_p,
                repetition_penalty=generation_config.repetition_penalty,
                stop_token_ids=generation_config.eos_token_ids,
                seed=generation_config.seed,
            ))
        completion = input_ids.new_tensor([result.token_ids])
        import torch

        sequences = torch.cat(
            (input_ids, completion),
            dim=-1,
        )
        from voicehub.generation import GenerationOutput

        return GenerationOutput(
            sequences=sequences,
            generated_lengths=input_ids.new_tensor(
                [len(result.token_ids)],
                dtype=torch.long,
            ),
            finished=input_ids.new_tensor(
                [result.finish_reason is not None],
                dtype=torch.bool,
            ),
            cache=None,
        )

    def save_pretrained(self, *_args, **_kwargs):
        raise RuntimeError(
            "An external token backend owns the language-model weights. "
            "Create a native VoiceHub wrapper before exporting a complete "
            "pretrained model.")


__all__ = [
    "LLMServingClient",
    "RemoteCausalLMProxy",
    "TokenGenerationRequest",
    "TokenGenerationResult",
]
