"""Runtime-only configuration for external LLM-TTS serving engines."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class LLMBackend(str, Enum):
    """Language-model serving engines understood by VoiceHub."""

    NATIVE = "native"
    VLLM = "vllm"
    SGLANG = "sglang"

    @classmethod
    def coerce(cls, value: str | LLMBackend) -> LLMBackend:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`backend` must be a string or LLMBackend.")
        normalized = value.strip().lower().replace("-", "").replace("_", "")
        aliases = {
            "native": cls.NATIVE,
            "voicehub": cls.NATIVE,
            "vllm": cls.VLLM,
            "vllmomni": cls.VLLM,
            "sglang": cls.SGLANG,
            "sglangomni": cls.SGLANG,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"Unknown LLM backend {value!r}. Choose one of: {choices}.") from error


class LLMBackendTransport(str, Enum):
    """Protocol used between a VoiceHub wrapper and an engine server."""

    AUTO = "auto"
    TOKENS = "tokens"
    SPEECH = "speech"

    @classmethod
    def coerce(
        cls,
        value: str | LLMBackendTransport,
    ) -> LLMBackendTransport:
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError("`transport` must be a string or LLMBackendTransport.")
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "auto": cls.AUTO,
            "token": cls.TOKENS,
            "tokens": cls.TOKENS,
            "token_ids": cls.TOKENS,
            "speech": cls.SPEECH,
            "audio": cls.SPEECH,
            "omni": cls.SPEECH,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown LLM backend transport {value!r}. Choose one of: "
                f"{choices}.") from error


_FORBIDDEN_HEADERS = frozenset({
    "connection",
    "content-length",
    "host",
    "transfer-encoding",
})


def _runtime_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
    string_values: bool,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping or None.")
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"`{name}` keys must be non-empty strings.")
        clean_key = key.strip()
        if "\r" in clean_key or "\n" in clean_key:
            raise ValueError(f"`{name}` keys cannot contain newlines.")
        if string_values:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"`{name}` values must be non-empty strings.")
            if "\r" in item or "\n" in item:
                raise ValueError(f"`{name}` values cannot contain newlines.")
            if clean_key.lower() in _FORBIDDEN_HEADERS:
                raise ValueError(f"`{name}` cannot override the reserved HTTP header "
                                 f"{clean_key!r}.")
            normalized[clean_key] = item.strip()
        else:
            normalized[clean_key] = item
    if not string_values:
        try:
            json.dumps(
                normalized,
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"`{name}` must contain finite JSON-serializable values.") from error
        normalized = {key: _freeze_json(item) for key, item in normalized.items()}
    return MappingProxyType(normalized)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Nested JSON object keys must be strings.")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalize_endpoint(value: str | None, *, required: bool) -> str | None:
    if value is None:
        if required:
            raise ValueError("External LLM backends require an HTTP(S) `endpoint`.")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("`endpoint` must be a non-empty HTTP(S) URL or None.")
    parts = urlsplit(value.strip())
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("`endpoint` must be an absolute HTTP(S) URL.")
    if parts.username is not None or parts.password is not None:
        raise ValueError("`endpoint` cannot contain credentials; use `api_key` or "
                         "runtime headers.")
    if parts.query or parts.fragment:
        raise ValueError("`endpoint` cannot contain a query string or fragment.")
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path.rstrip("/"),
        "",
        "",
    ))


@dataclass(frozen=True, slots=True)
class LLMBackendConfig:
    """Connection settings for a separately managed vLLM/SGLang server.

    Credentials are runtime-only. ``repr`` and :meth:`to_dict` never
    expose their values, and this object is deliberately not stored in
    model ``config.json`` files.
    """

    backend: LLMBackend | str
    endpoint: str | None = None
    transport: LLMBackendTransport | str = LLMBackendTransport.AUTO
    model: str | None = None
    api_key: str | None = field(default=None, repr=False, compare=False)
    timeout: float = 300.0
    headers: Mapping[str, str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    extra_body: Mapping[str, Any] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    max_response_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        backend = LLMBackend.coerce(self.backend)
        transport = LLMBackendTransport.coerce(self.transport)
        endpoint = _normalize_endpoint(
            self.endpoint,
            required=backend is not LLMBackend.NATIVE,
        )
        if backend is LLMBackend.NATIVE:
            if transport is not LLMBackendTransport.AUTO:
                raise ValueError("The native backend does not use an external transport.")
            if any(value is not None for value in (
                    endpoint,
                    self.api_key,
                    self.headers,
                    self.extra_body,
                    self.model,
            )):
                raise ValueError("The native backend does not accept connection settings.")
        model = self.model
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ValueError("`model` must be a non-empty server model ID or None.")
            model = model.strip()
        api_key = self.api_key
        if api_key is not None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ValueError("`api_key` must be a non-empty string or None.")
            if "\r" in api_key or "\n" in api_key:
                raise ValueError("`api_key` cannot contain newlines.")
            api_key = api_key.strip()
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise TypeError("`timeout` must be a real number.")
        timeout = float(self.timeout)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("`timeout` must be finite and greater than zero.")
        if (isinstance(self.max_response_bytes, bool) or not isinstance(self.max_response_bytes, int) or
                self.max_response_bytes <= 0):
            raise ValueError("`max_response_bytes` must be a positive integer.")
        headers = _runtime_mapping(
            self.headers,
            name="headers",
            string_values=True,
        )
        if api_key is not None and any(name.lower() == "authorization" for name in headers):
            raise ValueError(
                "Pass authentication through either `api_key` or an "
                "Authorization header, not both.")
        extra_body = _runtime_mapping(
            self.extra_body,
            name="extra_body",
            string_values=False,
        )
        forbidden_body = {"input", "prompt", "input_ids", "stream", "response_format"}
        conflicts = sorted(forbidden_body & set(extra_body))
        if conflicts:
            raise ValueError(
                "`extra_body` cannot override request-owned field(s): " + ", ".join(conflicts) + ".")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "extra_body", extra_body)

    @classmethod
    def from_value(
        cls,
        value: LLMBackendConfig | Mapping[str, Any] | None,
        *,
        backend: str | LLMBackend | None = None,
    ) -> LLMBackendConfig:
        """Normalize a typed config or constructor mapping."""
        if isinstance(value, cls):
            if backend is not None and value.backend is not LLMBackend.coerce(backend):
                raise ValueError(
                    "The explicit `llm_backend` disagrees with "
                    "`llm_backend_config.backend`.")
            return value
        if value is None:
            if backend is None:
                return cls(backend=LLMBackend.NATIVE)
            return cls(backend=backend)
        if not isinstance(value, Mapping):
            raise TypeError("`llm_backend_config` must be an LLMBackendConfig, mapping, "
                            "or None.")
        values = dict(value)
        configured_backend = values.pop("backend", backend)
        if configured_backend is None:
            raise ValueError("`llm_backend_config` requires `backend` when "
                             "`llm_backend` is omitted.")
        if backend is not None and LLMBackend.coerce(configured_backend) is not LLMBackend.coerce(backend):
            raise ValueError("The explicit `llm_backend` disagrees with "
                             "`llm_backend_config.backend`.")
        return cls(
            backend=configured_backend,
            **values,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a diagnostic representation with all credentials redacted."""
        return {
            "backend": self.backend.value,
            "endpoint": self.endpoint,
            "transport": self.transport.value,
            "model": self.model,
            "api_key": ("<redacted>" if self.api_key is not None else None),
            "timeout": self.timeout,
            "headers": {
                name: "<redacted>"
                for name in self.headers
            },
            "extra_body_keys": tuple(sorted(self.extra_body)),
            "max_response_bytes": self.max_response_bytes,
        }

    def request_extra_body(self) -> dict[str, Any]:
        """Return a mutable JSON copy for one request."""
        return _thaw_json(self.extra_body)

    def __repr__(self) -> str:
        return f"LLMBackendConfig({self.to_dict()!r})"


__all__ = [
    "LLMBackend",
    "LLMBackendConfig",
    "LLMBackendTransport",
]
