"""Small dependency-free HTTP transport for LLM serving backends."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from voicehub.errors import LLMBackendRequestError
from voicehub.json_utils import parse_json_value
from voicehub.llm_serving.configuration import LLMBackendConfig


class _RejectRedirects(HTTPRedirectHandler):
    """Fail closed so authentication headers never cross an origin."""

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        del req, fp, code, msg, headers, newurl
        return None


_NO_REDIRECT_OPENER = build_opener(_RejectRedirects())


def urlopen(request, *, timeout):
    """Open one request without following credential-bearing redirects."""
    return _NO_REDIRECT_OPENER.open(
        request,
        timeout=timeout,
    )


@dataclass(frozen=True, slots=True)
class HTTPBackendResponse:
    """Bounded response body and normalized HTTP metadata."""

    body: bytes
    headers: Mapping[str, str]
    status: int

    def header(self, name: str, default: str | None = None) -> str | None:
        return self.headers.get(name.lower(), default)


def join_endpoint(endpoint: str, route: str) -> str:
    """Join a server base URL with one known API route."""
    base = endpoint.rstrip("/")
    normalized_route = "/" + route.lstrip("/")
    if base.endswith(normalized_route):
        return base
    if base.endswith("/v1") and normalized_route.startswith("/v1/"):
        return base + normalized_route[3:]
    return base + normalized_route


class HTTPBackendClient:
    """Authenticated JSON POST client with bounded response reads."""

    def __init__(self, config: LLMBackendConfig):
        self.config = config

    def _headers(self, *, accept: str) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
            "User-Agent": "voicehub-llm-serving/1",
            **dict(self.config.headers),
        }
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def post_json(
        self,
        route: str,
        payload: Mapping[str, Any],
        *,
        accept: str,
    ) -> HTTPBackendResponse:
        if self.config.endpoint is None:
            raise LLMBackendRequestError("External backend configuration has no endpoint.")
        try:
            encoded = json.dumps(
                dict(payload),
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise LLMBackendRequestError("The backend request is not finite JSON.") from error
        url = join_endpoint(
            self.config.endpoint,
            route,
        )
        request = Request(
            url,
            data=encoded,
            headers=self._headers(accept=accept),
            method="POST",
        )
        try:
            with urlopen(
                    request,
                    timeout=self.config.timeout,
            ) as response:
                status = int(getattr(response, "status", 200))
                raw_headers = getattr(response, "headers", {})
                headers = {str(name).lower(): str(value) for name, value in raw_headers.items()}
                declared_size = headers.get("content-length")
                if declared_size is not None:
                    try:
                        size = int(declared_size)
                    except ValueError as error:
                        raise LLMBackendRequestError(
                            "The backend returned an invalid Content-Length "
                            "header.") from error
                    if size < 0:
                        raise LLMBackendRequestError(
                            "The backend returned a negative Content-Length "
                            "header.")
                    if size > self.config.max_response_bytes:
                        raise LLMBackendRequestError(
                            "The backend response exceeds "
                            f"{self.config.max_response_bytes} bytes.")
                body = response.read(self.config.max_response_bytes + 1)
        except HTTPError as error:
            try:
                error.close()
            except OSError:
                # Preserve the sanitized request error even if a broken
                # response stream also fails during cleanup.
                pass
            raise LLMBackendRequestError(
                f"{self.config.backend.value} returned HTTP {error.code} "
                f"for {route}.") from error
        except URLError as error:
            reason = getattr(error, "reason", None)
            reason_name = (type(reason).__name__ if reason is not None else type(error).__name__)
            raise LLMBackendRequestError(
                f"Could not reach the {self.config.backend.value} server "
                f"for {route} ({reason_name}).") from error
        except TimeoutError as error:
            raise LLMBackendRequestError(
                f"The {self.config.backend.value} request to {route} timed "
                "out.") from error
        if len(body) > self.config.max_response_bytes:
            raise LLMBackendRequestError(
                "The backend response exceeds "
                f"{self.config.max_response_bytes} bytes.")
        if not 200 <= status < 300:
            raise LLMBackendRequestError(
                f"{self.config.backend.value} returned HTTP {status} "
                f"for {route}.")
        return HTTPBackendResponse(
            body=body,
            headers=headers,
            status=status,
        )

    def post_json_document(
        self,
        route: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        response = self.post_json(
            route,
            payload,
            accept="application/json",
        )
        try:
            document = parse_json_value(
                response.body,
                source=f"{self.config.backend.value} response for {route}",
            )
        except (UnicodeDecodeError, ValueError) as error:
            raise LLMBackendRequestError(
                f"{self.config.backend.value} returned malformed JSON "
                f"for {route}: {error}.") from error
        if not isinstance(document, dict):
            raise LLMBackendRequestError(
                f"{self.config.backend.value} returned a non-object JSON "
                f"response for {route}.")
        return document


__all__ = [
    "HTTPBackendClient",
    "HTTPBackendResponse",
    "join_endpoint",
]
