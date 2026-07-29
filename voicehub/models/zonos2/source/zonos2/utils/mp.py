"""Optional ZeroMQ queues with a VoiceHub-owned wire codec.

ZeroMQ is a serving transport, not part of the Zonos2 model graph. Importing
the architecture must therefore never require ``pyzmq``. Queue construction
loads it lazily and reports a focused error when the optional distributed
serving strategy is selected.

The original source used MessagePack for small typed dictionaries. VoiceHub
uses a bounded JSON codec with explicit Base64 binary values instead, keeping
the default runtime independent from MessagePack and avoiding unsafe pickle
deserialization.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Mapping
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_BINARY_TYPE = "voicehub.binary.v1"
_MAX_MESSAGE_BYTES = 256 * 1024 * 1024


def _load_zmq() -> Any:
    try:
        import zmq
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Zonos2 distributed serving requires the optional `pyzmq` "
            "transport. Core inference and training do not require it."
        ) from error
    return zmq


def _load_zmq_asyncio() -> tuple[Any, Any]:
    zmq = _load_zmq()
    try:
        import zmq.asyncio
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Zonos2 asynchronous distributed serving requires `pyzmq` "
            "with asyncio support."
        ) from error
    return zmq, zmq.asyncio


def _json_compatible(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "__type__": _BINARY_TYPE,
            "data": base64.b64encode(value).decode("ascii"),
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Zonos2 wire dictionaries require string keys.")
            converted[key] = _json_compatible(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise TypeError(
        "Zonos2 queue encoders must return JSON-compatible values or bytes; "
        f"found {type(value).__name__}."
    )


def _restore_json(value: Any) -> Any:
    if isinstance(value, list):
        return [_restore_json(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__type__", "data"} and value["__type__"] == _BINARY_TYPE:
            encoded = value["data"]
            if not isinstance(encoded, str):
                raise ValueError("Encoded Zonos2 binary data must be text.")
            try:
                return base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError) as error:
                raise ValueError("Invalid Base64 data in Zonos2 message.") from error
        return {key: _restore_json(item) for key, item in value.items()}
    return value


def _pack_message(value: Mapping[str, Any]) -> bytes:
    if not isinstance(value, Mapping):
        raise TypeError("Zonos2 queue encoders must return a mapping.")
    payload = json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Zonos2 message is {len(payload)} bytes; the limit is "
            f"{_MAX_MESSAGE_BYTES}."
        )
    return payload


def _unpack_message(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    if len(payload) > _MAX_MESSAGE_BYTES:
        raise ValueError(
            f"Zonos2 message is {len(payload)} bytes; the limit is "
            f"{_MAX_MESSAGE_BYTES}."
        )
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Zonos2 queue received invalid UTF-8 JSON.") from error
    restored = _restore_json(decoded)
    if not isinstance(restored, dict):
        raise ValueError("Zonos2 queue messages must decode to a dictionary.")
    return restored


class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Mapping[str, Any]],
    ):
        zmq = _load_zmq()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put(self, obj: T):
        self.socket.send(_pack_message(self.encoder(obj)), copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Mapping[str, Any]],
    ):
        zmq, zmq_asyncio = _load_zmq_asyncio()
        self.context = zmq_asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        await self.socket.send(_pack_message(self.encoder(obj)), copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[dict[str, Any]], T],
    ):
        zmq = _load_zmq()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    def get(self) -> T:
        return self.decoder(_unpack_message(self.socket.recv()))

    def get_raw(self) -> bytes:
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        return self.decoder(_unpack_message(raw))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[dict[str, Any]], T],
    ):
        zmq, zmq_asyncio = _load_zmq_asyncio()
        self.context = zmq_asyncio.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder

    async def get(self) -> T:
        return self.decoder(_unpack_message(await self.socket.recv()))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Mapping[str, Any]],
    ):
        zmq = _load_zmq()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        self.socket.send(_pack_message(self.encoder(obj)), copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[dict[str, Any]], T],
    ):
        zmq = _load_zmq()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.decoder = decoder

    def get(self) -> T:
        return self.decoder(_unpack_message(self.socket.recv()))

    def empty(self) -> bool:
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


__all__ = [
    "ZmqAsyncPullQueue",
    "ZmqAsyncPushQueue",
    "ZmqPubQueue",
    "ZmqPullQueue",
    "ZmqPushQueue",
    "ZmqSubQueue",
]
