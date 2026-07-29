"""Strict, dependency-free reader for numeric NumPy ``.npy`` tensors.

The NPY container is small and documented, so requiring the full NumPy
runtime merely to read a SpeechT5 speaker vector is unnecessary.  This
reader accepts only dense, numeric arrays and materializes them as
PyTorch tensors. Object arrays, structured dtypes, trailing payloads,
and oversized headers are rejected before tensor construction.
"""

from __future__ import annotations

import ast
import re
import struct
import sys
from math import prod
from pathlib import Path
from typing import Any

from voicehub.checkpointing.errors import CheckpointFormatError

_MAGIC = b"\x93NUMPY"
_DESCRIPTOR = re.compile(r"^(?P<byte_order>[<>=|])(?P<kind>[biuf])(?P<size>[1248])$")
_DEFAULT_MAX_HEADER_BYTES = 64 * 1024
_DEFAULT_MAX_TENSOR_BYTES = 1024 * 1024 * 1024


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - package invariant
        raise RuntimeError("Native NPY loading requires PyTorch, VoiceHub's compute runtime.") from error
    return torch


def _read_exact(stream, size: int, *, context: str) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise CheckpointFormatError(
            f"NPY file ended while reading {context}: expected {size} bytes, "
            f"found {len(payload)}.")
    return payload


def _parse_header(
    encoded: bytes,
    *,
    encoding: str,
) -> tuple[str, bool, tuple[int, ...]]:
    try:
        header = ast.literal_eval(encoded.decode(encoding).strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise CheckpointFormatError(f"NPY header is not a valid literal dictionary: {error}.") from error
    if not isinstance(header, dict):
        raise CheckpointFormatError("NPY header must be a dictionary.")
    required = {"descr", "fortran_order", "shape"}
    if set(header) != required:
        raise CheckpointFormatError(
            "NPY header must contain exactly `descr`, `fortran_order`, and "
            "`shape`.")
    descriptor = header["descr"]
    fortran_order = header["fortran_order"]
    raw_shape = header["shape"]
    if not isinstance(descriptor, str):
        raise CheckpointFormatError("NPY `descr` must be a string.")
    if not isinstance(fortran_order, bool):
        raise CheckpointFormatError("NPY `fortran_order` must be a boolean.")
    if (not isinstance(raw_shape, tuple) or
            any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0
                for dimension in raw_shape)):
        raise CheckpointFormatError("NPY `shape` must be a tuple of non-negative integers.")
    return descriptor, fortran_order, tuple(raw_shape)


def _dtype(descriptor: str) -> tuple[Any, int]:
    match = _DESCRIPTOR.fullmatch(descriptor)
    if match is None:
        raise CheckpointFormatError(
            f"Unsupported NPY dtype descriptor {descriptor!r}; VoiceHub "
            "accepts dense boolean, integer, and floating-point arrays.")
    byte_order = match.group("byte_order")
    kind = match.group("kind")
    size = int(match.group("size"))
    if byte_order == ">" and size > 1:
        raise CheckpointFormatError("Big-endian multi-byte NPY tensors are not supported.")
    if byte_order == "=" and sys.byteorder != "little" and size > 1:
        raise CheckpointFormatError("Native-endian NPY tensors on big-endian hosts are not supported.")
    if byte_order == "|" and size > 1:
        raise CheckpointFormatError(
            "Byte-order-independent NPY descriptors may only use one-byte "
            "elements.")
    torch = _torch()
    mapping = {
        ("b", 1): torch.bool,
        ("i", 1): torch.int8,
        ("i", 2): torch.int16,
        ("i", 4): torch.int32,
        ("i", 8): torch.int64,
        ("u", 1): torch.uint8,
        ("u", 2): getattr(torch, "uint16", None),
        ("u", 4): getattr(torch, "uint32", None),
        ("u", 8): getattr(torch, "uint64", None),
        ("f", 2): torch.float16,
        ("f", 4): torch.float32,
        ("f", 8): torch.float64,
    }
    dtype = mapping.get((kind, size))
    if dtype is None:
        raise CheckpointFormatError(
            f"PyTorch does not expose the NPY dtype {descriptor!r} on this "
            "runtime.")
    return dtype, size


def load_numpy_tensor(
    path: str | Path,
    *,
    device: Any = "cpu",
    dtype: Any | None = None,
    max_header_bytes: int = _DEFAULT_MAX_HEADER_BYTES,
    max_tensor_bytes: int = _DEFAULT_MAX_TENSOR_BYTES,
):
    """Load one numeric NPY array as a PyTorch tensor.

    The implementation supports NPY versions 1, 2, and 3.  It never
    enables pickle and therefore cannot load object arrays.
    """
    if max_header_bytes <= 0 or max_tensor_bytes <= 0:
        raise ValueError("NPY parser limits must be positive.")
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"NPY file was not found: {source}")
    with source.open("rb") as stream:
        if _read_exact(stream, len(_MAGIC), context="magic") != _MAGIC:
            raise CheckpointFormatError("File does not contain the NPY magic.")
        major, minor = _read_exact(stream, 2, context="version")
        if major == 1:
            header_size = struct.unpack(
                "<H",
                _read_exact(stream, 2, context="header length"),
            )[0]
            encoding = "latin1"
        elif major in {2, 3}:
            header_size = struct.unpack(
                "<I",
                _read_exact(stream, 4, context="header length"),
            )[0]
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise CheckpointFormatError(f"Unsupported NPY version {major}.{minor}.")
        if header_size <= 0 or header_size > max_header_bytes:
            raise CheckpointFormatError(
                f"NPY header length {header_size} is outside the allowed "
                f"range 1..{max_header_bytes}.")
        encoded_header = _read_exact(
            stream,
            header_size,
            context="header",
        )
        if not encoded_header.endswith(b"\n"):
            raise CheckpointFormatError("NPY header must end with a newline.")
        descriptor, fortran_order, shape = _parse_header(
            encoded_header,
            encoding=encoding,
        )
        source_dtype, element_size = _dtype(descriptor)
        element_count = prod(shape, start=1)
        payload_size = element_count * element_size
        if payload_size > max_tensor_bytes:
            raise CheckpointFormatError(
                f"NPY tensor payload is {payload_size} bytes; the limit is "
                f"{max_tensor_bytes}.")
        payload = _read_exact(stream, payload_size, context="tensor payload")
        if stream.read(1):
            raise CheckpointFormatError("NPY file contains trailing bytes after its tensor payload.")

    torch = _torch()
    if element_count == 0:
        tensor = torch.empty(shape, dtype=source_dtype)
    else:
        tensor = torch.frombuffer(
            bytearray(payload),
            dtype=source_dtype,
            count=element_count,
        ).clone()
        if fortran_order and len(shape) > 1:
            reversed_dimensions = tuple(reversed(shape))
            tensor = tensor.reshape(reversed_dimensions).permute(*reversed(range(len(shape))))
        else:
            tensor = tensor.reshape(shape)
        tensor = tensor.contiguous()
    if dtype is not None or str(device) != "cpu":
        tensor = tensor.to(device=device, dtype=dtype or tensor.dtype)
    return tensor


__all__ = ["load_numpy_tensor"]
