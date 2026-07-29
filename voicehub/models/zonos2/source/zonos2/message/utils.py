from __future__ import annotations

from typing import Any, Dict, Type

import torch


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        materialized = self.detach().to(device="cpu").contiguous()
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = bytes(materialized.untyped_storage())
        serialized["dtype"] = str(materialized.dtype)
        serialized["shape"] = list(materialized.shape)
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        assert isinstance(buffer, bytes)
        dtype = getattr(torch, dtype_str, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported serialized tensor dtype {dtype_str!r}.")
        tensor = torch.frombuffer(bytearray(buffer), dtype=dtype)
        shape = data.get("shape")
        if shape is not None:
            if (
                not isinstance(shape, list)
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 0
                    for dimension in shape
                )
            ):
                raise ValueError("Serialized tensor shape must contain non-negative integers.")
            expected = 1
            for dimension in shape:
                expected *= dimension
            if tensor.numel() != expected:
                raise ValueError(
                    "Serialized tensor byte length does not match its shape."
                )
            tensor = tensor.reshape(shape)
        return tensor.clone()

    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
