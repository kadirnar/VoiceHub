"""Request-local streaming sessions for speech inference."""

from __future__ import annotations

from threading import RLock
from typing import Any

import torch

from voicehub.audio import load_audio


class BufferedSpeechSession:
    """Safe fallback session for models without incremental decoding.

    Chunks are normalized immediately and retained only inside this
    session. ``flush`` performs one regular offline call. Providers with
    true cache-aware streaming can override
    ``PreTrainedAudioModel.stream`` with the same push/flush/reset/close
    contract.
    """

    def __init__(
        self,
        model,
        *,
        sampling_rate: int,
        inference_kwargs: dict[str, Any] | None = None,
    ):
        if (isinstance(sampling_rate, bool) or not isinstance(sampling_rate, int) or sampling_rate <= 0):
            raise ValueError("Streaming `sampling_rate` must be a positive integer.")
        self.model = model
        self.sampling_rate = sampling_rate
        self.inference_kwargs = dict(inference_kwargs or {})
        self._chunks: list[Any] = []
        self._result = None
        self._closed = False
        self._lock = RLock()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def push(self, audio_chunk: Any):
        """Append one chunk; incremental providers may return events
        instead."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot push audio to a closed stream.")
            if self._result is not None:
                raise RuntimeError("Reset the stream before pushing after flush().")
            chunk = load_audio(
                audio_chunk,
                sampling_rate=self.sampling_rate,
                target_sampling_rate=self.sampling_rate,
            )
            self._chunks.append(chunk.waveform)
        return None

    def flush(self):
        """Finalize buffered audio exactly once and return task output."""
        with self._lock:
            if self._result is not None:
                return self._result
            if not self._chunks:
                raise ValueError("Cannot flush a stream that has no audio.")
            waveform = torch.cat(
                tuple(torch.as_tensor(chunk) for chunk in self._chunks),
                dim=-1,
            )
            self._result = self.model(
                waveform,
                sampling_rate=self.sampling_rate,
                **self.inference_kwargs,
            )
            return self._result

    def reset(self) -> None:
        """Discard request-local state and make the session reusable."""
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot reset a closed stream.")
            self._chunks.clear()
            self._result = None

    def close(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed stream.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
