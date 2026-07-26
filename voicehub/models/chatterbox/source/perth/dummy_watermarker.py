import numpy as np
from typing import Optional

from .watermarker import WatermarkerBase


class DummyWatermarker(WatermarkerBase):
    """
    A dummy watermarker for testing and demonstration purposes.

    This watermarker doesn't actually embed or extract real watermarks,
    but serves as a placeholder implementation for testing the framework.
    """

    def apply_watermark(
        self,
        wav: np.ndarray,
        watermark: Optional[np.ndarray] = None,
        sample_rate: int = 44100,
        **kwargs,
    ) -> np.ndarray:
        """
        Simulates applying a watermark by simply rounding the audio signal.

        Args:
            wav: Input audio signal as numpy array
            watermark: Ignored in this implementation
            sample_rate: Ignored in this implementation
            **kwargs: Additional ignored parameters

        Returns:
            The input audio with minimal modification (rounded to 5 decimal places)
        """
        return wav.round(5)

    def get_watermark(
        self,
        watermarked_wav: np.ndarray,
        sample_rate: int = 44100,
        watermark_length: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Simulates extracting a watermark by returning random data.

        Args:
            watermarked_wav: Watermarked audio signal as numpy array
            sample_rate: Ignored in this implementation
            watermark_length: Length of the dummy watermark to generate
            **kwargs: Additional ignored parameters

        Returns:
            A random binary watermark of specified length or default 32 bits
        """
        length = watermark_length if watermark_length is not None else 32
        return np.random.randint(0, 2, size=length).astype(np.float32)
