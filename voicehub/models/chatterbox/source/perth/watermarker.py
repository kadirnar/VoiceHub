import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class WatermarkingException(Exception):
    """Exception raised for errors in the watermarking process."""

    pass


class WatermarkerBase(ABC):
    """
    Base class for all audio watermarking algorithms.

    This abstract class defines the interface that all watermarking implementations
    must follow, providing methods for watermark application and extraction.
    """

    @abstractmethod
    def apply_watermark(
        self,
        wav: np.ndarray,
        watermark: Optional[np.ndarray] = None,
        sample_rate: int = 44100,
        **kwargs,
    ) -> np.ndarray:
        """
        Apply a watermark to an audio signal.

        Args:
            wav: Input audio signal as numpy array
            watermark: Optional watermark data to embed. If None, a default watermark may be generated.
            sample_rate: Sample rate of the audio signal in Hz
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Watermarked audio signal as numpy array

        Raises:
            WatermarkingException: If watermarking fails
        """
        raise NotImplementedError()

    @abstractmethod
    def get_watermark(
        self,
        watermarked_wav: np.ndarray,
        sample_rate: int = 44100,
        watermark_length: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Extract a watermark from a watermarked audio signal.

        Args:
            watermarked_wav: Watermarked audio signal as numpy array
            sample_rate: Sample rate of the audio signal in Hz
            watermark_length: Optional expected length of the watermark
            **kwargs: Additional algorithm-specific parameters

        Returns:
            Extracted watermark data as numpy array

        Raises:
            WatermarkingException: If watermark extraction fails
        """
        raise NotImplementedError()

    def verify_compatibility(self, wav: np.ndarray, sample_rate: int) -> bool:
        """
        Verify if the audio is compatible with this watermarking method.

        Args:
            wav: Input audio signal as numpy array
            sample_rate: Sample rate of the audio signal in Hz

        Returns:
            True if the audio is compatible, False otherwise
        """
        return True
