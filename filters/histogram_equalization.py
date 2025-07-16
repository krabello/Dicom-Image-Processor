import cv2
import numpy as np
import numpy.typing as npt
from .base import ImageFilter

class HistogramEqualizationFilter(ImageFilter):
    def apply(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Manually perform histogram equalization on a grayscale image.

        Args:
            image (npt.NDArray[np.uint8]): Grayscale image.

        Returns:
            npt.NDArray[np.uint8]: Equalized image.

        Raises:
            ValueError: If image is empty or not 2D.
        """
        if image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image")
        if image.size == 0:
            raise ValueError("Input image cannot be empty")

        histogram, _ = np.histogram(image.flatten(), 256, (0, 256))
        cumulative_distribution = histogram.cumsum()
        masked_distribution = np.ma.masked_equal(cumulative_distribution, 0)
        normalized_distribution = (masked_distribution - masked_distribution.min()) * 255 / (masked_distribution.max() - masked_distribution.min())
        lookup_table = np.ma.filled(normalized_distribution, 0).astype(np.uint8)
        return lookup_table[image]