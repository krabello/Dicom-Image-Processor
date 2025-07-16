import cv2
import numpy as np
import numpy.typing as npt
from .base import ImageFilter

class CannyEdgeFilter(ImageFilter):
    def __init__(self, low_threshold: float = 100.0, high_threshold: float = 200.0):
        """
        Initialize Canny edge detector with thresholds.

        Args:
            low_threshold (float): Lower threshold for edge detection
            high_threshold (float): Higher threshold for edge detection
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def apply(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Apply Canny edge detection to a grayscale image.

        Args:
            image (npt.NDArray[np.uint8]): Input grayscale image

        Returns:
            npt.NDArray[np.uint8]: Edge detected image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(
            blurred, 
            threshold1=self.low_threshold,
            threshold2=self.high_threshold
        )
        
        # Convert OpenCV output to numpy array
        return np.array(edges, dtype=np.uint8)
