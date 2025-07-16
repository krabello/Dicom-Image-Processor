import cv2
import numpy as np
import numpy.typing as npt
from .base import ImageFilter

class GaussianBlurFilter(ImageFilter):
    def __init__(self, kernel_size: int = 5, sigma: float = 0.0):
        """
        Initialize the GaussianBlurFilter.

        Args:
            kernel_size (int): Size of the Gaussian kernel (must be odd and positive).
            sigma (float): Standard deviation in X and Y direction.

        Raises:
            ValueError: If kernel_size is even or not positive.
        """
        if not isinstance(kernel_size, int):
            raise TypeError("kernel_size must be an integer")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number")
        if not isinstance(sigma, (int, float)):
            raise TypeError("sigma must be a number")
        
        self.kernel_size = kernel_size
        self.sigma = float(sigma)

    def apply(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Apply Gaussian blur to the input image.

        Args:
            image (npt.NDArray[np.uint8]): Input grayscale image.

        Returns:
            npt.NDArray[np.uint8]: Blurred image.

        Raises:
            ValueError: If image is empty or not 2D.
        """
        if image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image")
        if image.size == 0:
            raise ValueError("Input image cannot be empty")

        blurred = cv2.GaussianBlur(
            image, 
            (self.kernel_size, self.kernel_size), 
            self.sigma
        )
        # Convert OpenCV output to numpy array
        return np.array(blurred, dtype=np.uint8)
