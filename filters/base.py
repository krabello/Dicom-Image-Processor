import logging
from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np

# Configure logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageFilter(ABC):
    """
    Abstract base class for image filtering operations.
    
    All image filters must inherit from this class and implement
    the apply() method. The class provides common input validation
    and logging functionality.
    
    Example:
        class MyFilter(ImageFilter):
            def apply(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
                self._validate_input(image)
                # filter implementation
                return processed_image
    """
    
    @abstractmethod
    def apply(self, image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Apply the filter to an input image.

        Args:
            image: 2D grayscale image as numpy array with uint8 dtype

        Returns:
            Processed image as numpy array with uint8 dtype

        Raises:
            ValueError: If image validation fails
        """
        pass

    def _validate_input(self, image: npt.NDArray[np.uint8]) -> None:
        """
        Validate input image requirements.
        
        Args:
            image: Input image to validate

        Raises:
            ValueError: If image is not 2D or is empty
            TypeError: If image has wrong data type
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if image.dtype != np.uint8:
            raise TypeError("Input image must be uint8 type")
            
        if image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image")
            
        if image.size == 0:
            raise ValueError("Input image cannot be empty")
            
        logger.debug(f"Input validation passed for {self.__class__.__name__}")
