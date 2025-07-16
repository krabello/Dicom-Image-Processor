"""
Image processing filters for DICOM image enhancement.

This module provides a collection of filters for processing medical images:
- Histogram Equalization
- Gaussian Blur
- Canny Edge Detection

Each filter implements the ImageFilter interface and provides type-safe operations.
"""

from .base import ImageFilter
from .gaussian_blur import GaussianBlurFilter
from .canny import CannyEdgeFilter
from .histogram_equalization import HistogramEqualizationFilter

__all__ = [
    'ImageFilter',
    'GaussianBlurFilter',
    'CannyEdgeFilter',
    'HistogramEqualizationFilter'
]

__version__ = '1.0.0'