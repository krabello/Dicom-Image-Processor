# DICOM Image Processing & Analysis

This project processes a dataset of DICOM and PNG medical images by applying multiple image filters and computing quantitative metrics.

---

## Features

- Load and parse image metadata from a CSV file
- Apply modular image filters:
  - Manual Histogram Equalization
  - Gaussian Blur (OpenCV)
  - Canny Edge Detection
- Compute image metrics:
  - Mean
  - Standard Deviation
  - Entropy
  - Michelson Contrast
- Save processed images to disk with identifiable filenames
- Export all results to a CSV file
- Visualize selected image pipelines using a 2x3 matplotlib grid
