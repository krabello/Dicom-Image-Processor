import sys
import os
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from typing import Optional, TypedDict
import pydicom
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import RAW_DIR, PROCESSED_DIR, CSV_FILE, ENTROPY_EPSILON, \
IMAGE_SIZE, METRICS_FILENAME, MAX_WORKERS, LOG_FILE, SHOW_GRID
from filters.base import ImageFilter
from filters.histogram_equalization import HistogramEqualizationFilter
from filters.gaussian_blur import GaussianBlurFilter
from filters.canny import CannyEdgeFilter

import time
start = time.time()



# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to write ONLY to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class ImageMetrics(TypedDict):
    mean: float
    std: float
    entropy: float
    michelson_contrast: float


def clean_processed_directory(directory: Path) -> None:
    """Remove all files from the processed directory except .gitkeep."""
    if directory.exists():
        for file in directory.glob('*'):
            if file.name != '.gitkeep':
                try:
                    file.unlink()
                    logger.info(f"Removed: {file}")
                except Exception as e:
                    logger.error(f"Failed to remove {file}: {e}")
    logger.info("Cleaned processed directory")

def load_dicom(filepath: Path) -> Optional[tuple[np.ndarray, dict]]:
    """
    Load and normalize DICOM image

    Args:
        filepath: Path to DICOM file

    Returns:
        tuple: (normalized image array, metadata dictionary)
    """
    try:
        ds = pydicom.dcmread(filepath)
        pixels = ds.pixel_array.astype(float)
        normalized = ((pixels - pixels.min()) / np.ptp(pixels) * 255).astype(np.uint8)
        metadata = {
            'filename': filepath.name,
            'modality': getattr(ds, 'Modality', 'XR')
        }
        return normalized, metadata
    except Exception as e:
        logger.error(f"Error reading {filepath.name}: {e}")
        return None

def process_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Apply all processing steps using strategy pattern."""

    if image.shape != IMAGE_SIZE:
        image = cv2.resize(image, IMAGE_SIZE)

    filters: dict[str, ImageFilter] = {
        'hist_eq': HistogramEqualizationFilter(),
        'gaussian': GaussianBlurFilter(),
        'canny': CannyEdgeFilter()
    }

    processed = {'original': image}
    for name, strategy in filters.items():
        processed[name] = strategy.apply(image)

    return processed

def calculate_metrics(image: np.ndarray) -> ImageMetrics:
    """
    Calculate required image metrics.

    Args:
        image: Input image array (uint8 or compatible format)

    Returns:
        dict: Dictionary containing computed metrics
    """
    img = image.astype(np.float32)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + ENTROPY_EPSILON))
    Imin = np.min(img)
    Imax = np.max(img)
    denom = Imax + Imin
    michelson_contrast = (Imax - Imin) / denom if denom != 0 else 0.0

    return {
        'mean': float(np.mean(img)),
        'std': float(np.std(img)),
        'entropy': float(entropy),
        'michelson_contrast': float(michelson_contrast)
    }

def display_sample_grid(image: np.ndarray, processed_images: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Image Processing Results', fontsize=16)
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for (label, img), (row, col) in zip(processed_images.items(), positions):
        if label != 'original':
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].set_title(label)
            axes[row, col].axis('off')

    plt.tight_layout()
    output_path = PROCESSED_DIR / "summary_grid.png"
    plt.savefig(output_path)
    logger.info(f"Saved grid to {output_path}")
    plt.close()


def process_single_file(dcm_file: Path) -> Optional[list[dict]]:
    try:
        logger.info(f"Processing {dcm_file.name}")
        loaded = load_dicom(dcm_file)
        if not loaded:
            return None
        image, metadata = loaded

        processed = process_image(image)
        save_processed_images(processed, dcm_file.name, PROCESSED_DIR)

        results = []
        for label, img in processed.items():
            metrics = calculate_metrics(img)
            results.append({
                **metadata,
                'processing': label,
                **metrics
            })

        return results
    except Exception as e:
        logger.error(f"Failed to process {dcm_file}: {e}")
        return None


def save_processed_images(processed_images: dict, filename: str, output_dir: Path) -> None:
    base_name = filename.split('.')[0]
    for process_type, img in processed_images.items():
        if process_type == 'original':
            continue
        output_name = f"processed_{base_name}_{process_type}.png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), img)

def process_all_images() -> None:
    clean_processed_directory(PROCESSED_DIR)
    PROCESSED_DIR.mkdir(exist_ok=True)

    dicom_files = list(RAW_DIR.glob('*.dcm'))
    if not dicom_files:
        logger.warning(f"No DICOM files found in {RAW_DIR}. Exiting.")
        return
    logger.info(f"Found {len(dicom_files)} DICOM files in {RAW_DIR}")

    results = []
    first_image = None
    first_processed = None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_file, dcm_file): dcm_file for dcm_file in dicom_files}

        for i, future in enumerate(tqdm(as_completed(future_to_file), total=len(future_to_file))):
            file_results = future.result()
            if file_results:
                results.extend(file_results)

                if i == 0:
                    loaded = load_dicom(future_to_file[future])
                    if loaded is not None:
                        first_image, _ = loaded
                        first_processed = process_image(first_image)

    if SHOW_GRID and first_image is not None and first_processed is not None:
        display_sample_grid(first_image, first_processed)

    if results:
        pd.DataFrame(results).to_csv(PROCESSED_DIR / METRICS_FILENAME, index=False)
        logger.info(f"Successfully processed {len(results) // 4} images")
    else:
        logger.warning("No DICOM files were processed")



def main():
    base_dir = Path(__file__).parent
    process_all_images()
    logger.info(f"Processing completed in {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()
