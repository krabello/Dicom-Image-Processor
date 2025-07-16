import os
import cv2
import csv
import pandas as pd
from typing import List, Dict
from src.core.dicom_viewer import DicomViewer
import matplotlib.pyplot as plt

class FileManager:

    @staticmethod
    def load_dicom_viewer(filepath: str) -> DicomViewer:
        """
        Load and return a DicomViewer object.
        """
        return DicomViewer(filepath)

    @staticmethod
    def load_metadata(csv_path: str) -> List[Dict]:
        """
        Loads metadata from a CSV file.
        """
        return pd.read_csv(csv_path).to_dict(orient='records')

    @staticmethod
    def extract_metadata(viewer: DicomViewer, fallback: Dict) -> Dict:
        """
        Extract DICOM metadata or fallback to CSV metadata if unavailable.
        """
        return {
            "filename": os.path.basename(viewer.filepath),
            "study_date": getattr(viewer.ds, 'StudyDate', fallback.get('study_date', 'N/A')),
            "modality": getattr(viewer.ds, 'Modality', fallback.get('modality', 'N/A')),
            "sop_class_uid": getattr(viewer.ds, 'SOPClassUID', fallback.get('sop_class_uid', 'N/A')),
        }

    @staticmethod
    def merge_metadata(base: Dict, image_type: str, metrics: Dict) -> Dict:
        """
        Combine base metadata with image type and computed metrics.
        """
        merged = base.copy()
        merged["type"] = image_type
        return {**merged, **metrics}

    @staticmethod
    def save_metrics_csv(data: List[Dict], output_path: str) -> None:
        """
        Save all computed metrics and metadata to a CSV.
        """
        if not data:
            print("No data to write.")
            return
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    @staticmethod
    def save_image(image, original_filename: str, label: str) -> None:
        """
        Save a processed image to the data/processed directory.
        """
        name = os.path.splitext(os.path.basename(original_filename))[0]
        filename = f"processed_{name}_{label.replace(' ', '_')}.png"
        out_path = os.path.join("data", "processed", filename)

        # Normalize and convert to uint8
        if image.dtype != 'uint8':
            norm_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
            image = norm_img.astype('uint8')

        cv2.imwrite(out_path, image)
        print(f"[INFO] Saved processed image: {out_path}")