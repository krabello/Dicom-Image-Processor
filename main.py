from src.utils.file_manager import FileManager
from src.image_processor.image_processor import ImageProcessor
from src.image_processor.strategies import HistogramEqualizationStrategy, GaussianBlurStrategy, CannyEdgeDetectionStrategy
from src.metrics.image_metrics import ImageMetrics
from src.utils.visualizer import Visualizer

def process_all_images():
    metadata_entries = FileManager.load_metadata("metadata.csv")

    all_results = []

    for entry in metadata_entries:
        filename = entry['filename']
        full_path = f"data/raw/{filename}"

        try:
            viewer = FileManager.load_dicom_viewer(full_path)
            original_img = viewer.pixel_array
            base_metadata = FileManager.extract_metadata(viewer, entry)

            # Original metrics
            original_metrics = ImageMetrics.compute_all(original_img)
            all_results.append(FileManager.merge_metadata(base_metadata, "Original", original_metrics))

            # Apply strategies
            for Strategy, label in [
                (HistogramEqualizationStrategy, "Equalized"),
                (GaussianBlurStrategy, "Blurred"),
                (CannyEdgeDetectionStrategy, "Edge Detected"),
            ]:
                processor = ImageProcessor(Strategy())
                processed_img = processor.process(original_img)

                FileManager.save_image(processed_img, filename, label)
                metrics = ImageMetrics.compute_all(processed_img)
                all_results.append(FileManager.merge_metadata(base_metadata, label, metrics))

        except FileNotFoundError:
            print(f"[ERROR] File not found: {full_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

    FileManager.save_metrics_csv(all_results, "image_metrics.csv")
    Visualizer.show_sample_grid("data/raw", "data/processed")

if __name__ == "__main__":
    process_all_images()