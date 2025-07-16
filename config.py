from pathlib import Path

BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CSV_FILE = RAW_DIR / "CS390_Midterm_xray_chest.csv"
METRICS_FILENAME = "metrics.csv"
LOGS_DIRECTORY = BASE_DIR / "logs"
LOG_FILE = LOGS_DIRECTORY / "app.log"
MAX_WORKERS = 4
ENTROPY_EPSILON = 1e-7
GAUSSIAN_KERNEL = (5, 5)
IMAGE_SIZE = (256, 256)
SHOW_GRID = False