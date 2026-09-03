import os

# Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Settings
DEFAULT_MODEL_PATH = "yolov8n.pt"  # Lightweight nano model for fast CPU/GPU inference
CONFIDENCE_THRESHOLD = 0.35

# Target COCO Classes for Smart Campus Management
# COCO Indices: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Detection Colors (BGR format for OpenCV)
CLASS_COLORS = {
    "Person": (255, 120, 0),     # Vibrant Blue/Orange
    "Car": (0, 215, 255),        # Cyan/Yellow
    "Motorcycle": (0, 255, 128), # Spring Green
    "Bicycle": (255, 0, 128),    # Pink
    "Bus": (0, 165, 255),        # Orange
    "Truck": (128, 0, 255)       # Purple
}

DEFAULT_COLOR = (200, 200, 200)

# Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure required directories exist
for path in [DATA_DIR, VIDEOS_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)
