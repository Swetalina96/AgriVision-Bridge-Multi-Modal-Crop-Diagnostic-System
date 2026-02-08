from pathlib import Path
import sys

from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent

# YOLO model path (CHANGE filename if different)
DETECTION_MODEL = ROOT_DIR / "models" / "best.pt"

# Default images (optional but recommended)
DEFAULT_IMAGE = ROOT_DIR / "assets" / "default.jpg"
DEFAULT_DETECT_IMAGE = ROOT_DIR / "assets" / "default_detect.jpg"

# -------------------------------
# PATH SETUP
# -------------------------------
FILE = Path(__file__).resolve()
ROOT = FILE.parent

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# -------------------------------
# DIRECTORIES
# -------------------------------
WEIGHTS_DIR = ROOT / "weights"
IMAGES_DIR = ROOT / "images"

# -------------------------------
# DEFAULT IMAGES
# -------------------------------
DEFAULT_IMAGE = IMAGES_DIR / "office_4.jpg"
DEFAULT_DETECT_IMAGE = IMAGES_DIR / "office_4_detected.jpg"

# -------------------------------
# MODEL PATHS
# -------------------------------
DETECTION_MODEL = WEIGHTS_DIR / "yolo.pt"

