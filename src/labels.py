"""
labels.py

CNN class list, YOLO-to-CNN name mapping, and paths to local weights and video.

All paths are derived from this file's location. No drive letters are used.
"""
from __future__ import annotations

from pathlib import Path

# One level up from src/ is the project root (Sentinel-X/).
_ROOT = Path(__file__).resolve().parent.parent

# Local data folder - weights and video live here. Not in the repository.
DATA_DIR = _ROOT / "Data"
YOLO_FINAL_DIR = DATA_DIR / "YOLO-Models-Final"
EFFNET_PATH = DATA_DIR / "EffNet_b0.pt"
VIDEO_PATH = DATA_DIR / "video" / "combined_drone.mp4"

# The 10 EfficientNet output classes in the order stored in the checkpoint.
CNN_LABELS: tuple[str, ...] = (
    "Artillery",
    "M. Rocket Launcher",
    "Missile",
    "Radar",
    "Soldier",
    "camouflage_soldier",
    "military_aircraft",
    "military_vehicle",
    "military_warship",
    "trench",
)

# Maps every YOLO raw class name onto the shared CNN label.
# Lower-case "soldier" is the MV camouflage class. "Soldier" (capital) is the KIIT class.
# "Vehicle" was removed from the CNN set and has no entry here.
YOLO_TO_CNN: dict[str, str] = {
    "Artilary":                  "Artillery",
    "Artillery":                 "Artillery",
    "Missile":                   "Missile",
    "Radar":                     "Radar",
    "M. Rocket Launcher":        "M. Rocket Launcher",
    "Soldier":                   "Soldier",
    "soldier":                   "camouflage_soldier",
    "camouflage_soldier":        "camouflage_soldier",
    "Tank":                      "military_vehicle",
    "tank":                      "military_vehicle",
    "armoured personnel carrier":"military_vehicle",
    "military_vehicle":          "military_vehicle",
    "military_aircraft":         "military_aircraft",
    "military_warship":          "military_warship",
    "trench":                    "trench",
}

# The three detector weight files the window loads from YOLO_FINAL_DIR.
YOLO_FILES: tuple[str, ...] = (
    "KIIT-MiTA-yolo26s.pt",
    "MV-yolo26s.pt",
    "military-yolov8n.pt",
)


def cnn_label(raw_name: str) -> str | None:
    """Return the shared CNN name for a raw YOLO class, or None when that class was dropped."""
    mapped = YOLO_TO_CNN.get(raw_name)
    # Return None for any name not in the 10-label CNN list (e.g. "Vehicle", "weapon").
    if mapped not in CNN_LABELS:
        return None
    return mapped
